import os
import sys
import pandas as pd
import numpy as np
import yt_dlp
import librosa
import torch
from transformers import ClapModel, ClapProcessor
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MUSIC_ENRICHED_DATA_PATH, MUSIC_TRAINING_DATA_PATH

TEMP_AUDIO_DIR = "temp_audio_cache"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
BATCH_SIZE = 25

print("Loading Models to RTX 3050...")
audio_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to("cuda")
audio_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
text_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

def get_audio_embedding(track_name, artist_name, track_id):
    query = f"{track_name} {artist_name} official audio"
    audio_path = os.path.join(TEMP_AUDIO_DIR, f"{track_id}.m4a")
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'outtmpl': audio_path,
        'noplaylist': True,
        'quiet': True,
        'default_search': 'ytsearch1:',
        'match_filter': yt_dlp.utils.match_filter_func("duration < 480"), 
    }
    
    embedding = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])
            
        if os.path.exists(audio_path):
            audio_array, _ = librosa.load(audio_path, sr=48000, mono=True)
            inputs = audio_processor(audios=audio_array, sampling_rate=48000, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = audio_model.get_audio_features(**inputs)
            embedding = outputs.cpu().numpy()[0].tolist()
    except Exception:
        pass
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    return embedding

def engineer_features():
    df = pd.read_csv(MUSIC_ENRICHED_DATA_PATH)
    
    # 1. NLP Lyrics Embeddings (Fast, happens all at once on the GPU)
    print("Generating Lyric Embeddings...")
    lyric_embeddings = text_model.encode(df['lyrics'].tolist(), show_progress_bar=True)
    for i in range(384):
        df[f'nlp_dim_{i}'] = lyric_embeddings[:, i]

    # Initialize audio columns
    for i in range(512): 
        df[f'clap_dim_{i}'] = np.nan

    # 2. Audio Embeddings (Slower, requires downloading)
    updates = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Audio"):
        if pd.isna(row['clap_dim_0']):
            emb = get_audio_embedding(row['track_name'], row['artist_name'], row['track_id'])
            if emb:
                for i in range(512):
                    df.at[index, f'clap_dim_{i}'] = emb[i]
                
                updates += 1
                if updates % BATCH_SIZE == 0:
                    df.to_csv(MUSIC_TRAINING_DATA_PATH, index=False)

    df = df.dropna(subset=['clap_dim_0']).reset_index(drop=True)
    df.to_csv(MUSIC_TRAINING_DATA_PATH, index=False)
    print(f"Features Engineered! 896-dimension vectors saved to {MUSIC_TRAINING_DATA_PATH}")

if __name__ == "__main__":
    engineer_features()
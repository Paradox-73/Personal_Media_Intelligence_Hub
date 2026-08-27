import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.music.model_training import Oracle

sys.path.append(str(Path(__file__).resolve().parent.parent))  # app/ dir for shared helpers
from mobile import apply_mobile_css
st.set_page_config(page_title="Music Oracle", page_icon="🎧", layout="wide")
apply_mobile_css()

st.title("🎧 The Music Oracle")
st.markdown(
    "Music isn't rated — it's **clustered**. Explore your taste *personas*, see which "
    "one any track belongs to, and find the songs most similar to the ones you love."
)

@st.cache_resource
def get_oracle():
    try:
        return Oracle()
    except Exception as e:
        st.error(f"Failed to initialize Oracle: {e}")
        return None

oracle = get_oracle()

if oracle is None:
    st.warning("Oracle not ready. Run the music pipeline (feature_engineering.py + profile_builder.py) first.")
    st.stop()

# Shared track picker data
track_options = oracle.df[['track_id', 'name', 'artists']].copy()
track_options['track_id'] = track_options['track_id'].astype(str)
track_options['name'] = track_options['name'].fillna('Unknown').astype(str)
track_options['artists'] = track_options['artists'].fillna('Unknown').astype(str)
track_options['display'] = track_options['name'] + " — " + track_options['artists']
track_options = track_options.drop_duplicates(subset='track_id')
track_ids = track_options['track_id'].tolist()
label_map = dict(zip(track_options['track_id'], track_options['display']))

tab1, tab2, tab3 = st.tabs(["🧬 Taste Personas", "🎼 Find Similar", "🌱 Discover New"])

with tab1:
    st.subheader("Your Taste Personas")
    st.markdown(
        "Your library clustered into distinct listening modes (KMeans on the audio + "
        "genre + lyric + semantic feature space). No stars — just *how your taste groups*."
    )
    personas = oracle.taste_personas()
    if not personas:
        st.info("No taste profile found yet. Build it with `python -m src.music.profile_builder`.")
    else:
        cols = st.columns(min(3, len(personas)))
        for k, meta in enumerate(personas):
            with cols[k % len(cols)]:
                st.markdown(f"#### {meta['label']}")
                st.caption(f"{meta['size']} tracks")
                if meta.get('genres'):
                    st.write("**Genres:** " + ", ".join(meta['genres'][:5]))
                if meta.get('exemplars'):
                    st.write("**Anchors:** " + "; ".join(meta['exemplars'][:3]))

        st.divider()
        st.markdown("**Which persona does a track belong to?**")
        sel_cluster = st.selectbox("Pick a track", options=track_ids,
                                   format_func=lambda x: label_map.get(str(x), str(x)),
                                   key="cluster_select")
        if st.button("Find its persona"):
            info = oracle.cluster_of(sel_cluster)
            if info is None:
                st.info("No taste profile available.")
            else:
                st.success(f"**{info['label']}**  ·  match {info['similarity']:.2f}")
                if info.get('genres'):
                    st.write("Signature genres: " + ", ".join(info['genres']))

with tab2:
    st.subheader("Similar Tracks in Your Library")
    st.markdown("Cosine similarity in the full feature space — the closest songs to your seed.")

    selected_track_id_sim = st.selectbox("Select a Seed Track",
                                         options=track_ids,
                                         format_func=lambda x: label_map.get(str(x), str(x)),
                                         key="sim_select")

    n_sim = st.slider("Number of similar tracks", 5, 20, 10)

    if st.button("Find Similar Gems"):
        with st.spinner("Calculating sonic similarity..."):
            similar_df = oracle.similar_to(selected_track_id_sim, n=n_sim)

            st.write("### Closest tracks from your library:")
            for _, r in similar_df.iterrows():
                shared = str(r.get('shared_tags', '') or '')
                tag_txt = f"  ·  _{shared}_" if shared else ""
                st.write(f"• **{r['name']}** — {r['artists']}  "
                         f"(similarity {r['similarity']:.3f}){tag_txt}")

with tab3:
    st.subheader("Discover New Music")
    st.markdown("Uses ReccoBeats to find tracks outside your library based on seeds.")

    seed_ids_input = st.text_input("Enter Spotify Track IDs (comma separated)",
                                   placeholder="e.g., 3n3Ppam7vgaVa1iaRUc9Lp, 7ouMYWpwJ422jRcDASZB7P")

    n_disco = st.slider("Number of tracks to discover", 5, 30, 15)

    if st.button("Discover New Sounds"):
        if not seed_ids_input:
            st.error("Please enter at least one Spotify Track ID.")
        else:
            with st.spinner("Scanning the musical multiverse..."):
                seeds = [s.strip() for s in seed_ids_input.split(',')]
                discovery = oracle.discover(seeds, n=n_disco)

                if discovery:
                    st.write("### The Oracle recommends:")
                    for obj in discovery:
                        title = obj.get('trackTitle') or obj.get('name')
                        artists = obj.get('artists')
                        if isinstance(artists, list):
                            artists = ", ".join(a.get('name', "") for a in artists)
                        st.write(f"• **{title}** — {artists}")
                else:
                    st.info("No discoveries found or ReccoBeats unavailable.")

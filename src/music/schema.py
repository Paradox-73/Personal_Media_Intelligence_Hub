from typing import List, Optional
from pydantic import BaseModel, Field, validator
import json

class MusicTrackSchema(BaseModel):
    track_id: str
    name: str
    artists: str
    artist_genres: List[str] = Field(default_factory=list)
    album: Optional[str] = None
    release_year: Optional[int] = None
    popularity: float = 0.0
    rating: float = 2.5
    
    @validator('artist_genres', pre=True)
    def parse_genres(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            # Handle JSON-encoded strings
            if v.startswith('[') and v.endswith(']'):
                try:
                    return json.loads(v)
                except:
                    pass
            # Handle comma-separated strings
            return [x.strip() for x in v.split(',')]
        return [str(v)]

def validate_track_data(data: dict) -> dict:
    """Validates and coerces track data using MusicTrackSchema."""
    return MusicTrackSchema(**data).dict()

"""daily_playlist with the network stubbed: Last.fm retry and the short-playlist guard."""
import importlib.util
from pathlib import Path

import pytest
import spotipy

_PATH = Path(__file__).resolve().parent.parent / "daily_playlist" / "daily_playlist.py"
_spec = importlib.util.spec_from_file_location("daily_playlist_mod", _PATH)
dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dp)


class FakeResponse:
    status_code = 200

    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


@pytest.fixture
def lastfm_replies(monkeypatch):
    """Queue of JSON bodies Last.fm returns, one per HTTP call; records each call."""
    calls = []

    def install(*bodies):
        queue = list(bodies)

        def fake_get(url, params=None, timeout=None):
            calls.append(params["method"])
            return FakeResponse(queue.pop(0))

        monkeypatch.setattr(dp.requests, "get", fake_get)
        return calls

    monkeypatch.setattr(dp.time, "sleep", lambda s: None)
    return install


SIMILAR = {"similartracks": {"track": [
    {"artist": {"name": "B"}, "name": "Song", "match": "0.5"}]}}


def test_transient_lastfm_errors_are_retried(lastfm_replies):
    calls = lastfm_replies({"error": 29, "message": "Rate limit"},
                           {"error": 11, "message": "Offline"},
                           {"error": 16, "message": "Unavailable"},
                           SIMILAR)
    assert dp.lastfm_similar("A", "Seed", 5) == [("B", "Song", 0.5)]
    assert len(calls) == 4


def test_permanent_lastfm_error_is_not_retried(lastfm_replies):
    calls = lastfm_replies({"error": 6, "message": "Track not found"})
    with pytest.raises(RuntimeError):
        dp.lastfm("track.getSimilar", artist="A", track="Seed")
    assert len(calls) == 1


def test_seed_that_keeps_failing_costs_only_that_seed(lastfm_replies):
    calls = lastfm_replies(*[{"error": 29, "message": "Rate limit"}] * 4)
    assert dp.lastfm_similar("A", "Seed", 5) == []
    assert len(calls) == 4


def test_short_playlist_never_replaces_the_old_one(monkeypatch):
    monkeypatch.setattr(dp, "LASTFM_API_KEY", "k")
    monkeypatch.setattr(dp, "LASTFM_USER", "u")
    monkeypatch.setattr(dp, "PLAYLIST_SIZE", 200)
    monkeypatch.setattr(dp, "get_spotify", lambda: object())
    monkeypatch.setattr(dp, "get_liked_tracks", lambda sp: [])
    monkeypatch.setattr(dp, "lastfm_recent_keys", lambda days, pages: set())
    monkeypatch.setattr(dp, "bucket_heavy_rotation",
                        lambda sp, n, ex: [f"spotify:track:{i}" for i in range(10)])
    monkeypatch.setattr(dp, "bucket_discovery", lambda sp, n, ex, keys: [])
    writes = []
    monkeypatch.setattr(dp, "find_or_create_playlist", lambda sp, name: writes.append("find"))
    monkeypatch.setattr(dp, "set_playlist_tracks", lambda *a: writes.append("set"))
    with pytest.raises(SystemExit):
        dp.main()
    assert writes == []


def test_failed_chunk_add_is_retried(monkeypatch):
    monkeypatch.setattr(dp.time, "sleep", lambda s: None)

    class FakeSpotify:
        def __init__(self):
            self.tracks, self.fail_once = [], True

        def playlist_replace_items(self, pid, uris):
            self.tracks = list(uris)

        def playlist_add_items(self, pid, uris):
            if self.fail_once:
                self.fail_once = False
                raise spotipy.SpotifyException(502, -1, "Bad gateway")
            self.tracks += uris

    sp = FakeSpotify()
    uris = [f"spotify:track:{i}" for i in range(250)]
    dp.set_playlist_tracks(sp, "pid", uris)
    assert sp.tracks == uris

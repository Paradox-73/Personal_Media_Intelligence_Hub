import pandas as pd

from src import config
from src.shows import ingestion


def _fake_fetch(name, year, **kw):
    return {'name': 'API Name', 'genres': ['X'], 'created_by': ['API Creator'],
            'overview': 'API overview', 'processing_status': 'success'}


def _setup(tmp_path, monkeypatch):
    raw = pd.DataFrame({
        'title': ['Show A', 'Show B', 'Show C'],
        'my_rating': [4.0, 3.5, 5.0],
        'premiered': ['2015-01-01', '2016-01-01', '2017-01-01'],
        'tvmaze_id': [1, 2, 3],
    })
    raw.to_csv(tmp_path / 'ratings.csv', index=False)
    enriched = pd.DataFrame({
        'tvmaze_id': [1, 2, 3],
        'name': ['Hand Name', 'Show B', 'Show C'],
        'year': [2015, 2016, 2017],
        'genres': ["['Drama']", "['Comedy']", "['Crime']"],
        'created_by': [None, None, None],  # empty -> every row needs a refetch
        'tvdb_score': [9.1, 8.2, 7.3],
    })
    enriched_path = tmp_path / 'enriched.csv'
    enriched.to_csv(enriched_path, index=False)
    monkeypatch.setattr(config, 'TV_SHOWS_RAW_DIR', tmp_path)
    monkeypatch.setattr(config, 'TV_SHOWS_ENRICHED_DATA_PATH', enriched_path)
    monkeypatch.setattr(ingestion, 'TMDB_KEY', 'test')
    calls = []
    monkeypatch.setattr(ingestion, 'get_tv_show_metadata_ultimate',
                        lambda *a, **kw: calls.append(a) or _fake_fetch(*a, **kw))
    return enriched_path, calls


def test_refetch_fills_gaps_only(tmp_path, monkeypatch):
    path, calls = _setup(tmp_path, monkeypatch)
    ingestion.run_ingestion()
    out = pd.read_csv(path)
    assert len(out) == 3
    assert len(calls) == 3
    assert out.loc[out.tvmaze_id == 1, 'name'].item() == 'Hand Name'
    assert out.loc[out.tvmaze_id == 1, 'genres'].item() == "['Drama']"
    assert out['tvdb_score'].tolist() == [9.1, 8.2, 7.3]
    assert (out['created_by'] == "['API Creator']").all()


def test_limit_caps_fetches_but_writes_every_row(tmp_path, monkeypatch):
    path, calls = _setup(tmp_path, monkeypatch)
    ingestion.run_ingestion(limit=1)
    out = pd.read_csv(path)
    assert len(calls) == 1
    assert len(out) == 3
    assert out['tvdb_score'].tolist() == [9.1, 8.2, 7.3]
    assert out.loc[out.tvmaze_id == 1, 'name'].item() == 'Hand Name'

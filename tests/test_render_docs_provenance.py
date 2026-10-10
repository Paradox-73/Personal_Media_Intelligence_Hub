"""Numbers and direction words in render_docs prose must come from the input data.

Guards the repeated failure where a marker block's caption kept old hand-typed numbers
(bias share "93–96%", TV "1.014 → 0.963") or a fixed "effect direction negative" next to
a table that said otherwise.
"""
import re

from src.reporting import render_docs as rd

NUM = re.compile(r"\d+\.\d+|\d+")


def _prose(md):
    """The caption lines (italic paragraphs), not the table rows."""
    return [ln for ln in md.splitlines() if ln.startswith("*") and not ln.startswith("**")]


def _bias_fixture():
    def row(n, bs, vs):
        return {"N": n, "n_features": 10, "bias2": 0.5, "variance": 0.05, "bias_share": bs,
                "var_share": vs, "test_mae": 0.6, "prescription": "BIAS-limited → x"}
    return {"models": {"movie": row(1000, 0.904, 0.096), "tv": row(150, 0.817, 0.183),
                       "game": row(60, 0.875, 0.125)}}


def test_bias_variance_prose_numbers_come_from_json(monkeypatch):
    monkeypatch.setattr(rd, "_load_json", lambda path: _bias_fixture())
    md = rd.format_bias_variance()
    caption = _prose(md)[-1]
    assert "**82–90%**" in caption and "10–18%" in caption
    assert set(NUM.findall(caption)) <= {"82", "90", "10", "18"}, caption
    assert "| 82% |" in md  # 0.817 rounds to 82, not truncates to 81
    assert "Every model is bias-dominated" in caption


def test_bias_variance_flags_variance_dominated(monkeypatch):
    fx = _bias_fixture()
    fx["models"]["game"].update(bias_share=0.4, var_share=0.6)
    monkeypatch.setattr(rd, "_load_json", lambda path: fx)
    caption = _prose(rd.format_bias_variance())[-1]
    assert "Every model is bias-dominated" not in caption
    assert "variance-dominated" in caption


def test_distillation_direction_and_verdict_follow_data():
    data = {"distillation": [
        {"Domain": "game", "MAE_NoPrior": 0.5, "Skill_NoPrior": 0.3, "MAE_WithPrior": 0.55,
         "Skill_WithPrior": 0.25, "p_value": 0.4, "Verdict": "DROP"},
        {"Domain": "book", "MAE_NoPrior": 0.3, "Skill_NoPrior": 0.5, "MAE_WithPrior": 0.25,
         "Skill_WithPrior": 0.6, "p_value": 0.01, "Verdict": "KEEP"},
    ]}
    caption = _prose(rd.format_distillation_table(data))[-1]
    assert "Game: the prior raised MAE 0.500 → 0.550, p = 0.40 (not significant)" in caption
    assert "Book: the prior lowered MAE 0.300 → 0.250, p = 0.01 (significant)" in caption
    assert "kept for Book" in caption and "dropped" not in caption
    assert set(NUM.findall(caption)) <= {"0.500", "0.550", "0.40", "0.300", "0.250", "0.01"}


def test_zero_shot_compares_against_deployed_model(monkeypatch):
    summary = {"verdict": "NULL", "zero_shot": {
        "tv": {"rho_full_model": 0.40, "best": {"source": "movie", "rho": 0.50,
                                                "pct_of_full_model": 1.25}},
        "game": {"rho_full_model": 0.30, "best": {"source": "movie", "rho": 0.60,
                                                  "pct_of_full_model": 2.0}}}}
    sanity = {"models": {"tv": {"spearman": 0.45}, "game": {"spearman": 0.70}}}
    files = {"reports/transfer_grid_summary.json": summary,
             "reports/baseline_sanity.json": sanity}
    monkeypatch.setattr(rd, "_load_json", lambda path: files.get(path))
    md = rd.format_transfer_verdict()
    assert "| tv | 0.400 | 0.450 | `movie` | +0.500 | **125%** |" in md
    assert "Own fully-trained" not in md and "the model trained on it" not in md
    assert "also wins for `movie->tv`" in md and "movie->game`." not in md.split("also wins for")[1]

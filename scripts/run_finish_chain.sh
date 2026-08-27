#!/usr/bin/env bash
# Re-run only the 4 downstream stages that failed (enrichment+feateng already OK).
# Fixes applied: PYTHONPATH (model_training's absolute import), UTF-8 (emoji prints),
# plus code fixes in profile_builder + cross_domain_transfer.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
PY=".venv/Scripts/python.exe"
R="reports"
ts() { date '+%H:%M:%S'; }
echo "FINISH_START $(ts)" > "$R/_finish.log"
run() {
  echo "[$(ts)] >>> $1" >> "$R/_finish.log"
  "$PY" "$2" > "$R/_$1.log" 2>&1
  echo "[$(ts)] <<< $1 exit=$?" >> "$R/_finish.log"
}
run feateng  src/music/feature_engineering.py
run train    src/music/model_training.py
run profile  src/music/profile_builder.py
run unified  src/unified_model/unified_feature_engineering.py
run transfer src/unified_model/cross_domain_transfer.py
echo "FINISH_DONE $(ts)" >> "$R/_finish.log"

#!/usr/bin/env bash
# One-shot music enrichment -> retrain -> cross-domain transfer chain.
# Continue-on-error (;) so every stage's log is produced even if one hiccups.
cd "E:/Personal_Media_Intelligence_Hub" || exit 1
PY="content_rec/Scripts/python.exe"
R="reports"
ts() { date '+%H:%M:%S'; }

echo "CHAIN_START $(ts)" > "$R/_chain.log"

run() { # run <label> <script>
  echo "[$(ts)] >>> $1" >> "$R/_chain.log"
  "$PY" "$2" > "$R/_$1.log" 2>&1
  echo "[$(ts)] <<< $1 exit=$?" >> "$R/_chain.log"
}

run mb        src/music/musicbrainz_enrichment.py
run extra     src/music/extra_enrichment.py
run genius    src/music/genius_lyrics.py
run feateng   src/music/feature_engineering.py
run train     src/music/model_training.py
run profile   src/music/profile_builder.py
run unified   src/unified_model/unified_feature_engineering.py
run transfer  src/unified_model/cross_domain_transfer.py

echo "CHAIN_DONE $(ts)" >> "$R/_chain.log"

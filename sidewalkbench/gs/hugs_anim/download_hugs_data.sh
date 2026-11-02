#!/bin/bash
# Download Apple HUGS assets (NeuMan data / pretrained avatars) into the current directory:
#   download_hugs_data.sh neuman_data.zip hugs_pretrained_models.zip
# Apple's CDN resets long single connections, so each file is fetched as N parallel byte ranges
# that resume after a reset, then concatenated.
set -e
N=8
for f in "$@"; do
  u=https://docs-assets.developer.apple.com/ml-research/models/hugs/$f
  want=$(curl -sIL "$u" | grep -i '^content-length' | tail -1 | tr -dc 0-9)
  [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" = "$want" ] && { echo "have $f"; continue; }
  rm -f "$f"; chunk=$(( (want + N - 1) / N ))
  for k in $(seq 0 $((N - 1))); do
    ( a=$((k * chunk)); b=$((a + chunk - 1)); [ $b -ge $want ] && b=$((want - 1)); len=$((b - a + 1))
      until [ "$(stat -c %s "$f.part$k" 2>/dev/null || echo 0)" -ge $len ]; do
        got=$(stat -c %s "$f.part$k" 2>/dev/null || echo 0)
        curl -sS -L --max-time 600 -r $((a + got))-$b "$u" >> "$f.part$k" || sleep 3
      done ) &
  done
  wait
  cat $(for k in $(seq 0 $((N - 1))); do echo "$f.part$k"; done) > "$f" && rm "$f".part*
  [ "$(stat -c %s "$f")" = "$want" ] && echo "done $f ($want bytes)" || { echo "size mismatch for $f" >&2; exit 1; }
done

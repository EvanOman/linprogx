#!/usr/bin/env bash
# Download the LPnetlib suite instances used by experiments/suite_bench.py.
set -euo pipefail
dir="${1:-/tmp/lpsuite}"
mkdir -p "$dir"
cd "$dir"
for p in lp_ken_18 lp_pds_20 lp_ken_13 lp_stocfor3 lp_pds_10 lp_ken_11 lp_osa_60 \
         lp_cre_b lp_cre_d lp_qap15 lp_osa_30 lp_cre_a lp_qap12 lp_maros_r7 \
         lp_fit2p lp_ken_07 lp_greenbea lp_osa_14 lp_80bau3b lp_d2q06c \
         lp_pilot87 lp_degen3 lp_truss lp_woodw; do
  [ -f "$p.mat" ] || curl -sL -o "$p.mat" "https://sparse.tamu.edu/mat/LPnetlib/$p.mat" &
done
wait
ls -la "$dir"

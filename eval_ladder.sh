#!/bin/bash
# every trained model on the same unseen scenes: 384 standard + 192 hard; safety rule on for all, raw (no rule) for the seed-0 / main models
cd "$(dirname "$0")"
out=results_ladder.txt; : > $out
run() { echo "== $*" >> $out; python3 eval_policy.py "$@" --workers 22 2>&1 | tail -n 1 >> $out; }
MAIN="diffusion:d16k_s0 diffusion:d8k_s0 diffusion:d4k_s0 diffusion:d2k_s0 diffusion:d1k_s0 diffusion:d32k diffusion:d100k diffusion:d340k diffusion:d1m diffusion:d5m8_s0 diffusion:diff_r3 diffusion:diff_big_r3 \
 bc:bc16k_s0 bc:bc8k_s0 bc:bc4k_s0 bc:bc2k_s0 bc:bc1k_s0 bc:bc32k bc:bc100k bc:bc5m8_s0 bc:bc_r3"
EXTRA="diffusion:d32k_s1 diffusion:d32k_s2 diffusion:d2k_s1 diffusion:d2k_s2 diffusion:d1k_s1 diffusion:d1k_s2 bc:bc32k_s1 bc:bc32k_s2 bc:bc2k_s1 bc:bc2k_s2 bc:bc1k_s1 bc:bc1k_s2"
for set in "--n 384" "--n 192 --hard"; do
  echo "##### $set" >> $out
  for m in $MAIN $EXTRA; do kind=${m%%:*}; k=$([ $kind = diffusion ] && echo diff || echo bc); c=ckpt/${m#*:}.pt; run $k --ckpt $c $set --safety2; done
  for m in $MAIN; do kind=${m%%:*}; k=$([ $kind = diffusion ] && echo diff || echo bc); c=ckpt/${m#*:}.pt; run $k --ckpt $c $set; done
done
echo FINISHED >> $out

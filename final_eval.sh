#!/bin/bash
# final comparison: same unseen scenes for every method
cd "$(dirname "$0")"
out=results.txt; : > $out
run() { echo "== $*" >> $out; python3 eval_policy.py "$@" --workers 22 2>&1 | tail -n 1 >> $out; }
for set in "--n 192" "--n 96 --hard"; do
  echo "##### test set: $set" >> $out
  run rule $set
  run rule $set --safety2
  run teacher $set
  run bc --ckpt ckpt/bc_r3.pt $set
  run bc --ckpt ckpt/bc_r3.pt $set --safety2
  run diff --ckpt ckpt/diff_r3.pt $set
  run diff --ckpt ckpt/diff_r3.pt $set --safety2
  run diff --ckpt ckpt/diff_big_r3.pt $set
  run diff --ckpt ckpt/diff_big_r3.pt $set --safety2
done
echo FINISHED >> $out

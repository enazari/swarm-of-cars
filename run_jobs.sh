#!/bin/bash
# run every line of jobs.txt (name|gpu|train args), 8 at a time
cd "$(dirname "$0")"
run(){ IFS='|' read -r name gpu args <<< "$1"; python3 train.py --gpu $gpu --data data/r0 data/r1 data/r2 data/r3 --steps 150000 --bs 512 --wsteps 4,2 --out ckpt/$name $args > logs/$name.log 2>&1; }
export -f run
tr '\n' '\0' < jobs.txt | xargs -0 -P 8 -I{} bash -c 'run "$1"' _ {}
echo ALLDONE > logs/ALLDONE

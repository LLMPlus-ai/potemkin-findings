#!/bin/zsh
# Remaining world simulations (resumable).
cd /Volumes/Crucial3/12_si/10_os_1
source scripts/env.sh
for m in pyramid_z remote_z firm_age_z; do
  python3 -u scripts/11_null_worlds.py typeB $m 0 500 6 > logs/11_typeB_$m.out 2>&1
done
python3 -u scripts/11_null_worlds.py guardA 0 500 6 > logs/11_guardA.out 2>&1
for b in 0.02 0.05 0.10; do
  python3 -u scripts/11_null_worlds.py planted $b 0.05 0 100 6 > logs/11_planted_$b.out 2>&1
done
echo ALL_WORLDS_DONE

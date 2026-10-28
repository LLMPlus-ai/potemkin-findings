#!/bin/zsh
# Moderator Monte Carlo (scripts/24) in reverse scenario order, run alongside run_mc_validity.sh (resumable).
cd /Volumes/Crucial3/12_si/10_os_1
source scripts/env.sh
for s in all nonlinear proxy trend clean; do
  python3 -u scripts/24_mc_moderators.py $s 200 59 ${1:-4} > logs/24_mc_${s}_b.out 2>&1
done
echo MC_VALIDITY2_DONE

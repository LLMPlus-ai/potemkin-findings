#!/bin/zsh
# Conditional-permutation moderator worlds (scripts/22 residB; resumable).
cd /Volumes/Crucial3/12_si/10_os_1
source scripts/env.sh
for m in pyramid_z remote_z firm_age_z; do
  python3 -u scripts/22_validity_worlds.py residB $m 0 500 ${1:-6} > logs/22_residB_$m.out 2>&1
done
echo VALIDITY2_DONE

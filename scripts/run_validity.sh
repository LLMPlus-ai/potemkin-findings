#!/bin/zsh
# Driver for the validity worlds of scripts/22 (resumable; finished worlds are skipped).
cd /Volumes/Crucial3/12_si/10_os_1
source scripts/env.sh
python3 -u scripts/22_validity_worlds.py real > logs/22_real.out 2>&1
for m in pyramid_z remote_z firm_age_z; do
  python3 -u scripts/22_validity_worlds.py permB $m 0 500 8 > logs/22_permB_$m.out 2>&1
done
python3 -u scripts/22_validity_worlds.py joint 0 1000 8 > logs/22_joint.out 2>&1
python3 -u scripts/22_validity_worlds.py covA 0 1000 8 > logs/22_covA.out 2>&1
for m in pyramid_z remote_z firm_age_z; do
  python3 -u scripts/22_validity_worlds.py fl $m 0 500 8 > logs/22_fl_$m.out 2>&1
done
echo VALIDITY_DONE

#!/bin/zsh
cd /Volumes/Crucial3/12_si/10_os_1
source scripts/env.sh
python3 -u scripts/13_monte_carlo.py null 100 39 4 > logs/13_mc_null.out 2>&1
python3 -u scripts/13_monte_carlo.py leakage 100 39 4 > logs/13_mc_leakage.out 2>&1
python3 -u scripts/13_monte_carlo.py power 50 39 4 > logs/13_mc_power.out 2>&1
python3 -u scripts/13_monte_carlo.py confounded 50 39 4 > logs/13_mc_confounded.out 2>&1
python3 -u scripts/13_monte_carlo.py leakage_replayed 50 39 4 > logs/13_mc_leakage_replayed.out 2>&1
echo MC_DONE

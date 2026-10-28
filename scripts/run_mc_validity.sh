#!/bin/zsh
# Driver for the validity Monte Carlo studies of scripts/23 and scripts/24 (resumable).
cd /Volumes/Crucial3/12_si/10_os_1
source scripts/env.sh
NW=${1:-6}
for c in null level post_r0.2_d0.025 post_r0.2_d0.05 post_r0.2_d0.1 post_r0.2_d0.2 \
         post_r0.4_d0.025 post_r0.4_d0.05 post_r0.4_d0.1 post_r0.4_d0.2 \
         post_r0.6_d0.025 post_r0.6_d0.05 post_r0.6_d0.1 post_r0.6_d0.2; do
  python3 -u scripts/23_mc_exchangeability.py $c 200 99 $NW > logs/23_mc_$c.out 2>&1
done
for s in clean trend proxy nonlinear all; do
  python3 -u scripts/24_mc_moderators.py $s 200 59 $NW > logs/24_mc_$s.out 2>&1
done
echo MC_VALIDITY_DONE

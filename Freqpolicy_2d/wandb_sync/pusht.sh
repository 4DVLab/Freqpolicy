#!/bin/bash 
while true; do
  wandb sync /inspurfs/group/mayuexin/zym/nips_dp_2025/diffusion_policy/data/outputs/2025.05.08/23.01.52_train_diffusion_far_lowdim_pusht_lowdim/wandb/offline-run-20250508_230209-mapklz6j
  sleep 60  # 每隔 60 秒同步一次 
done
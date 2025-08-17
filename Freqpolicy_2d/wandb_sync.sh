#!/bin/bash 
while true; do
  wandb sync /inspurfs/group/mayuexin/zym/nips_dp_2025/diffusion_policy/data/outputs/2025.05.06/13.02.43_train_diffusion_far_lowdim_square_lowdim/wandb/offline-run-20250506_130301-n61v3rse
  sleep 60  # 每隔 60 秒同步一次 
done
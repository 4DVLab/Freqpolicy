#!/bin/bash 
while true; do
  wandb sync /inspurfs/group/mayuexin/zym/nips_dp_2025/diffusion_policy/data/outputs/2025.05.08/23.24.55_train_diffusion_far_lowdim_lift_lowdim/wandb/offline-run-20250508_232510-50pcg4vs
  sleep 60  # 每隔 60 秒同步一次 
done
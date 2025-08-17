#!/bin/bash 
while true; do
  wandb sync /inspurfs/group/mayuexin/zym/nips_dp_2025/diffusion_policy/data/outputs/2025.05.08/23.02.31_train_diffusion_far_lowdim_can_lowdim/wandb/offline-run-20250508_230251-cc449myn
  sleep 60  # 每隔 60 秒同步一次 
done
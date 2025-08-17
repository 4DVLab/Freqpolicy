#!/bin/bash 
while true; do
  wandb sync /inspurfs/group/mayuexin/zym/nips_dp_2025/diffusion_policy/data/outputs/2025.05.08/23.16.28_train_diffusion_far_lowdim_square_lowdim/wandb/offline-run-20250508_231648-bx10tvrh
  sleep 60  # 每隔 60 秒同步一次 
done
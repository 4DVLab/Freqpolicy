#!/bin/bash 
while true; do
  wandb sync /inspurfs/group/mayuexin/zym/nips_dp_2025/diffusion_policy/data/outputs/2025.05.08/23.15.20_train_diffusion_far_hybrid_can_image/wandb/offline-run-20250508_231602-mbxejv7y
  sleep 60  # 每隔 60 秒同步一次 
done
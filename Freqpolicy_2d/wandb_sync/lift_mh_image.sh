#!/bin/bash 
while true; do
  wandb sync /inspurfs/group/mayuexin/zym/nips_dp_2025/diffusion_policy/data/outputs/2025.05.09/11.03.57_train_diffusion_far_hybrid_lift_image/wandb/offline-run-20250509_110440-luq5l5pd
  sleep 60  # 每隔 60 秒同步一次 
done
#!/bin/bash 
while true; do
  wandb sync /inspurfs/group/mayuexin/zym/nips_dp_2025/diffusion_policy/data/outputs/2025.05.09/00.19.04_train_diffusion_far_hybrid_square_image/wandb/offline-run-20250509_001939-8fqqc0xs
  sleep 60  # 每隔 60 秒同步一次 
done
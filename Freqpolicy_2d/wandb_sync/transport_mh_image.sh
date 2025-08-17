#!/bin/bash 
while true; do
  wandb sync /inspurfs/group/mayuexin/zym/nips_dp_2025/diffusion_policy/data/outputs/2025.05.08/23.03.00_train_diffusion_far_hybrid_transport_image/wandb/offline-run-20250508_230438-jgy8j4bp
  sleep 60  # 每隔 60 秒同步一次 
done
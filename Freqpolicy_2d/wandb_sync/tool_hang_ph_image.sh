#!/bin/bash 
while true; do
  wandb sync /inspurfs/group/mayuexin/zym/nips_dp_2025/diffusion_policy/data/outputs/2025.05.08/23.11.11_train_diffusion_far_hybrid_tool_hang_image_abs/wandb/offline-run-20250508_231158-13420v84
  sleep 60  # 每隔 60 秒同步一次 
done
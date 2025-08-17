#!/bin/bash 
while true; do
  wandb sync /root/xiaochy/nips_2025/diffusion_policy/data/outputs/2025.05.08/07.26.46_train_diffusion_far_hybrid_transport_image/wandb/offline-run-20250508_072745-2u8y2apn
  sleep 60  # 每隔 60 秒同步一次 
done
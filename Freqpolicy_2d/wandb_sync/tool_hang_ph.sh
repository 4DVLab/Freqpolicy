while true; do
  wandb sync /inspurfs/group/mayuexin/zym/nips_dp_2025/diffusion_policy/data/outputs/2025.05.09/00.21.01_train_diffusion_far_lowdim_tool_hang_lowdim/wandb/offline-run-20250509_002133-lfor0ger
  sleep 60  # 每隔 60 秒同步一次 
done
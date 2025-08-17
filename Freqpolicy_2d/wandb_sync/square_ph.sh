while true; do
  wandb sync /inspurfs/group/mayuexin/zym/nips_dp_2025/diffusion_policy/data/outputs/2025.05.08/23.07.14_train_diffusion_far_lowdim_square_lowdimbs/wandb/offline-run-20250508_230733-t6x25p4g
  sleep 60  # 每隔 60 秒同步一次 
done
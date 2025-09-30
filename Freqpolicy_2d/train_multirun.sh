#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2
ray start --head --num-gpus=3
################################################################################
#                           Training Script Collection                         #
################################################################################

# ==============================================================================
# Ray Multi-Run - Freqpolicy Low Dimension Task
# ==============================================================================
python ray_train_multirun.py \
    --config-dir=config_task/low_dim \
    --config-name=pusht.yaml \
    --seeds=42,43,44 \
    --monitor_key=test/mean_score \
    -- multi_run.run_dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
    multi_run.wandb_name_base='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}'

# ==============================================================================
# Ray Multi-Run - Freqpolicy Image Task
# ==============================================================================
# python ray_train_multirun.py \
#     --config-dir=config_task/image \
#     --config-name=pusht.yaml \
#     --seeds=42,43,44 \
#     --monitor_key=test/mean_score \
#     -- multi_run.run_dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
#     multi_run.wandb_name_base='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}'

# ==============================================================================
# Ray Multi-Run - Freqpolicy Resume Training
# ==============================================================================
# export CUDA_VISIBLE_DEVICES=0,1,2
# ray start --head --num-gpus=3
# python ray_train_multirun.py \
#     --config-dir=config_task/low_dim \
#     --config-name=pusht.yaml \
#     --seeds=42,43,44 \
#     --monitor_key=test/mean_score \
#     -- multi_run.run_dir='data/outputs/2025.09.29/16.15.29_train_freqpolicy_lowdim_pusht_lowdim' \
#     multi_run.wandb_name_base='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}'

# ==============================================================================
# Ray Multi-Run - Diffusion Policy Image Task
# ==============================================================================
# python ray_train_multirun.py \
#     --config-dir=. \
#     --config-name=image_pusht_diffusion_policy_cnn.yaml \
#     --seeds=42,43,44 \
#     --monitor_key=test/mean_score \
#     -- multi_run.run_dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
#     multi_run.wandb_name_base='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}'
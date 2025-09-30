#!/bin/bash

################################################################################
#                           Training Script Collection                         #
################################################################################

# ==============================================================================
# Freqpolicy - Low Dimension Task
# ==============================================================================
python train.py \
    --config-dir=config_task/low_dim \
    --config-name=pusht.yaml \
    training.seed=42 \
    training.device=cuda:0 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# ==============================================================================
# Freqpolicy - Image Task
# ==============================================================================
# python train.py \
#     --config-dir=config_task/image \
#     --config-name=pusht.yaml \
#     training.seed=42 \
#     training.device=cuda:0 \
#     hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# ==============================================================================
# Freqpolicy - Resume Training
# ==============================================================================
# python train.py \
#     --config-dir=config_task/low_dim \
#     --config-name=pusht.yaml \
#     training.seed=42 \
#     training.device=cuda:0 \
#     hydra.run.dir='data/outputs/2025.09.29/16.15.29_train_freqpolicy_lowdim_pusht_lowdim'

# ==============================================================================
# Diffusion Policy - Task
# ==============================================================================
# python train.py \
#     --config-dir=. \
#     --config-name=image_pusht_diffusion_policy_cnn.yaml \
#     training.seed=42 \
#     training.device=cuda:0 \
#     hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

################################################################################
import argparse
import datetime
import numpy as np
import os
import time
from einops import rearrange, reduce
import math
import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from typing import Dict
from diffusion_policy_3d.common.model_util import print_params
import diffusion_policy_3d.util.misc as misc
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.util.misc import NativeScalerWithGradNormCount as NativeScaler
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.models.vae import AutoencoderKL
from diffusion_policy_3d.models import far_ours
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder
from termcolor import cprint
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy_3d.models.far_ours import FAR
from functools import partial

class Freqpolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            mask=True,
            mask_ratio_min=0.7,
            diffloss_d=3,
            diffloss_w=1024,
            num_sampling_steps='ddim10',
            diffusion_batch_mul=4,
            num_iter=4,
            temperature=1.0,
            obs_as_global_cond=True,
            condition_type="film",
            point_feature_dim=128,
            state_mlp_size=128,
            encoder_embed_dim=256,
            decoder_embed_dim=256,
            encoder_depth=4,
            decoder_depth=4,
            encoder_num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=4,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            loss_weight=1,
            cfg=1.0,
            **kwargs):
        super().__init__()

        self.condition_type = condition_type
        self.point_feature_dim = point_feature_dim
        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                                img_crop_shape=crop_shape,
                                                out_channel=point_feature_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                state_mlp_size=(state_mlp_size, state_mlp_size),
                                                )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")
        self.cfg = cfg
        self.obs_encoder = obs_encoder

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.state_mlp_size = state_mlp_size
        self.kwargs = kwargs
        self.mask_ratio_min = mask_ratio_min
        self.loss_weight = loss_weight
        self.mask = mask # 是否使用mask
        self.diffloss_d = diffloss_d
        self.diffloss_w = diffloss_w
        self.num_sampling_steps = num_sampling_steps
        self.diffusion_batch_mul = diffusion_batch_mul
        self.temperature = temperature  # 采样温度
        self.num_iter = num_iter  # 采样步数
        ####################use VAE TODO####################
        # self.vae = AutoencoderKL()
        # for param in self.vae.parameters():
        #     param.requires_grad = True
        ####################################################
        print(f"创建模型，参数：action_dim={action_dim}, obs_feature_dim={obs_feature_dim if self.obs_as_global_cond else None}")
        # 直接实例化FAR类，而不是通过Freqpolicy_base函数
        
        self.model = FAR(
            trajectory_dim=self.action_dim,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            mask=self.mask,
            mask_ratio_min=self.mask_ratio_min,
            diffloss_d=self.diffloss_d,
            diffloss_w=self.diffloss_w,
            num_iter=self.num_iter,
            condition_dim=self.point_feature_dim + self.state_mlp_size, #
            num_sampling_steps=self.num_sampling_steps,
            diffusion_batch_mul=self.diffusion_batch_mul,
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            encoder_num_heads=encoder_num_heads,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        # print("Model = %s" % str(self.model))
        # following timm: set wd as 0 for bias and norm layers
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of trainable parameters: {}M".format(n_params / 1e6))
        print_params(self)
        print('self.num_iter', self.num_iter)
        print('self.cfg', self.cfg)

    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
            ):
        B = condition_data.shape[0]
        model = self.model
        with torch.no_grad():
            sampled_trajectory = model.sample_tokens_mask(
                bsz=B,
                num_iter=self.num_iter,
                conditions=global_cond,
                temperature=self.temperature,
                cfg=self.cfg
            )
        return sampled_trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        import time
        start_time = time.time()
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        end_time = time.time()
        # print(f"采样时间: {end_time - start_time} 秒")
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result
    
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch): 
        # self.vae.train(True)
        self.model.train(True)
        # normalize input

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
       
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        conditions = global_cond     
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.model(trajectory, conditions, loss_weight=self.loss_weight)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss_dict = {
                'bc_loss': loss.item(),
            }
        
        return loss, loss_dict



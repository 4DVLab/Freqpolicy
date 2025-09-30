
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.Freqpolicy.Freqpolicy import Freqpolicy
from functools import partial
from typing import Optional, Dict, Tuple, Union, List, Type
import math
import sys
# from diffusion_policy.common.model_util import print_params

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules



class FreqpolicyLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: Freqpolicy, # new
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            mask=True, # need to add to dp
            mask_ratio_min=0.7, # need to add to dp
            diffloss_d=3, # need to add to dp
            diffloss_w=1024, # need to add to dp
            num_sampling_steps='ddim10', # need to add to dp
            diffusion_batch_mul=1, # need to add to dp
            num_iter=4, # need to add to dp
            temperature=1.0, # need to add to dp
            point_feature_dim=64, # need to add to dp
            state_mlp_size=64, # need to add to dp
            encoder_embed_dim=512, # need to add to dp
            decoder_embed_dim=512, # need to add to dp
            encoder_depth=4, # need to add to dp
            decoder_depth=4, # need to add to dp
            encoder_num_heads=8, # need to add to dp
            decoder_num_heads=8, # need to add to dp
            mlp_ratio=4, # need to add to dp
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            loss_weight=1, # need to add to dp
            cfg=1.0, # need to add to dp
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs
        # new
        self.cfg = cfg
        self.state_mlp_size = state_mlp_size
        self.mask_ratio_min = mask_ratio_min
        self.loss_weight = loss_weight
        self.mask = mask # whether to use mask
        self.diffloss_d = diffloss_d
        self.diffloss_w = diffloss_w
        self.num_sampling_steps = num_sampling_steps
        self.diffusion_batch_mul = diffusion_batch_mul
        self.temperature = temperature  # sampling temperature
        self.num_iter = num_iter  # number of sampling steps

        # if num_inference_steps is None:
        #     num_inference_steps = noise_scheduler.config.num_train_timesteps
        # self.num_inference_steps = num_inference_steps
        
        # # set new obs_encoder, only need to map state here, so it's a simple state_mlp
        # self.state_mlp_activation_fn = nn.ReLU
        # self.input_dim = self.obs_dim
        # self.output_dim = self.state_mlp_size
        # self.net_arch = [64, 64]
        # self.obs_encoder = nn.Sequential(*create_mlp(self.input_dim, self.output_dim, self.net_arch, self.state_mlp_activation_fn))
        
        # import pdb; pdb.set_trace()
        
        self.model = Freqpolicy(
            trajectory_dim=self.action_dim, # 10
            horizon=self.horizon,  # 16
            n_obs_steps=self.n_obs_steps, # 2
            mask=self.mask, # True
            mask_ratio_min=self.mask_ratio_min, # 0.7
            diffloss_d=self.diffloss_d, # 3
            diffloss_w=self.diffloss_w, # 1024
            num_iter=self.num_iter, # 4
            condition_dim= self.state_mlp_size, 
            num_sampling_steps=self.num_sampling_steps, # '100'
            diffusion_batch_mul=self.diffusion_batch_mul, # 4
            encoder_embed_dim=encoder_embed_dim, # 256
            decoder_embed_dim=decoder_embed_dim, # 256
            encoder_depth=encoder_depth, # 4
            decoder_depth=decoder_depth, # 4
            encoder_num_heads=encoder_num_heads, # 8
            decoder_num_heads=decoder_num_heads, # 8
            mlp_ratio=mlp_ratio, # 4
            norm_layer=partial(nn.LayerNorm, eps=1e-6) 
        )
        # following timm: set wd as 0 for bias and norm layers
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of trainable parameters: {}M".format(n_params / 1e6))
        # print_params(self)
        print('self.num_iter', self.num_iter)
        print('self.cfg', self.cfg)
        
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
            ):
        # import pdb; pdb.set_trace()
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

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        # import pdb; pdb.set_trace()
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        # import pdb; pdb.set_trace()
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        # the following obs and action are all normalized
        obs = nbatch['obs']
        action = nbatch['action']
        batch_size = action.shape[0]
        horizon = action.shape[1]

        # handle different ways of passing observation
        global_cond = None
        trajectory = action # using normalized action
        local_cond = None 
        
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)
        
        conditions = global_cond     
        # with torch.cuda.amp.autocast():
        loss = self.model(trajectory, conditions, loss_weight=self.loss_weight)
        loss_value = loss.item()
        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)
        return loss

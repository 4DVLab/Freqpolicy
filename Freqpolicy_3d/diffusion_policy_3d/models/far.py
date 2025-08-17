from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from timm.models.vision_transformer import Block

from diffusion_policy_3d.models.diffloss import DiffLoss

import torch.nn.functional as F
import torchvision.transforms as T
import random


from torchvision.utils import make_grid
from typing import Optional
from PIL import Image


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


class FAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask=True,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(1000, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        self.loss_weight = [1 + np.sin(math.pi / 2. * (bands + 1) / self.seq_h) for bands in range(self.seq_h)]

        # --------------------------------------------------------------------------
        self.mask = mask
        # FAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # --------------------------------------------------------------------------
        # FAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # FAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        if self.mask:
            torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        """将输入动作张量划分为patch序列
        Args:
            x: [bsz, c, h, w] 例如 [B, 16, 256, 256] 输入动作
        Returns:
            [bsz, h_*w_, c*p^2] 例如 [B, 256, 16] patch序列
        """
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        """将patch序列重构为动作张量
        Args:
            x: [bsz, h_*w_, c*p^2] 例如 [B, 256, 16] patch序列
        Returns:
            [bsz, c, h, w] 例如 [B, 16, 256, 256] 重构的动作
        """
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders, x_index):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        stage = x_index[0]
        mask_ratio_min = 0.7 * stage / 16
        mask_rate = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25).rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask


    def processingpregt_latent(self, imgs):
        """处理输入动作的渐进式降采样和上采样
        Args:
            imgs: [B, C, H, W] 例如 [B, 16, 256, 256] 输入动作
        Returns:
            out: [B, C, H, W] 处理后的动作
            core_index: [B] 降采样尺寸的索引
        """
        B, C, H, W = imgs.shape
        out = torch.zeros_like(imgs)
        latent_core = list(range(H))
        core_index = []

        random_number = torch.randint(0, len(latent_core), (1,))
        for i in range(B):
            chosen_core = latent_core[random_number]
            core_index.append(chosen_core)
            if random_number == 0:
                out[i] = torch.zeros(C, H, W).cuda()    # torch.Size([256, 256, 16])
            else:
                imgs_resize = F.interpolate(imgs[i].unsqueeze(0), size=(chosen_core, chosen_core), mode='area')
                out[i] = F.interpolate(imgs_resize, size=(H, W), mode='bicubic').squeeze(0)
        core_index = torch.tensor(core_index).to(out.device).half()
        return out, core_index

    
    def forward_mae_encoder(self, x, class_embedding, mask=None):
        """MAE编码器前向传播
        Args:
            x: [bsz, seq_len, token_embed_dim] 例如 [B, 256, 16] patch序列
            class_embedding: [bsz, encoder_embed_dim] 例如 [B, 1024] 类别嵌入
            mask: [bsz, seq_len] 掩码矩阵
        Returns:
            [bsz, 非掩码tokens数, encoder_embed_dim] 编码器输出
        流程:
            1. 线性投影: [B, 256, 16] -> [B, 256, 1024]
            2. 添加buffer: [B, 256, 1024] -> [B, 320, 1024]
            3. 应用掩码: [B, 320, 1024] -> [B, 非掩码tokens数, 1024]
        """
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        
        if self.mask:
            mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:  
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # dropping
        if self.mask:
            x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
        
        # apply Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask=None):
        """MAE解码器前向传播
        Args:
            x: [bsz, 非掩码tokens数, encoder_embed_dim] 编码器输出
            mask: [bsz, seq_len] 掩码矩阵
        Returns:
            [bsz, seq_len, decoder_embed_dim] 解码器输出
        流程:
            1. 投影到解码空间: [B, 非掩码tokens数, 1024]
            2. 恢复掩码位置: [B, 320, 1024]
            3. 移除buffer: [B, 256, 1024]
        """
        x = self.decoder_embed(x)
        if self.mask:
            mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)
            # pad mask tokens
            mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
            x_after_pad = mask_tokens.clone()
            x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            x = x_after_pad + self.decoder_pos_embed_learned
        else:
            x = x + self.decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask, index, loss_weight=False):
        """计算损失函数
        Args:
            z: [bsz, seq_len, decoder_embed_dim] 例如 [B, 256, 1024] 解码器输出
            target: [bsz, seq_len, token_embed_dim] 例如 [B, 256, 16] 目标动作
            mask: [bsz, seq_len] 掩码矩阵
            index: [bsz] 降采样尺寸索引
        流程:
            1. 重塑维度: [B*256, 16/1024] (target/z)
            2. 重复diffusion_batch_mul次
            3. 计算损失
        """
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        index = index.unsqueeze(1).unsqueeze(-1).repeat(1, seq_len, 1).reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        
        if loss_weight:
            loss_weight = loss_weight.unsqueeze(1).repeat(1, seq_len).reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, index=index, loss_weight=loss_weight)
        return loss



    def forward(self, imgs, labels, loss_weight=False):
        class_embedding = self.class_emb(labels)
        
        process_imgs, x_index = self.processingpregt_latent(imgs)
        if loss_weight:
            loss_weight = self.loss_weight

        x = self.patchify(process_imgs)         # x.shape: torch.Size([B, 256, 16]))
        gt_latents = self.patchify(imgs)
        
        mask = None
        if self.mask:
            orders = self.sample_orders(bsz=x.size(0))
            mask = self.random_masking(x, orders, x_index)

        x = self.forward_mae_encoder(x, class_embedding, mask)
        z = self.forward_mae_decoder(x, mask)
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask, index=x_index, loss_weight=loss_weight)

        return loss




    def sample_tokens_mask(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):
        latent_core = [0,12,12,4,5,6,7,8,9,10,11]

        num_iter = len(latent_core)
        mask = torch.ones(bsz, self.seq_len).cuda()      
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        for step in list(range(num_iter)):
            cur_tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
            if labels is not None:
                class_embedding = self.class_emb(labels)      
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
    
            tokens = torch.cat([tokens, tokens], dim=0)
            class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
            mask = torch.cat([mask, mask], dim=0)

            x = self.forward_mae_encoder(tokens, class_embedding, mask)
            z = self.forward_mae_decoder(x, mask)    # torch.Size([512, 256, 768])
            B, L, C = z.shape
            z = z.reshape(B * L, -1)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()     # 251, 236, 212, 181, 142, 97, 49, 0 
            cfg_iter = 1 + (cfg - 1) * step / num_iter
            temperature_iter = 0.85 + (temperature - 0.85) * step / num_iter

            index = torch.tensor([latent_core[step]]).unsqueeze(1).unsqueeze(-1).repeat(B, L, 1).reshape(B * L, -1).to(z.device).half()
            z = self.diffloss.sample(z, temperature_iter, cfg_iter, index)     # torch.Size([512, 16])
            z, _ = z.chunk(2, dim=0)  # Remove null class samples.  torch.Size([256, 16])
            
            if step < num_iter-1:
                z = z.reshape(bsz, L, -1).transpose_(1, 2).reshape(bsz, -1, 16, 16)
                
                if step > 0:
                    imgs_resize = F.interpolate(z, size=(latent_core[step+1], latent_core[step+1]), mode='area')
                    z = F.interpolate(imgs_resize, size=(16, 16), mode='bicubic')
                z = z.reshape(bsz, -1, L).transpose_(1, 2).reshape(bsz*L, -1)
                
            
            sampled_token = z.reshape(bsz, L, -1)
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            mask_to_pred = torch.logical_not(mask_next)
            mask = mask_next
            sampled_token_latent = sampled_token[mask_to_pred.nonzero(as_tuple=True)]
            
            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()


        tokens = tokens.transpose_(1, 2).reshape(bsz, -1, 16, 16)
        return tokens
    

    def sample_tokens_nomask(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", 
                           labels=None, temperature=1.0, progress=False):
        """无掩码的渐进式生成过程
        Args:
            bsz: batch size
            num_iter: 迭代次数
            cfg: 条件引导系数
        流程:
            1. 初始化: [B, 256, 16] 全零tensor
            2. 每次迭代:
               - 编码器输入: [B*2, 256, 16]
               - 编码器输出: [B*2, 256, 1024]
               - 解码器输出: [B*2, 256, 1024]
               - 扩散采样: [B*2*256, 1024] -> [B*256, 16]
               - 重塑为图像: [B, 16, 16, 16]
               - 分辨率变换(如果需要)
        Returns:
            [bsz, C, H, W] 生成的动作序列
        """
        latent_core = [0,2,3,4,5,6,7,8,9,10]
        # latent_core = [0,1,2,3,4,5,6,7,8,9]
        num_iter = len(latent_core)

        # init and sample generation orders
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda().half()
        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        
        for step in indices:
            if labels is not None:
                class_embedding = self.class_emb(labels)        # torch.Size([256, 768])
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            
            tokens = torch.cat([tokens, tokens], dim=0)
            class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)

            # mae encoder
            x = self.forward_mae_encoder(tokens, class_embedding)
            z = self.forward_mae_decoder(x)    # torch.Size([512, 256, 768])    var输出的condition的维度很高(768)，var一次生成所有token后，只随机取部分(nge)送到diffusion中作为条件，生成部分token。

            B, L, C = z.shape
            z = z.reshape(B * L, -1)


            cfg_iter = 1 + (cfg - 1) * step / num_iter
            temperature_iter = 0.8 + (1 - np.cos(math.pi / 2. * (step + 1) / num_iter)) * (1-0.8)

            index = torch.tensor([latent_core[step]]).unsqueeze(1).unsqueeze(-1).repeat(B, L, 1).reshape(B * L, -1).to(z.device).half()
            sampled_token_latent = self.diffloss.sample(z, temperature_iter, cfg_iter, index=index)     # torch.Size([512, 16])
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples.  torch.Size([256, 16])

            sampled_token_latent = sampled_token_latent.reshape(bsz, L, -1).transpose_(1, 2).reshape(bsz, -1, 16, 16)
            if step < num_iter-1:
                if step > -1:
                    sampled_token_latent = F.interpolate(sampled_token_latent, size=(latent_core[step+1], latent_core[step+1]), mode='area')
                    sampled_token_latent = F.interpolate(sampled_token_latent, size=(16, 16), mode='bicubic')
                sampled_token_latent = sampled_token_latent.view(bsz, 16, -1).transpose(1, 2)

            tokens = sampled_token_latent.clone()

        return tokens
        


def far_base(**kwargs):
    model = FAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def far_large(**kwargs):
    model = FAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def far_huge(**kwargs):
    model = FAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



import numpy as np
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock
from diffusion_policy_3d.models.diffloss import DiffLoss
import torch_dct


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda())
    return masking


class Freqpolicy(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, trajectory_dim=26, horizon=16, n_obs_steps=2,
                 encoder_embed_dim=256, encoder_depth=8, encoder_num_heads=8,
                 decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
                 norm_layer=nn.LayerNorm,
                 mask=True,
                 condition_dim=128,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_iter = 4,
                 num_sampling_steps='100',
                 diffusion_batch_mul=1,
                 patch_size=1,
                 **kwargs
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # 轨迹相关设置
        self.trajectory_dim = trajectory_dim  # 动作轨迹的维度
        self.horizon = horizon  # 时间步长
        self.n_obs_steps = n_obs_steps
        self.patch_size = patch_size
        self.seq_len = horizon//self.patch_size  # 序列长度为时间步长除以patch大小
        self.token_embed_dim = trajectory_dim  # token嵌入维度为轨迹维度
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.condition_embed_dim = encoder_embed_dim
        self.buffer_size = 3
        core_2 =  5
        if num_iter == 1:
            self.core = [0]
        else:
            self.core =  [int(i * self.seq_len  / (num_iter - 1)) for i in range(num_iter)]
            if self.core[1] < core_2:
                    # 先放0和core_2，剩下的均匀分布在(core_2, seq_len-1]之间
                    remain = num_iter - 2
                    if remain > 0:
                        # 在(core_2, seq_len-1]之间均匀采样remain个点
                        interval = (self.seq_len  - core_2) / (remain)
                        self.core = [0, core_2] + [int(core_2 + interval * i) for i in range(1, remain+1)]
                    else:
                        self.core = [0, self.seq_len ]
        # --------------------------------------------------------------------------
        # 条件嵌入相关
        self.condition_dim = condition_dim
        # 添加条件投影层，用于将任意维度的条件向量投影到encoder_embed_dim
        self.condition_proj = nn.Linear(condition_dim, self.condition_embed_dim, bias=True)
        self.embedding_index =  nn.Linear(1,  self.condition_embed_dim, bias=True)
        # --------------------------------------------------------------------------
        # 掩码相关设置
        self.mask = mask
        self.loss_weight = [2- np.sin(math.pi / 2. * (bands + 1) / self.seq_len) for bands in range(self.seq_len)]
        # Freqpolicy variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25 
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim)) #(1,1,256)
        # --------------------------------------------------------------------------
        # 编码器相关设置
        self.z_proj = nn.Linear(self.token_embed_dim, self.encoder_embed_dim, bias=True) #(26 → 256)
        self.z_proj_ln = nn.LayerNorm(self.encoder_embed_dim, eps=1e-6) #(256)
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len+self.buffer_size, self.encoder_embed_dim)) #(1,16+1,256)

        self.encoder_blocks = nn.ModuleList([
            BasicTransformerBlock(
                self.encoder_embed_dim,
                encoder_num_heads,
                64,
                dropout=0.0,
                cross_attention_dim=self.encoder_embed_dim,
                activation_fn="geglu",
                attention_bias=True,
                upcast_attention=False,
            ) for _ in range(encoder_depth)])# 256 → 256
        self.encoder_norm = norm_layer(self.encoder_embed_dim)# 256 → 256

        # --------------------------------------------------------------------------
        # 解码器相关设置
        self.decoder_embed = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim, bias=True)# 256 → 256
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len+self.buffer_size , self.decoder_embed_dim))#(1,16+1,256)

        self.decoder_blocks = nn.ModuleList([
            BasicTransformerBlock(
                self.decoder_embed_dim,
                decoder_num_heads,
                64,
                dropout=0.0,
                cross_attention_dim=self.decoder_embed_dim,
                activation_fn="geglu",
                attention_bias=True,
                upcast_attention=False,
            ) for _ in range(decoder_depth)])# 256 → 256

        self.decoder_norm = norm_layer(self.decoder_embed_dim)# 256 → 256
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, self.decoder_embed_dim))#(1,16,256)
        self.initialize_weights()

        # --------------------------------------------------------------------------
        # 扩散损失
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,#26
            z_channels=self.decoder_embed_dim,#256
            width=diffloss_w,#1024
            depth=diffloss_d,#3
            num_sampling_steps=num_sampling_steps,#100
        )
        self.diffusion_batch_mul = diffusion_batch_mul #4

    def initialize_weights(self):
        # 初始化条件投影层
        if self.mask:
            torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # 初始化nn.Linear和nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用xavier_uniform初始化，遵循官方JAX ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def sample_orders(self, bsz):
        # 生成一批随机生成顺序
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # 生成token掩码
        bsz, seq_len, embed_dim = x.shape
        mask_ratio_min = 0.7 
        mask_rate = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25).rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                            src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def processingpregt_dct(self, trajectory):
        """使用GPU加速的DCT变换处理输入动作轨迹
        Args:
            trajectory: [B, horizon, trajectory_dim] 输入动作轨迹
        Returns:
            out: [B, horizon, trajectory_dim] 处理后的轨迹
            core_index: [B] 采样因子索引
        """
        B, H, D = trajectory.shape
        # 定义频域系数的可能范围
        min_core = 0  # 最小保留的频域系数数量
        # 范围：[min_core, H+1)，例如H=16时范围是[0,16]
        # 为每个样本独立选择频域系数数量，避免训练偏差
        chosen_cores = torch.randint(min_core, H+1, (B,))  # [B] 每个样本独立选择
        core_index = chosen_cores.to(trajectory.device, dtype=torch.float32)

        # if self.core is not None:
        #     # 从self.core列表中随机选择一个值
        #     selected_core = random.choice(self.core)
        #     core_index = torch.full((B,), selected_core, device=trajectory.device, dtype=torch.float32)

        # 使用批量矩阵运算处理DCT变换
        # 转置以适应torch_dct的输入格式 [B, D, H]
        traj_reshaped = trajectory.transpose(1, 2).to(torch.float64)
        
        # 执行DCT变换
        dct_coeffs = torch_dct.dct(traj_reshaped, norm='ortho')
        
        # 创建批量掩码 - 高效向量化操作
        # 创建频域索引 [H] 并扩展为 [1, 1, H]
        freq_indices = torch.arange(H, device=trajectory.device).view(1, 1, H)
        # 扩展 core_index 为 [B, 1, 1] 并广播比较
        core_thresholds = core_index.view(B, 1, 1)
        # 向量化掩码：[B, 1, H] -> [B, D, H]
        dct_mask = (freq_indices < core_thresholds).float().expand(B, D, H)
        # 应用掩码并保留指定数量的系数
        masked_coeffs = dct_coeffs * dct_mask
        
        # 执行逆DCT变换
        idct_result = torch_dct.idct(masked_coeffs, norm='ortho').to(trajectory.dtype)
        # 恢复原始形状 [B, H, D]
        out = idct_result.transpose(1, 2)
        
        return out, core_index
    
    def forward_mae_encoder(self, x, condition_embedding, mask=None, index=None):
        """MAE编码器前向传播
        Args:
            x: [bsz, seq_len, token_embed_dim] 例如 [B, 16, 26] patch序列
            condition_embedding: [bsz, n_obs_steps, encoder_embed_dim] 例如 [B, 2, 256] 条件嵌入
            mask: [bsz, seq_len] 掩码矩阵，可为None
            index: [bsz] 当前处理的频域系数索引
        Returns:
            [bsz, 非掩码tokens数, encoder_embed_dim] 编码器输出
        """
        x = self.z_proj(x)  # [B, 16, 26]→ [B, 16, 256]
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # 处理位置嵌入
        if index is not None:
            index = index.unsqueeze(-1).unsqueeze(-1)
            embed_index = self.embedding_index(index)
            # 添加位置嵌入和索引嵌入
            condition_embedding = torch.cat([condition_embedding, embed_index], dim=-2)

        x[:, :self.buffer_size] = condition_embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # 应用掩码，仅在mask_with_buffer不为None时
        if self.mask and mask_with_buffer is not None:
            # dropping
            x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
        
        # 应用Transformer块，使用条件嵌入作为cross attention的输入
        for blk in self.encoder_blocks:
            x = blk(
                x,
                attention_mask=None,
                encoder_hidden_states=condition_embedding,
                timestep=None,
            )
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, condition_embedding, mask=None, index=None):
        """MAE解码器前向传播
        Args:
            x: [bsz, 非掩码tokens数, encoder_embed_dim] 编码器输出
            condition_embedding: [bsz, n_obs_steps, encoder_embed_dim] 条件嵌入
            mask: [bsz, seq_len] 掩码矩阵，可以为None
            index: [bsz] 当前处理的频域系数索引
        Returns:
            [bsz, seq_len, decoder_embed_dim] 解码器输出
        """
        # 对于对称的编码器-解码器，可以保持维度不变
        # 当encoder_embed_dim等于decoder_embed_dim时，这个可以是恒等变换
        # 但为了兼容性，我们保留这个线性层
        x = self.decoder_embed(x)
        bsz = x.size(0)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)
        # 处理索引嵌入
        if index is not None:
            index = index.unsqueeze(-1).unsqueeze(-1)
            embed_index = self.embedding_index(index)
            # 添加位置嵌入和索引嵌入
            condition_embedding = torch.cat([condition_embedding, embed_index], dim=-2)

            
        # 处理掩码，仅在mask不为None且self.mask为True时应用
        if self.mask and mask_with_buffer is not None:
            # pad mask tokens
            mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
            x_after_pad = mask_tokens.clone()
            x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            # 添加位置嵌入和索引嵌入
            x = x_after_pad + self.decoder_pos_embed_learned
        else:
            # 如果没有掩码，直接添加位置嵌入
            x = x + self.decoder_pos_embed_learned 

        # 应用Transformer块
        for blk in self.decoder_blocks:
            x = blk(
                x,
                attention_mask=None,
                encoder_hidden_states=condition_embedding,
                timestep=None,
            )
        x = self.decoder_norm(x)
        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x
    def forward_loss(self, z, target, mask, index, loss_weight=False):
        """计算损失函数
        Args:
            z: [bsz, seq_len, decoder_embed_dim] 解码器输出
            target: [bsz, seq_len, token_embed_dim] 目标动作
            mask: [bsz, seq_len] 掩码矩阵
            index: [bsz] 采样因子索引
            loss_weight: 加权损失的权重值或布尔值
        """
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        index = index.unsqueeze(1).unsqueeze(-1).repeat(1, seq_len, 1).reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        if loss_weight is not False:
            if isinstance(loss_weight, list):
                # 将Python列表转换为PyTorch张量
                loss_weight_tensor = torch.tensor(loss_weight, device=z.device)
                loss_weight = loss_weight_tensor.unsqueeze(0).repeat(bsz, 1)
            # 现在loss_weight是一个[bsz, seq_len]的张量
            loss_weight = loss_weight.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, index=index)
        return loss

    def forward(self, trajectory, conditions, loss_weight=False):
        """前向传播函数
        Args:
            trajectory: [B, horizon, trajectory_dim] 动作轨迹
            conditions: [B, encoder_embed_dim] 条件标签
        """
        B = trajectory.shape[0]
        conditions = conditions.reshape(B, self.n_obs_steps, -1)
        condition_embedding = self.condition_proj(conditions)
        # 处理轨迹数据
        process_trajectory, x_index = self.processingpregt_dct(trajectory)
        # 如果需要使用加权损失，使用预定义的权重
        # 掩码处理
        mask = None
        if self.mask:
            orders = self.sample_orders(bsz=trajectory.size(0))
            mask = self.random_masking(process_trajectory, orders)
        if loss_weight:
            loss_weight = self.loss_weight  # 这里self.loss_weight是一个Python列表
        # 前向编码和解码
        x = self.forward_mae_encoder(process_trajectory, condition_embedding, mask, index=x_index)
        z = self.forward_mae_decoder(x, condition_embedding, mask, index=x_index)
        
        # 计算损失
        loss = self.forward_loss(z=z, target=trajectory, mask=mask, index=x_index, loss_weight=loss_weight)
        return loss

    # def sample_tokens_mask(self, bsz, num_iter=5, conditions=None, cfg=3.0, temperature=1.0):
    #     """使用渐进式掩码的轨迹生成过程"""
    #     # 获取设备信息
    #     device = conditions.device
    #     dtype = conditions.dtype  # 直接从conditions获取数据类型
        
    #     # 初始化掩码、token和生成顺序
    #     mask = torch.ones(bsz, self.seq_len, device=device, dtype=dtype)
    #     tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device=device, dtype=dtype)
    #     orders = self.sample_orders(bsz)
    #     if self.core is not None:
    #         latent_core = self.core

    #     # 预处理条件嵌入
    #     conditions = conditions.to(device=device, dtype=dtype)
    #     # 将条件重塑为[B, n_obs_steps, condition_dim]格式
    #     conditions = conditions.reshape(bsz, self.n_obs_steps, -1)
    #     # 使用条件投影层处理条件
    #     condition_embedding = self.condition_proj(conditions)  # [B, n_obs_steps, encoder_embed_dim]
    #     # 显示进度条
    #     steps = list(range(num_iter))

    #     for step in steps:
    #         # 为当前步骤创建一个新的token张量
    #         current_freq_idx = torch.tensor([latent_core[step]], device=device, dtype=torch.float32).repeat(bsz)
    #         cur_tokens = torch.zeros_like(tokens)

    #         # 编码器-解码器前向传播
    #         x = self.forward_mae_encoder(tokens, condition_embedding, mask, index=current_freq_idx)
    #         z = self.forward_mae_decoder(x, condition_embedding, mask, index=current_freq_idx)
            
    #         B, L, C = z.shape
    #         z = z.reshape(B * L, -1)
    #         # 计算当前步骤的掩码比例和参数
    #         mask_ratio =np.cos(math.pi / 2. * (step + 1) / num_iter)
    #         mask_len = torch.tensor([np.floor(self.seq_len * mask_ratio)], device=device)
    #         temperature_iter = temperature
    #         # 使用预定义的latent_core序列
    #         index = torch.tensor([latent_core[step]], device=device).unsqueeze(1).unsqueeze(-1)
    #         index = index.repeat(B, L, 1).reshape(B * L, -1).to(dtype=torch.float16 if dtype == torch.float16 else torch.float32)
    #         # 执行扩散采样
    #         z = self.diffloss.sample(z, temperature_iter, index=index)
    #         # 重塑为轨迹格式
    #         sampled_token = z.reshape(bsz, L, -1)
    #         # 在当前步骤结束后，如果不是最后一步，对tokens应用DCT变换
    #         if step < num_iter-1:
    #             # 对tokens应用DCT变换，为下一步准备
    #             current_core = latent_core[step+1]
                
    #             # 批量处理DCT变换 - 优化为单次操作
    #             traj_reshaped = sampled_token.transpose(1, 2)  # [B, D, H]
    #             dct_coeffs = torch_dct.dct(traj_reshaped, norm='ortho')  # [B, D, H]
                
    #             # 创建掩码，直接指定保留前current_core个系数
    #             dct_mask = torch.zeros_like(dct_coeffs)
    #             dct_mask[:, :, :current_core] = 1.0
                
    #             # 应用掩码并进行逆DCT变换
    #             filtered_coeffs = dct_coeffs * dct_mask
    #             sampled_token = torch_dct.idct(filtered_coeffs, norm='ortho').transpose(1, 2)  # [B, H, D]
    #         # 更新掩码，生成下一步要预测的位置
    #         mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
    #         mask_to_pred = torch.logical_not(mask_next)
    #         mask = mask_next
    #         # 更新token中需要预测的部分
    #         sampled_token_latent = sampled_token[mask_to_pred.nonzero(as_tuple=True)]
    #         cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
    #         tokens = cur_tokens.clone()
    #     return tokens



    def sample_tokens_mask(self, bsz, num_iter=5, conditions=None, cfg=3.0, temperature=1.0):
        """使用渐进式掩码的轨迹生成过程"""
        # 获取设备信息
        device = conditions.device
        dtype = conditions.dtype  # 直接从conditions获取数据类型
        
        # 初始化掩码、token和生成顺序
        mask = torch.ones(bsz, self.seq_len, device=device, dtype=dtype)
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device=device, dtype=dtype)
        orders = self.sample_orders(bsz)
        if self.core is not None:
            latent_core = self.core

        # 预处理条件嵌入
        conditions = conditions.to(device=device, dtype=dtype)
        # 将条件重塑为[B, n_obs_steps, condition_dim]格式
        conditions = conditions.reshape(bsz, self.n_obs_steps, -1)
        # 使用条件投影层处理条件
        condition_embedding = self.condition_proj(conditions)  # [B, n_obs_steps, encoder_embed_dim]
        # 显示进度条
        steps = list(range(num_iter))
        
        for step in steps:
            # 为当前步骤创建一个新的token张量
            current_freq_idx = torch.tensor([latent_core[step]], device=device, dtype=torch.float32).repeat(bsz)
            cur_tokens = torch.zeros_like(tokens)

            # 编码器-解码器前向传播
            x = self.forward_mae_encoder(tokens, condition_embedding, mask, index=current_freq_idx)
            z = self.forward_mae_decoder(x, condition_embedding, mask, index=current_freq_idx)
            
            B, L, C = z.shape
            z = z.reshape(B * L, -1)
            # 计算当前步骤的掩码比例和参数
            mask_ratio =np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.tensor([np.floor(self.seq_len * mask_ratio)], device=device)
            temperature_iter = temperature
            # 使用预定义的latent_core序列
            index = torch.tensor([latent_core[step]], device=device).unsqueeze(1).unsqueeze(-1)
            index = index.repeat(B, L, 1).reshape(B * L, -1).to(dtype=torch.float16 if dtype == torch.float16 else torch.float32)
            current_steps = None
            # # 计算当前迭代的采样步数，从1开始，最后一次为10，中间均匀分布
            # current_steps = int(6 + (10 - 6) * step / (num_iter - 1)) if num_iter > 1 else 10
            # if steps[-1] == step:
            #     current_steps = 10
            # else:
            #     current_steps = 10
            # 执行扩散采样
            z = self.diffloss.sample(z, temperature_iter, index=index, num_steps=current_steps)
            # 重塑为轨迹格式
            sampled_token = z.reshape(bsz, L, -1)
            if step < num_iter-1:
                # 对tokens应用DCT变换，为下一步准备
                current_core = latent_core[step+1]
                # 批量处理DCT变换 - 优化为单次操作
                traj_reshaped = sampled_token.transpose(1, 2)  # [B, D, H]
                dct_coeffs = torch_dct.dct(traj_reshaped, norm='ortho')  # [B, D, H]
                
                # 创建掩码，直接指定保留前current_core个系数
                dct_mask = torch.zeros_like(dct_coeffs)
                dct_mask[:, :, :current_core] = 1.0
                
                # 应用掩码并进行逆DCT变换
                filtered_coeffs = dct_coeffs * dct_mask
                sampled_token = torch_dct.idct(filtered_coeffs, norm='ortho').transpose(1, 2)  # [B, H, D]
            # 更新掩码，生成下一步要预测的位置
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            mask_to_pred = torch.logical_not(mask_next)
            mask = mask_next
            # 更新token中需要预测的部分
            sampled_token_latent = sampled_token[mask_to_pred.nonzero(as_tuple=True)]
            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()
        return tokens

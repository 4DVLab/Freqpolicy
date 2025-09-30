import numpy as np
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock
from diffusion_policy.model.Freqpolicy.diffloss import DiffLoss
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
        # Trajectory-related settings
        self.trajectory_dim = trajectory_dim  # Dimension of action trajectory
        self.horizon = horizon  # Time horizon
        self.n_obs_steps = n_obs_steps
        self.patch_size = patch_size
        self.seq_len = horizon//self.patch_size  # Sequence length = horizon / patch_size
        self.token_embed_dim = trajectory_dim  # Token embedding dimension = trajectory dimension
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

                    remain = num_iter - 2
                    if remain > 0:

                        interval = (self.seq_len  - core_2) / (remain)
                        self.core = [0, core_2] + [int(core_2 + interval * i) for i in range(1, remain+1)]
                    else:
                        self.core = [0, self.seq_len ]
        # --------------------------------------------------------------------------
        # Condition embedding related
        self.condition_dim = condition_dim
        # Add condition projection layer to project condition vectors of any dimension to encoder_embed_dim
        self.condition_proj = nn.Linear(condition_dim, self.condition_embed_dim, bias=True)
        self.embedding_index =  nn.Linear(1,  self.condition_embed_dim, bias=True)
        # --------------------------------------------------------------------------
        # Masking related settings
        self.mask = mask
        self.loss_weight = [2- np.sin(math.pi / 2. * (bands + 1) / self.seq_len) for bands in range(self.seq_len)]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim)) #(1,1,256)
        # --------------------------------------------------------------------------
        # Encoder related settings
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
        # Decoder related settings
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
        # Diffusion loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,#26
            z_channels=self.decoder_embed_dim,#256
            width=diffloss_w,#1024
            depth=diffloss_d,#3
            num_sampling_steps=num_sampling_steps,#100
        )
        self.diffusion_batch_mul = diffusion_batch_mul #4

    def initialize_weights(self):
        # Initialize condition projection layer
        if self.mask:
            torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def sample_orders(self, bsz):
        # Generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # Generate token masks
        bsz, seq_len, embed_dim = x.shape
        mask_ratio_min = 0.7 
        mask_rate = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25).rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                            src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def processingpregt_dct(self, trajectory):
        """Process input action trajectory using GPU-accelerated DCT transform
        Args:
            trajectory: [B, horizon, trajectory_dim] Input action trajectory
        Returns:
            out: [B, horizon, trajectory_dim] Processed trajectory
            core_index: [B] Sampling factor indices
        """
        B, H, D = trajectory.shape
        # Define possible range of frequency domain coefficients
        min_core = 0  # Minimum number of frequency coefficients to retain
        chosen_cores = torch.randint(min_core, H+1, (B,))  # [B] Independent selection for each sample
        core_index = chosen_cores.to(trajectory.device, dtype=torch.float32)
        # Use batch matrix operations for DCT transform
        # Transpose to adapt to torch_dct input format [B, D, H]
        traj_reshaped = trajectory.transpose(1, 2).to(torch.float64)
        # Execute DCT transform
        dct_coeffs = torch_dct.dct(traj_reshaped, norm='ortho')

        freq_indices = torch.arange(H, device=trajectory.device).view(1, 1, H)

        core_thresholds = core_index.view(B, 1, 1)
        # Vectorized mask: [B, 1, H] -> [B, D, H]
        dct_mask = (freq_indices < core_thresholds).float().expand(B, D, H)
        # Apply mask and retain specified number of coefficients
        masked_coeffs = dct_coeffs * dct_mask
        
        # Execute inverse DCT transform
        idct_result = torch_dct.idct(masked_coeffs, norm='ortho').to(trajectory.dtype)
        # Restore original shape [B, H, D]
        out = idct_result.transpose(1, 2)
        
        return out, core_index
    
    def forward_mae_encoder(self, x, condition_embedding, mask=None, index=None):
        """MAE encoder forward pass
        Args:
            x: [bsz, seq_len, token_embed_dim] e.g., [B, 16, 26] patch sequence
            condition_embedding: [bsz, n_obs_steps, encoder_embed_dim] e.g., [B, 2, 256] condition embedding
            mask: [bsz, seq_len] mask matrix, can be None
            index: [bsz] current frequency domain coefficient index being processed
        Returns:
            [bsz, num_unmasked_tokens, encoder_embed_dim] encoder output
        """
        x = self.z_proj(x)  # [B, 16, 26]→ [B, 16, 256]
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        if index is not None:
            index = index.unsqueeze(-1).unsqueeze(-1)
            embed_index = self.embedding_index(index)
            condition_embedding = torch.cat([condition_embedding, embed_index], dim=-2)

        x[:, :self.buffer_size] = condition_embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        if self.mask and mask_with_buffer is not None:
            x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
        
        # apply transformer blocks
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
        """forward decoder
        Args:
            x: [bsz, unmasked tokens, encoder_embed_dim] encoder output
            condition_embedding: [bsz, n_obs_steps, encoder_embed_dim]
            mask: [bsz, seq_len]
            index: [bsz]
        Returns:
            [bsz, seq_len, decoder_embed_dim] 
        """
        x = self.decoder_embed(x)
        bsz = x.size(0)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        if index is not None:
            index = index.unsqueeze(-1).unsqueeze(-1)
            embed_index = self.embedding_index(index)
            condition_embedding = torch.cat([condition_embedding, embed_index], dim=-2)

            
        # process mask
        if self.mask and mask_with_buffer is not None:
            # pad mask tokens
            mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
            x_after_pad = mask_tokens.clone()
            x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            x = x_after_pad + self.decoder_pos_embed_learned
        else:
            x = x + self.decoder_pos_embed_learned 

        # apply transformer blocks
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
        """
        Args:
            z: [bsz, seq_len, decoder_embed_dim] 
            target: [bsz, seq_len, token_embed_dim] 
            mask: [bsz, seq_len]
            index: [bsz]
            loss_weight:
        """
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        index = index.unsqueeze(1).unsqueeze(-1).repeat(1, seq_len, 1).reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        if loss_weight is not False:
            if isinstance(loss_weight, list):
                loss_weight_tensor = torch.tensor(loss_weight, device=z.device)
                loss_weight = loss_weight_tensor.unsqueeze(0).repeat(bsz, 1)
            loss_weight = loss_weight.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, index=index)
        return loss

    def forward(self, trajectory, conditions, loss_weight=False):
        """forward pass
        Args:
            trajectory: [B, horizon, trajectory_dim]
            conditions: [B, encoder_embed_dim] 
        """
        B = trajectory.shape[0]
        conditions = conditions.reshape(B, self.n_obs_steps, -1)
        condition_embedding = self.condition_proj(conditions)
        # process trajectory
        process_trajectory, x_index = self.processingpregt_dct(trajectory)
        # masking
        mask = None
        if self.mask:
            orders = self.sample_orders(bsz=trajectory.size(0))
            mask = self.random_masking(process_trajectory, orders)
        if loss_weight:
            loss_weight = self.loss_weight  
        # forward encoder and decoder
        x = self.forward_mae_encoder(process_trajectory, condition_embedding, mask, index=x_index)
        z = self.forward_mae_decoder(x, condition_embedding, mask, index=x_index)
        
        # compute loss
        loss = self.forward_loss(z=z, target=trajectory, mask=mask, index=x_index, loss_weight=loss_weight)
        return loss


    def sample_tokens_mask(self, bsz, num_iter=5, conditions=None, cfg=3.0, temperature=1.0):
        """progressive masking trajectory generation"""
        # get device
        device = conditions.device
        dtype = conditions.dtype  # get dtype from conditions
        
        # init mask, token and order
        mask = torch.ones(bsz, self.seq_len, device=device, dtype=dtype)
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device=device, dtype=dtype)
        orders = self.sample_orders(bsz)
        if self.core is not None:
            latent_core = self.core

        # preprocess conditions
        conditions = conditions.to(device=device, dtype=dtype)
        # reshape conditions to [B, n_obs_steps, condition_dim]
        conditions = conditions.reshape(bsz, self.n_obs_steps, -1)
        # process condition
        condition_embedding = self.condition_proj(conditions)  # [B, n_obs_steps, encoder_embed_dim]
        steps = list(range(num_iter))
        
        for step in steps:
            # create new token
            current_freq_idx = torch.tensor([latent_core[step]], device=device, dtype=torch.float32).repeat(bsz)
            cur_tokens = torch.zeros_like(tokens)


            x = self.forward_mae_encoder(tokens, condition_embedding, mask, index=current_freq_idx)
            z = self.forward_mae_decoder(x, condition_embedding, mask, index=current_freq_idx)
            
            B, L, C = z.shape
            z = z.reshape(B * L, -1)
            # compute mask ratio and parameters
            mask_ratio =np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.tensor([np.floor(self.seq_len * mask_ratio)], device=device)
            temperature_iter = temperature
            index = torch.tensor([latent_core[step]], device=device).unsqueeze(1).unsqueeze(-1)
            index = index.repeat(B, L, 1).reshape(B * L, -1).to(dtype=torch.float16 if dtype == torch.float16 else torch.float32)
            current_steps = None
            # diffusion sampling
            z = self.diffloss.sample(z, temperature_iter, index=index, num_steps=current_steps)
            sampled_token = z.reshape(bsz, L, -1)
            if step < num_iter-1:
                # apply DCT to tokens, for next step
                current_core = latent_core[step+1]
                # batch DCT
                traj_reshaped = sampled_token.transpose(1, 2)  # [B, D, H]
                dct_coeffs = torch_dct.dct(traj_reshaped, norm='ortho')  # [B, D, H]
                
                # create mask, directly specify the first current_core coefficients
                dct_mask = torch.zeros_like(dct_coeffs)
                dct_mask[:, :, :current_core] = 1.0
                
                # apply mask and inverse DCT
                filtered_coeffs = dct_coeffs * dct_mask
                sampled_token = torch_dct.idct(filtered_coeffs, norm='ortho').transpose(1, 2)  # [B, H, D]
            # update mask
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            mask_to_pred = torch.logical_not(mask_next)
            mask = mask_next
            # pred next tokens
            sampled_token_latent = sampled_token[mask_to_pred.nonzero(as_tuple=True)]
            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()
        return tokens
import torch.nn.functional as F

from src.deit_vision_transformer import Attention as deit_attention
from .qbias import LearnableBias
from .qlinear import QLinear
from ..quantizer.lsq import LsqQuantizer, LsqQuantizer4v


class QAttention(deit_attention):
    """
    Quantized Attention module using OFQ method:
    - Weight quantization: StatsQ (statistics-based, no learnable parameters)
    - Activation quantization: LSQ (learnable step size)
    
    Supports SQuaT feature distillation by saving quantized activations.
    """
    def __init__(self, m, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable=True,
                 weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq",
                 pretrained_initialized=False,
                 **kwargs):
        assert isinstance(m, deit_attention)
        
        qqkkvv = getattr(m, 'qqkkvv', False)
        
        super().__init__(
            dim=m.qkv.in_features,
            num_heads=m.num_heads,
            attn_drop=m.attn_drop.p,
            proj_drop=m.proj_drop.p,
            qqkkvv=qqkkvv
        )
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        
        # SQuaT: Flag for last attention layer (set by replace_module_by_qmodule_deit)
        self.is_last_attention = False
        self.saved_qact = None

        # QKV projection with quantization (StatsQ for weights, LSQ for activations)
        self.qkv = QLinear(
            m=self.qkv,
            weight_bits=weight_bits,
            input_bits=input_bits,
            weight_channelwise=weight_channelwise,
            input_channelwise=input_channelwise,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
            aq_learnable=aq_learnable,
            wq_learnable=wq_learnable,
            symmetric=True,
            pretrained_initialized=pretrained_initialized
        )
        
        # Output projection with quantization
        self.proj = QLinear(
            m=self.proj,
            weight_bits=weight_bits,
            input_bits=input_bits,
            weight_channelwise=weight_channelwise,
            input_channelwise=input_channelwise,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
            aq_learnable=aq_learnable,
            wq_learnable=wq_learnable,
            symmetric=True,
            pretrained_initialized=pretrained_initialized
        )

        # Per-channel activation quantizers for Q, K, V (after QKV split)
        self.quan_a_q_fn = LsqQuantizer(bit=input_bits, all_positive=False, per_channel=True, learnable=aq_learnable)
        self.quan_a_k_fn = LsqQuantizer(bit=input_bits, all_positive=False, per_channel=True, learnable=aq_learnable)
        self.quan_a_v_fn = LsqQuantizer4v(bit=input_bits, all_positive=False, per_channel=True, learnable=aq_learnable)

        # Learnable biases for optimizing activation distribution (OFQ method)
        self.move_qkv_b4 = LearnableBias(m.qkv.in_features * 3)
        self.move_q_aft = LearnableBias(m.qkv.in_features)
        self.move_k_aft = LearnableBias(m.qkv.in_features)
        self.move_v_aft = LearnableBias(m.qkv.in_features)

        # Softmax output quantizer
        self.quan_a_softmax_fn = LsqQuantizer(bit=input_bits, all_positive=True, per_channel=True, learnable=aq_learnable)

    def forward(self, x):
        """
        Forward pass with quantized attention computation.
        
        Args:
            x: Input tensor (B, N, C)
            
        Returns:
            Output tensor (B, N, C)
        """
        B, N, C = x.shape
        
        # QKV projection with quantization (StatsQ weight + LSQ activation)
        qkv = self.qkv(x)  # (B, N, 3*C)
        
        # Apply pre-quantization bias for QKV
        if self.input_bits < 32:
            qkv = self.move_qkv_b4(qkv)
        
        # Split into Q, K, V
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, C//num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, C//num_heads)

        # Quantize Q and K (per-channel along sequence length dimension)
        q = self.quan_a_q_fn(q)
        k = self.quan_a_k_fn(k)
        
        # Quantize V (per-channel along feature dimension)
        v = v.permute(0, 2, 1, 3).reshape(B, N, C)
        v = self.quan_a_v_fn(v)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Apply post-quantization biases
        if self.input_bits < 32:
            q = q.permute(0, 2, 1, 3).reshape(B, N, C)
            k = k.permute(0, 2, 1, 3).reshape(B, N, C)
            v = v.permute(0, 2, 1, 3).reshape(B, N, C)
            q = self.move_q_aft(q)
            k = self.move_k_aft(k)
            v = self.move_v_aft(v)
        
        # Reshape back to multi-head format
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, num_heads, N, C//num_heads)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Compute attention scores
        attn_weights = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn_prob = F.softmax(attn_weights, dim=-1)

        # Quantize softmax output
        attn_prob = self.quan_a_softmax_fn(attn_prob)
        attn_prob = self.attn_drop(attn_prob)

        # Apply attention to values and reshape
        x = (attn_prob @ v).transpose(1, 2).reshape(B, N, C)
        
        # SQuaT: Save quantized activation for feature distillation
        # Enable saving quantized input in proj layer for last attention (MLPR pattern)
        if hasattr(self, 'is_last_attention') and self.is_last_attention:
            self.proj.save_quantized_input = True
        
        # Output projection with quantization
        x = self.proj(x)
        
        # SQuaT: Retrieve saved quantized activation from proj layer
        # This is the quantized input (input_bits) before move_aft, matching MLPR pattern
        # The saved value is exactly quantized with input_bits (no move_aft bias)
        if hasattr(self, 'is_last_attention') and self.is_last_attention:
            if hasattr(self.proj, 'saved_quantized_input') and self.proj.saved_quantized_input is not None:
                self.saved_qact = self.proj.saved_quantized_input.clone()  # (B, N, C) - input_bits quantized, gradient preserved
            self.proj.save_quantized_input = False  # Reset flag after saving
        
        x = self.proj_drop(x)
        
        return x, None


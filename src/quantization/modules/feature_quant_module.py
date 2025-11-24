"""
Feature Quantizer for ViT models (SQuaT method)
Uses the same LSQ quantization method as student model for consistency
"""
import torch
import torch.nn as nn
from ..quantizer.lsq import LsqQuantizer


class FeatureQuantizerViT(nn.Module):
    """
    Feature Quantizer for Vision Transformer
    Uses the same LsqQuantizer as student model to ensure identical quantization method.
    Quantizes teacher features using student's quantization parameters (s, bit, all_positive).
    """
    def __init__(self, args):
        super(FeatureQuantizerViT, self).__init__()
        # Extract bit width from feature_levels (feature_levels = 2^bit)
        # For example, feature_levels=2 means 1-bit, feature_levels=4 means 2-bit
        if hasattr(args, 'feature_levels'):
            # feature_levels should be 2^bit
            import math
            self.feature_levels = args.feature_levels
            self.bit = int(math.log2(args.feature_levels)) if args.feature_levels > 1 else args.feature_levels
        else:
            self.feature_levels = 4  # Default: 2-bit
            self.bit = 2
        
        self.use_student_quant_params = args.use_student_quant_params if hasattr(args, 'use_student_quant_params') else True
        
        # Use LsqQuantizer (same as student model)
        # all_positive=False for symmetric quantization (signed)
        # per_channel=True to match student's per-channel quantization
        # learnable=False since teacher is frozen and uses student's parameters
        self.lsq_quantizer = LsqQuantizer(
            bit=self.bit,
            all_positive=False,  # Signed quantization (can be overridden by student params)
            per_channel=True,
            learnable=False  # Teacher uses student's parameters, not learnable
        )
        
        # Flag to track if quantizer has been initialized
        self.quantizer_initialized = False
        
        # Store student's quantizer parameters (will be updated each forward pass)
        self.student_s = None
        self.student_all_positive = False

    def forward(self, x, save_dict=None, quant_params=None):
        """
        Forward pass with LSQ quantization (same as student)
        
        Args:
            x: Teacher features (B, N, C)
            save_dict: Dictionary for saving statistics (not used currently)
            quant_params: Dictionary with quantization parameters from student:
                - 's': step size parameter (tensor, per-channel or scalar)
                - 'bit': bit width
                - 'all_positive': whether quantization is unsigned
        
        Returns:
            Quantized features using same LSQ method as student
        """
        # Initialize quantizer on first forward pass
        # LsqQuantizer initializes 's' parameter based on input shape
        # For per-channel quantization, we need to ensure 's' shape matches student's
        if not self.quantizer_initialized:
            with torch.no_grad():
                # If we have student's s parameter, use it to determine shape
                # Otherwise, initialize with input shape
                if quant_params is not None and 's' in quant_params and quant_params['s'] is not None:
                    student_s = quant_params['s']
                    # Use student's s shape to initialize teacher's quantizer
                    if isinstance(student_s, torch.Tensor) and student_s.dim() > 0:
                        # Student's s is per-channel, shape should be (C,) for channel dimension
                        # Initialize dummy input with matching channel dimension
                        C = student_s.shape[0] if len(student_s.shape) == 1 else student_s.numel()
                        dummy_input = torch.zeros(1, C, device=x.device)  # (1, C)
                    else:
                        # Scalar s, initialize normally
                        if len(x.shape) == 3:  # (B, N, C)
                            dummy_input = x.view(-1, x.shape[-1])[0:1]  # (1, C)
                        else:
                            dummy_input = x[0:1] if len(x.shape) >= 2 else x
                else:
                    # No student params yet, initialize with input shape
                    if len(x.shape) == 3:  # (B, N, C)
                        dummy_input = x.view(-1, x.shape[-1])[0:1]  # (1, C) - just first row
                    elif len(x.shape) == 2:  # (B, C)
                        dummy_input = x[0:1]  # (1, C)
                    else:
                        dummy_input = x.view(-1, x.shape[-1])[0:1] if x.numel() > 0 else x
                
                # Initialize quantizer (this sets up 's' parameter shape)
                self.lsq_quantizer.init_from(dummy_input)
                self.quantizer_initialized = True
        
        # Update quantizer parameters from student if provided
        if self.use_student_quant_params and quant_params is not None:
            # Update step size 's' parameter
            if 's' in quant_params and quant_params['s'] is not None:
                student_s = quant_params['s']
                # Ensure student_s is on same device
                if isinstance(student_s, torch.Tensor):
                    if self.lsq_quantizer.s is not None:
                        # Copy student's s to teacher's quantizer
                        if student_s.shape == self.lsq_quantizer.s.shape:
                            self.lsq_quantizer.s.data.copy_(student_s.to(self.lsq_quantizer.s.device))
                        else:
                            # Shape mismatch: try to broadcast or use mean
                            # This should rarely happen if initialization used student's s shape
                            if student_s.dim() == 0:
                                # Scalar s: broadcast to all channels
                                self.lsq_quantizer.s.data.fill_(student_s.item())
                            elif student_s.dim() == 1 and self.lsq_quantizer.s.dim() == 1:
                                # Both 1D: try to slice or pad
                                if student_s.shape[0] == self.lsq_quantizer.s.shape[0]:
                                    self.lsq_quantizer.s.data.copy_(student_s.to(self.lsq_quantizer.s.device))
                                else:
                                    # Different sizes: use mean (info loss, but safe)
                                    s_mean = student_s.mean()
                                    self.lsq_quantizer.s.data.fill_(s_mean.item())
                            else:
                                # Complex mismatch: use mean
                                if student_s.dim() > 0:
                                    s_mean = student_s.mean()
                                else:
                                    s_mean = student_s
                                self.lsq_quantizer.s.data.fill_(s_mean.item())
                    else:
                        # Initialize s if not initialized
                        device = x.device
                        if student_s.dim() > 0:
                            self.lsq_quantizer.s = nn.Parameter(
                                student_s.clone().detach().to(device),
                                requires_grad=False
                            )
                        else:
                            self.lsq_quantizer.s = nn.Parameter(
                                torch.tensor(student_s.item(), device=device),
                                requires_grad=False
                            )
                    self.student_s = student_s
            
            # Update bit width and all_positive if provided
            if 'bit' in quant_params and quant_params['bit'] is not None:
                self.lsq_quantizer.bit = quant_params['bit']
            
            if 'all_positive' in quant_params and quant_params['all_positive'] is not None:
                self.lsq_quantizer.all_positive = quant_params['all_positive']
                self.student_all_positive = quant_params['all_positive']
            
            # Recalculate thresholds
            if self.lsq_quantizer.all_positive:
                if self.lsq_quantizer.bit == 1:
                    self.lsq_quantizer.thd_neg = 0
                    self.lsq_quantizer.thd_pos = 1
                else:
                    self.lsq_quantizer.thd_neg = 0
                    self.lsq_quantizer.thd_pos = 2 ** self.lsq_quantizer.bit - 1
            else:
                if self.lsq_quantizer.bit == 1:
                    self.lsq_quantizer.thd_neg = -1
                    self.lsq_quantizer.thd_pos = 1
                else:
                    self.lsq_quantizer.thd_neg = - 2 ** (self.lsq_quantizer.bit - 1)
                    self.lsq_quantizer.thd_pos = 2 ** (self.lsq_quantizer.bit - 1) - 1
        
        # Quantize using same LSQ method as student
        # Reshape input for quantizer: LsqQuantizer expects 2D (..., C) or 3D (B, N, C)
        original_shape = x.shape
        if len(x.shape) == 3:  # (B, N, C)
            # Quantize along channel dimension (last dim)
            # Reshape to (B*N, C) for per-channel quantization
            x_flat = x.view(-1, x.shape[-1])  # (B*N, C)
            x_quantized = self.lsq_quantizer(x_flat)
            output = x_quantized.view(original_shape)
        else:
            # For other shapes, quantize directly
            output = self.lsq_quantizer(x)
        
        return output


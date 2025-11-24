"""
Utility functions for SQuaT (Student Quantized Activation Feature Distillation)
"""
import torch
import torch.nn as nn
import os


def load_teacher_model(model_t, teacher_path, device='cuda'):
    """
    Load pretrained teacher model and freeze its parameters
    
    Args:
        model_t: Teacher model instance
        teacher_path: Path to teacher checkpoint
        device: Device to load model on
    
    Returns:
        Loaded and frozen teacher model
    """
    if os.path.isfile(teacher_path):
        print(f"Loading teacher checkpoint from '{teacher_path}'")
        checkpoint_t = torch.load(teacher_path, map_location=device)
        
        # Load state dict (strict=False to allow partial loading)
        model_t.load_state_dict(checkpoint_t.get('model', checkpoint_t), strict=False)
        
        # Freeze all parameters
        for name, p in model_t.named_parameters():
            p.requires_grad = False
        
        teacher_num_params = sum(p.numel() for p in model_t.parameters())
        teacher_num_trainable = sum(p.numel() for p in model_t.parameters() if p.requires_grad)
        
        print(f'Teacher model: {teacher_num_params} params, '
              f'{teacher_num_trainable} trainable params\n')
    else:
        raise FileNotFoundError(f"No checkpoint found at '{teacher_path}'")
    
    return model_t


def get_student_quant_params(model_s, feature_extraction_layer='last_attention'):
    """
    Extract quantization parameters from student model
    
    Args:
        model_s: Student model
        feature_extraction_layer: Which layer to extract from
    
    Returns:
        Dictionary with quantization parameters:
            - 's': Step size parameter (tensor, per-channel or scalar)
            - 'bit': Bit width
            - 'all_positive': Whether quantization is unsigned
    """
    quant_params = {
        's': None,
        'bit': None,
        'all_positive': None
    }
    
    # Find the last attention or MLP layer with quantization
    if hasattr(model_s, 'blocks'):
        last_block = model_s.blocks[-1]
        
        # Try to get from attention layer
        if hasattr(last_block, 'attn'):
            attn = last_block.attn
            
            # Check if it's a quantized attention module (QAttention)
            # Extract from proj layer (where saved_qact is stored) instead of qkv
            # This matches MLPR pattern: extract params from the same layer that saves saved_qact
            if hasattr(attn, 'proj'):
                # QAttention uses QLinear for proj
                if hasattr(attn.proj, 'input_quant_fn'):
                    quantizer = attn.proj.input_quant_fn
                    
                    # Check for LSQ quantizer with 's' parameter
                    if hasattr(quantizer, 's'):  # LSQ quantizer
                        if quantizer.s is not None:
                            # Extract all LSQ parameters
                            quant_params['s'] = quantizer.s.clone().detach()  # Per-channel step size
                            quant_params['bit'] = quantizer.bit
                            quant_params['all_positive'] = getattr(quantizer, 'all_positive', False)
                            return quant_params
                    # Try alternative: check if quantizer has step_size or scale
                    elif hasattr(quantizer, 'step_size'):
                        step_size = quantizer.step_size
                        if step_size is not None:
                            quant_params['s'] = step_size.clone().detach() if isinstance(step_size, torch.Tensor) else torch.tensor(step_size)
                            quant_params['bit'] = getattr(quantizer, 'bit', 8)
                            quant_params['all_positive'] = getattr(quantizer, 'all_positive', False)
                            return quant_params
            
            # Fallback: try qkv if proj doesn't have quantizer
            if quant_params['s'] is None and hasattr(attn, 'qkv'):
                # QAttention uses QLinear for qkv
                if hasattr(attn.qkv, 'input_quant_fn'):
                    quantizer = attn.qkv.input_quant_fn
                    
                    # Check for LSQ quantizer with 's' parameter
                    if hasattr(quantizer, 's'):  # LSQ quantizer
                        if quantizer.s is not None:
                            # Extract all LSQ parameters
                            quant_params['s'] = quantizer.s.clone().detach()
                            quant_params['bit'] = quantizer.bit
                            quant_params['all_positive'] = getattr(quantizer, 'all_positive', False)
                            return quant_params
                    # Try alternative: check if quantizer has step_size or scale
                    elif hasattr(quantizer, 'step_size'):
                        step_size = quantizer.step_size
                        if step_size is not None:
                            quant_params['s'] = step_size.clone().detach() if isinstance(step_size, torch.Tensor) else torch.tensor(step_size)
                            quant_params['bit'] = getattr(quantizer, 'bit', 8)
                            quant_params['all_positive'] = getattr(quantizer, 'all_positive', False)
                            return quant_params
        
        # Try to get from MLP layer
        if quant_params['s'] is None and hasattr(last_block, 'mlp'):
            mlp = last_block.mlp
            if hasattr(mlp, 'fc1') and hasattr(mlp.fc1, 'input_quant_fn'):
                quantizer = mlp.fc1.input_quant_fn
                if hasattr(quantizer, 's'):
                    if quantizer.s is not None:
                        # Extract all LSQ parameters
                        quant_params['s'] = quantizer.s.clone().detach()
                        quant_params['bit'] = quantizer.bit
                        quant_params['all_positive'] = getattr(quantizer, 'all_positive', False)
                        return quant_params
    
    # If no quantization params found, return default values
    if quant_params['s'] is None:
        # Default: use reasonable values
        quant_params['s'] = torch.tensor(1.0/3.0)  # Default for 2-bit signed
        quant_params['bit'] = 2
        quant_params['all_positive'] = False
        # Only warn once, not every iteration
        if not hasattr(get_student_quant_params, '_warned_defaults'):
            print("Warning: Could not extract quantization parameters, using defaults")
            get_student_quant_params._warned_defaults = True
    
    return quant_params


def create_adaptor(dim, use_bn=False, adaptor_type='linear'):
    """
    Create adaptor module for matching student and teacher features
    
    Args:
        dim: Feature dimension
        use_bn: Whether to use batch normalization
        adaptor_type: Type of adaptor ('linear', 'mlp', 'conv')
    
    Returns:
        Adaptor module
    """
    if adaptor_type == 'linear':
        layers = [nn.Linear(dim, dim, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm1d(dim))
        return nn.Sequential(*layers)
    
    elif adaptor_type == 'mlp':
        layers = [
            nn.Linear(dim, dim, bias=False),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False)
        ]
        if use_bn:
            layers.insert(1, nn.BatchNorm1d(dim))
        return nn.Sequential(*layers)
    
    else:
        raise ValueError(f"Unknown adaptor type: {adaptor_type}")


def compute_feature_distillation_loss(fd_map_s, fd_map_t, loss_type='L2', T=4.0):
    """
    Compute feature distillation loss
    
    Args:
        fd_map_s: Student feature map (quantized activation)
        fd_map_t: Teacher feature map (quantized feature)
        loss_type: Loss type ('L1', 'L2', 'KL_Div')
        T: Temperature for KL divergence (default: 4.0)
    
    Returns:
        Loss value
    """
    if loss_type == 'L1':
        return nn.L1Loss()(fd_map_s, fd_map_t)
    elif loss_type == 'L2':
        return nn.MSELoss()(fd_map_s, fd_map_t)
    elif loss_type == 'KL_Div':
        # Flatten features for KL divergence
        batch_size = fd_map_s.shape[0]
        fd_s_flat = fd_map_s.view(batch_size, -1)
        fd_t_flat = fd_map_t.view(batch_size, -1)
        
        # Convert to probabilities with temperature
        prob_s = torch.softmax(fd_s_flat / T, dim=1)
        prob_t = torch.softmax(fd_t_flat / T, dim=1)
        
        # KL divergence
        log_prob_s = torch.log_softmax(fd_s_flat / T, dim=1)
        kl_loss = nn.KLDivLoss(reduction='batchmean')(log_prob_s, prob_t)
        return kl_loss * (T ** 2)  # Scale by temperature squared
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


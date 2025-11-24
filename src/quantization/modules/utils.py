import torch
# from .conv import QConv2d, QConvBn2d
# from .linear import QLinear
from .qlinear import QLinear, QMLP
from .attention import QAttention

# Note: QAttention_qkreparam, QAttention_qkreparam_4_cga, QAttention_lsq are removed
# as they are not used in OFQ_SQuaT project (only QAttention is used)
# If needed in the future, these imports can be restored:
# from .attention import QAttention_lsq, QAttention_qkreparam, QAttention_qkreparam_4_cga

# from src.utils import Attention
from src.deit_vision_transformer import Attention as deit_attention 
from src.deit_vision_transformer import Mlp

# Import timm's Attention for compatibility (not used in current project, but kept for compatibility)
try:
    from timm.models.vision_transformer import Attention as timm_attention
    HAS_TIMM_ATTENTION = True
except ImportError:
    timm_attention = None
    HAS_TIMM_ATTENTION = False

QMODULE_MAPPINGS = {
    torch.nn.Linear: QLinear,
    deit_attention: QAttention,
    Mlp: QMLP
}
# Add timm Attention mapping if available
if HAS_TIMM_ATTENTION and timm_attention is not None:
    QMODULE_MAPPINGS[timm_attention] = QAttention

# Note: QMODULE_MAPPINGS_QK_REPARAM and QMODULE_MAPPINGS_W_AND_ACT are kept for compatibility
# but they reference classes that are no longer available. If qk_reparam=True or 
# both weight/act modes are 'lsq', the code will need to be updated to restore those classes.
# For now, using QMODULE_MAPPINGS (QAttention) only.

# QMODULE_MAPPINGS_QK_REPARAM = [...]  # Disabled - requires QAttention_qkreparam classes
# QMODULE_MAPPINGS_W_AND_ACT = {...}  # Disabled - requires QAttention_lsq class
def get_module_by_name(model, module_name):
    names = module_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)
    return module


def set_module_by_name(model, module_name, module):
    if module_name == 'head' or module_name == 'head_dist':
        setattr(model, module_name, module)
    else:
        names = module_name.split(".")
        parent = get_module_by_name(model, ".".join(names[:-1]))
        setattr(parent, names[-1], module)


def replace_module_by_qmodule_deit(model, qconfigs, pretrained_initialized = False,
                        qk_reparam = False, qk_reparam_type = 0, boundaryRange = 0.005): 
        # Current OFQ_SQuaT project uses: wq_mode='statsq', aq_mode='lsq', qmodules=['blocks'], qk_reparam=False
        # Only blocks are quantized (patch_embed.proj and head are not in qmodules)

        # Process quantization configs
        for name, cfg in qconfigs.items():
            if name == "blocks":
                # Handle blocks (nn.Sequential) by processing each Block's attn and mlp
                if hasattr(model, 'blocks'):
                    for i, block in enumerate(model.blocks):
                            # Process attention module
                            if hasattr(block, 'attn'):
                                attn_module = block.attn
                                # Check both exact type and base class
                                attn_type = type(attn_module)
                                if attn_type in QMODULE_MAPPINGS:
                                    pass  # Found exact match
                                elif HAS_TIMM_ATTENTION and isinstance(attn_module, timm_attention) and timm_attention in QMODULE_MAPPINGS:
                                    attn_type = timm_attention  # Use timm_attention mapping
                                elif isinstance(attn_module, deit_attention) and deit_attention in QMODULE_MAPPINGS:
                                    attn_type = deit_attention  # Use deit_attention mapping
                                
                                if attn_type in QMODULE_MAPPINGS:
                                    # Get device from original module
                                    device = next(attn_module.parameters()).device
                                    qmodule_attn = QMODULE_MAPPINGS[attn_type](
                                        m = attn_module,
                                        weight_bits = cfg["weight"]['bit'],
                                        input_bits = cfg["act"]['bit'],
                                        weight_channelwise = cfg["weight"]["per_channel"],
                                        input_channelwise = cfg["act"]["per_channel"],
                                        weight_quant_method = cfg["weight"]["mode"],
                                        input_quant_method = cfg["act"]["mode"],
                                        aq_learnable = cfg["act"]["learnable"],
                                        wq_learnable = cfg["weight"]["learnable"],
                                        act_layer = cfg["act_layer"],
                                        pretrained_initialized = pretrained_initialized
                                    ).to(device)  # Move to same device as original module
                                    block.attn = qmodule_attn
                            
                            # Process MLP module
                            if hasattr(block, 'mlp'):
                                mlp_module = block.mlp
                                if type(mlp_module) in QMODULE_MAPPINGS:
                                    # Get device from original module
                                    device = next(mlp_module.parameters()).device
                                    qmodule_mlp = QMODULE_MAPPINGS[type(mlp_module)](
                                        m = mlp_module,
                                        weight_bits = cfg["weight"]['bit'],
                                        input_bits = cfg["act"]['bit'],
                                        weight_channelwise = cfg["weight"]["per_channel"],
                                        input_channelwise = cfg["act"]["per_channel"],
                                        weight_quant_method = cfg["weight"]["mode"],
                                        input_quant_method = cfg["act"]["mode"],
                                        aq_learnable = cfg["act"]["learnable"],
                                        wq_learnable = cfg["weight"]["learnable"],
                                        act_layer = cfg["act_layer"],
                                        pretrained_initialized = pretrained_initialized
                                    ).to(device)  # Move to same device as original module
                                    block.mlp = qmodule_mlp
        
        # SQuaT: Mark last attention layer for feature extraction
        if hasattr(model, 'blocks'):
            last_block = model.blocks[-1]
            if hasattr(last_block, 'attn'):
                last_attn = last_block.attn
                if hasattr(last_attn, 'is_last_attention'):
                    last_attn.is_last_attention = True
                    print("Marked last attention layer for SQuaT feature extraction")

        return model

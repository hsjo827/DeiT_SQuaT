import torch
import torch.nn as nn

from .qbias import LearnableBias
from ..quantizer.statsq import StatsQuantizer
from ...deit_vision_transformer import Mlp
from timm.models.layers import to_2tuple
from ..quantizer.lsq import LsqQuantizer

class QLinear(nn.Linear):

    def __init__(self, *kargs, m: torch.nn.Linear, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True,
                 symmetric=True, weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq",
                 pretrained_initialized = False,
                 **kwargs):
        super(QLinear, self).__init__(m.in_features, m.out_features,bias=True)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.aq_learnable = aq_learnable
        self.wq_learnable = wq_learnable
        self.symmetric = symmetric
        self.weight_channelwise = weight_channelwise # not gonna used atm
        self.input_channelwise = input_channelwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self.input_quant_fn = LsqQuantizer(bit=input_bits,all_positive=(symmetric==False), learnable =  aq_learnable)
        self.pretrained_initialized = pretrained_initialized
        if pretrained_initialized != False:
            self.weight = torch.nn.Parameter(m.weight.detach())
            if m.bias is not None:
                self.bias = torch.nn.Parameter(m.bias.detach())
        if weight_quant_method == 'statsq':
            self.statsq_fn = StatsQuantizer(num_bits=self.weight_bits, clip_learnable=wq_learnable).to(m.weight.device)
        else:
            raise ValueError("Unknown quant_method")        

        self.move_b4 = LearnableBias(self.weight.shape[1])
        self.move_aft = LearnableBias(self.weight.shape[1])
        
        # SQuaT: Flag to save quantized input for feature distillation
        self.save_quantized_input = False
        self.saved_quantized_input = None

    def forward(self, input):

        # quantize weight
        if self.weight_quant_method == 'statsq':
            weight = self.statsq_fn(self.weight)
        else:
            raise ValueError("Unknown quant_method")    
        # quantize input
        input = self.move_b4(input)
        input_quantized = self.input_quant_fn(input)  # input_bits로 양자화된 값
        
        # SQuaT: Save quantized input (before move_aft) if flag is set
        # This ensures saved value is exactly quantized with input_bits (matching MLPR pattern)
        if self.save_quantized_input:
            self.saved_quantized_input = input_quantized.clone()  # input_bits bit로 양자화된 값
        
        input = self.move_aft(input_quantized)
        out = nn.functional.linear(input, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def extra_repr(self):
        return (
            f"act_bit={self.input_bits}, "
            f"weight_bit={self.weight_bits}, "
            f"act_all_positive={not self.symmetric}, "
            f"wq_learnable={self.wq_learnable}, "
            f"aq_learnable={self.aq_learnable}, "
            f"weight_channelwise ={self.weight_channelwise}, "
            f"input_channelwise ={self.input_channelwise}, "
            f"weight_quant_method={self.weight_quant_method}, "
            f"activation_quant_method={self.input_quant_method}, "
            f"pretrained_initialized = {self.pretrained_initialized}"
        )

class QMLP(Mlp):
    def __init__(self, *kargs, m: Mlp, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True, weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq", act_layer=nn.GELU,
                    pretrained_initialized = False,
                    **kwargs):
            super().__init__(
                in_features = m.in_features, 
                hidden_features = m.hidden_features,
                out_features = m.out_features, 
                drop = m.drop
            )
                
            out_features = m.out_features or m.in_features
            hidden_features = m.hidden_features or m.in_features
            drop_probs = to_2tuple(self.drop)

            self.fc1 = QLinear(m = m.fc1,weight_bits=weight_bits,input_bits=input_bits,
            aq_learnable=aq_learnable,wq_learnable=wq_learnable,symmetric=True,weight_channelwise=weight_channelwise,input_channelwise=input_channelwise,
            weight_quant_method=weight_quant_method,input_quant_method=input_quant_method, pretrained_initialized = pretrained_initialized)

            self.act_layer = act_layer
            
            if act_layer != 'rprelu':
                if act_layer != 'None':
                    self.act = act_layer()
                else:
                    self.act = nn.Identity()
                

            self.drop1 = nn.Dropout(drop_probs[0])
            self.fc2 = QLinear(m = m.fc2,weight_bits=weight_bits,input_bits=input_bits,
            aq_learnable=aq_learnable,wq_learnable=wq_learnable,symmetric=False,weight_channelwise=weight_channelwise,input_channelwise=input_channelwise,
            weight_quant_method=weight_quant_method,input_quant_method=input_quant_method, pretrained_initialized = pretrained_initialized)
            self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        
        x = self.fc1(x)
        if self.act_layer != 'rprelu':
            x = self.act(x)
        else:
            x = self.move1(x)
            x = self.act(x)
            x = self.move2(x)

        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x






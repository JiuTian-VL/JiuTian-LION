import math

import torch
import torch.nn as nn

from models.modeling_t5 import T5LayerFF

import torch
import torch.nn as nn
from transformers import BertLayer, BertConfig
from transformers.models.bert.modeling_bert import BertOutput, BertSelfOutput

class FusionAdapter(nn.Module):
    def __init__(
        self,
        num_blocks: int = 2,
        dim: int = 1408,
        num_heads: int = 16,
    ):
        super().__init__()
        config = BertConfig(
            hidden_size=dim,
            num_attention_heads=num_heads
        )
        config.add_cross_attention = True
        config.is_decoder = True
        self.config = config
        self.blocks = nn.ModuleList([BertLayer(config) for _ in range(num_blocks)])
        self.apply(self._init_weights)

    def forward(self, hidden_states, encoder_hidden_states):
        if isinstance(encoder_hidden_states, list):
            assert len(encoder_hidden_states) == len(self.blocks)
            for idx,block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states[idx],
                )[0]
        else:
            for idx,block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                )[0]
        return hidden_states
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, BertSelfOutput) or isinstance(module, BertOutput):
            module.dense.weight.data.zero_()
            module.dense.bias.data.zero_()

class Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=64,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="learnable_scalar",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
                
    def forward(self, x):
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        return up

class AdapterRouter(nn.Module):
    def __init__(self,
        d_model,
        bottleneck=64,
        dropout=0.0,
        init_option="lora",
        adapter_scalar="learnable_scalar",
        adapter_layernorm_option="none",
        num_adapters = 2,
    ):
        super().__init__()
        self.adapters = nn.ModuleList([])
        for _ in range(num_adapters):
            self.adapters.append(Adapter(
                d_model=d_model,
                bottleneck=bottleneck,
                dropout=dropout,
                init_option=init_option,
                adapter_scalar=adapter_scalar,
                adapter_layernorm_option=adapter_layernorm_option
            ))
        if num_adapters > 1:
            self.router_ratio1 = nn.Parameter(
                torch.tensor([[1],[0]],dtype=torch.float32).repeat(1,d_model)
            )
            self.router_ratio2 = nn.Parameter(
                torch.tensor([[0],[1]],dtype=torch.float32).repeat(1,d_model)
            )
        self.num_adapters = num_adapters
        self.router_idx = None
        
    def forward(self, x):
        assert self.router_idx in [0,1]
        output1 = self.adapters[0](x)
        if self.num_adapters == 1:
            return output1
        
        output2 = self.adapters[1](x)
        ratio1 = self.router_ratio1[self.router_idx]
        ratio2 = self.router_ratio2[self.router_idx]
        return output1 * ratio1 + output2 * ratio2
    
def forward_ffn_t5(self, hidden_states):
    adapt_hidden_states = self.adapter(hidden_states)
    forwarded_states = self.layer_norm(hidden_states)
    forwarded_states = self.DenseReluDense(forwarded_states)
    hidden_states = hidden_states + self.dropout(forwarded_states)
    return hidden_states + adapt_hidden_states

def set_adapter_t5(model: nn.Module, d_model: int, n: int, bottleneck: int = 64):
    for c in model.children():
        if isinstance(c, T5LayerFF):
            c.adapter = AdapterRouter(d_model=d_model, bottleneck=bottleneck, num_adapters=n)
            bound_method = forward_ffn_t5.__get__(c, c.__class__)
            setattr(c, 'forward', bound_method)
        elif len(list(c.children())) > 0:
            set_adapter_t5(c, d_model, n, bottleneck)
        
def set_router_idx(model: nn.Module, idx: int):
    for c in model.children():
        if isinstance(c, AdapterRouter):
            c.router_idx = idx
        elif len(list(c.children())) > 0:
            set_router_idx(c, idx)
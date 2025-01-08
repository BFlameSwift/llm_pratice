import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from utils import recursive_getattr, recursive_setattr


class LoRALinear(torch.nn.Module):
    def __init__(self, weight, bias, lora_dim, lora_scaling):
        super(LoRALinear, self).__init__()
        # Save original weight and bias
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)
        # TODO: Implement lora left and right weights
        # 参考 https://github.com/microsoft/LoRA  lora的实现， loralib/layers.py
        self.lora_right_weight = nn.Parameter(self.weight.new_zeros((self.weight.size(0), lora_dim)))
        self.lora_left_weight = nn.Parameter(self.weight.new_zeros((lora_dim, self.weight.size(1))))
        #############################################
        self.lora_scaling = lora_scaling / lora_dim
        self.init_parameters()
        # TODO: Freeze original weight and bias
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        #
        #######################################

    def init_parameters(self):
        # TODO: Initialize LoRA parameters
        nn.Linear.reset_parameters(self)
        
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)
        # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # nn.init.zeros_(self.lora_B)
        # raise NotImplementedError
        ##################################

    def forward(self, input):
        # TODO: Implement the forward function
        # 1. Calculate the linear output using the original weight and bias
        result = F.linear(input, self.weight, self.bias)
        result += self.lora_right_weight @ self.lora_left_weight * self.lora_scaling
        return result
        # raise NotImplementedError
        ######################################


def convert_linear_layer_to_lora(model, part_module_name, lora_dim=0, lora_scaling=1):
    replace_name = []
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear) or isinstance(module, transformers.pytorch_utils.Conv1D)) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        if isinstance(module, torch.nn.Linear):
            tmp = LoRALinear(module.weight, module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            tmp = LoRALinear(module.weight.t().detach(), module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        else:
            raise ValueError("Unsupported module type")
        recursive_setattr(model, name, tmp)
    return model


def only_optimize_lora_parameters(model):
    # study from  loralib/utils.py
    # TODO: Turn off the gradient of all the parameters except the LoRA parameters
    # raise NotImplementedError
    for n, p in model.named_parameters():
        if 'lora_' not in n:
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad = True
    ##############################################################################

def get_lora_state_dict(model):
    # TODO: return lora left and right weights as state dict
    # The saved state dict will be used later for loading
    # raise NotImplementedError
    my_state_dict = model.state_dict()
    to_return = {}
    for k in my_state_dict:
        if 'lora_' in k:
            to_return[k] = my_state_dict[k]
            bias_name = k.split('lora_')[0]+'bias'
            if bias_name in my_state_dict:
                to_return[bias_name] = my_state_dict[bias_name]
    return to_return
    ########################################################
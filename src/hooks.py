import torch

from lib import AlphaOptions, register_alpha_map_hook
from src.forward import (ca_forward, crossattnupblock2d_forward,
                        upblock2d_forward)

def register_forward_hooks(model: torch.nn.Module, alpha_masks: dict):
    target_module_names = ['UNetMidBlock2DCrossAttn', 'CrossAttnUpBlock2D', 'UpBlock2D']
    for name, module in model.named_modules():
        if module.__class__.__name__ in target_module_names:
            register_alpha_map_hook(module, module_name=name, store_loc=alpha_masks)

def replace_call_methods(module: torch.nn.Module):
    for name, subnet in module.named_children():
        if subnet.__class__.__name__ == 'CrossAttnUpBlock2D':
            subnet.forward = crossattnupblock2d_forward(subnet)
        if subnet.__class__.__name__ == 'UpBlock2D':
            subnet.forward = upblock2d_forward(subnet)
        elif hasattr(subnet, 'children'):
            replace_call_methods(subnet)

def register_unet(pipe, mask_options: AlphaOptions, reset_masks=True):
    if reset_masks:
        pipe.unet.alpha_masks = dict()
    pipe.unet.forward = ca_forward(pipe.unet.cuda(), mask_options=mask_options)
    register_forward_hooks(pipe.unet, pipe.unet.alpha_masks)
    replace_call_methods(pipe.unet)

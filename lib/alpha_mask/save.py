import os
import cv2

from tqdm import tqdm
import numpy as np
import torch
from torch import Size, Tensor
from torch.nn import Module
from torch.nn import functional as F
from torchvision.transforms import ToPILImage

from ..utils import assert_path, gray2colormap


def get_empty_mask(alpha_masks: dict, unconditional: bool):
	total_alpha_mask = list(list(alpha_masks.values())[0].values())[0]
	if unconditional:
		total_alpha_mask = total_alpha_mask.chunk(2)[1]  # (batch, channel, height, width)
	return torch.zeros_like(total_alpha_mask)

def resize_alpha_mask(alpha_mask: Tensor, size: Size, unconditional: bool):
	reshaped = alpha_mask
	if unconditional:
		reshaped = alpha_mask.chunk(2)[1]

	resized = F.interpolate(reshaped, size=size, mode='bilinear', align_corners=False)
	return (reshaped, resized)

def save_alpha_masks(alpha_masks: dict, base_dir='log/alpha_masks', unconditional=True):
	to_pil = ToPILImage()

	assert_path(base_dir)

	# total_alpha_mask  = get_empty_mask(alpha_masks, unconditional=unconditional)
	total_alpha_mask = torch.zeros(1, 1, 512, 512)
	total_alpha_mask_shape = total_alpha_mask.shape[-2:]
	total_alpha_mask_num = 0

	for timestep, layers in tqdm(alpha_masks.items(), desc="Saving alpha masks"):
		timestep_dir = os.path.join(base_dir, f'{timestep}')
		assert_path(timestep_dir)

		for layer, alpha_mask in layers.items():
			layer_dir = os.path.join(timestep_dir, f'{layer}')
			assert_path(layer_dir)

			_, resized = resize_alpha_mask(
				alpha_mask,
				size=total_alpha_mask_shape,
				unconditional=unconditional
			)
			total_alpha_mask += resized
			total_alpha_mask_num += 1

			for batch, mask in enumerate(resized):
				batch_dir = os.path.join(layer_dir, f'batch-{batch}')
				assert_path(batch_dir)

				colormap = gray2colormap(mask.squeeze().to(torch.float32))
				# colormap.save(os.path.join(batch_dir, f"alpha_mask.png"))
				cv2.imwrite(os.path.join(batch_dir, f"alpha_mask.png"), colormap)
				# np.savetxt(os.path.join(batch_dir, f"alpha_mask.txt"), mask.to(torch.float32).squeeze().numpy(), fmt="%.4f")

	total_alpha_mask /= total_alpha_mask_num
	for batch, mask in enumerate(total_alpha_mask):
		batch_dir = os.path.join(base_dir, f'batch-{batch}')
		assert_path(batch_dir)

		colormap = gray2colormap(mask.squeeze().to(torch.float32))
		# colormap.save(os.path.join(batch_dir, f"alpha_mask.png"))
		cv2.imwrite(os.path.join(batch_dir, f"alpha_mask.png"), colormap)
		np.savetxt(os.path.join(batch_dir, f"alpha_mask.txt"), mask.to(torch.float32).squeeze().numpy(), fmt="%.4f")


def store_alpha_mask(module: Module, module_name: str, store_loc: dict, detach=True):
	if hasattr(module, "alpha_mask"):
		timestep = module.timestep

		store_loc[timestep] = store_loc.get(timestep, dict())
		store_loc[timestep][module_name] = module.alpha_mask.cpu() if detach \
			else module.alpha_mask

def register_alpha_map_hook(module: Module, module_name: str, store_loc: dict):
	def hook_function(module: Module, name: str):
		return lambda module, input, output: store_alpha_mask(
			module,
			module_name=name,
			store_loc=store_loc
		)
	module.store_alpha_mask = True
	handle = module.register_forward_hook(hook_function(module, module_name))

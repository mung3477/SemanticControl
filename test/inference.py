import os
from typing import List, TypedDict

import torch
from controlnet_aux import CannyDetector, OpenposeDetector, ZoeDetector
from diffusers import AutoencoderKL, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
from pytorch_lightning import seed_everything

from lib import (assert_path, image_grid, init_store_attn_map,
                 save_alpha_masks, save_attention_maps)
from src import SemanticControlPipeline, register_unet

from .types import ModelType


class GenerateParam(TypedDict):
	seed: int
	prompt: str
	ignore_special_tkns: bool

class EvalModel():
	base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"
	vae_model_path = "stabilityai/sd-vae-ft-mse"
	control: str = "depth"	# "depth" | "pose" | "canny"
	generate_param: GenerateParam = {}

	def _control_setup(self, control: str):
		if control == "depth":
			self.controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"
			self.preprocessor = ZoeDetector.from_pretrained("lllyasviel/Annotators")
		elif control == "canny":
			self.controlnet_path = "lllyasviel/control_v11p_sd15_canny"
			self.preprocessor = CannyDetector()
		elif control == "pose":
			self.controlnet_path = "lllyasviel/control_v11p_sd15_openpose"
			self.preprocessor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

	def __init__(self, control: str):
		assert control in ["depth", "pose", "canny"], f"control argument should be one of depth, pose, canny, but you provied {control}"
		self.control = control

		self._control_setup(control)
		vae = AutoencoderKL.from_pretrained(self.vae_model_path).to(dtype=torch.float16)
		controlnet = ControlNetModel.from_pretrained(self.controlnet_path, torch_dtype=torch.float16)

		pipe =	SemanticControlPipeline.from_pretrained(
				pretrained_model_name_or_path=self.base_model_path,
				controlnet=controlnet,
				vae=vae,
	   		torch_dtype=torch.float16
		)
		pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
		pipe.enable_model_cpu_offload()

		self.pipe = pipe

	def _prepare_control(self, reference: str):
		image = Image.open(reference)
		control_img = self.preprocessor(image)

		self.reference = image
		self.control_img = control_img
		return control_img

	def _record_generate_params(self, seed: int,  prompt: str, ignore_special_tkns: bool):
		self.generate_param["seed"] = seed
		self.generate_param["prompt"] = prompt
		self.generate_param["ignore_special_tkns"] = ignore_special_tkns

	def get_inference_func(self, modelType: ModelType):
		self.modelType = modelType

		if modelType == ModelType.ControlNet:
			return self.inference_ControlNet
		elif modelType == ModelType.SemanticControl:
			return self.inference_SemanticControl

		return None

	def set_output_dir(self, output_dir: str):
		assert_path(output_dir)
		self.output_dir = output_dir

	@staticmethod
	def _filepath2name(filepath: str):
		no_extension = filepath.split(".")[0]
		name = " ".join(no_extension.split("/")[-2:])
		return name


	def _is_already_generated(self, ref_subj, prmpt_subj, prompt, seed, alpha_mask=[1.0], prefix=""):
		save_dir = f"{self.output_dir}/{ref_subj}/{prmpt_subj}/{prompt}/{self.modelType.name}/{self.control}"
		filename = f"{save_dir}/{prefix}seed {seed}.png"
		if self.modelType.name == "ControlNet":
			filename = f"{save_dir}/alpha {alpha_mask} - seed {seed}.png"

		self.save_dir = save_dir
		self.filename = filename

		return os.path.exists(filename)

	def inference_ControlNet(self, prompt: str, reference: str, ref_subj: str, prmpt_subj: str, seed: int, alpha_mask: List[float] = [1.0], **kwargs):
		if self._is_already_generated(ref_subj, prmpt_subj, prompt, seed, alpha_mask):
			print(f"{self.filename} is already generated. Skipping.")
			return None
			# print(f"{self.filename} is already generated. Overwriting.")

		control_img = self._prepare_control(reference)
		pipe_options = {
			"ignore_special_tkns": True
		}
		self.pipe.options = pipe_options

		init_store_attn_map(self.pipe)
		register_unet(self.pipe, mask_options={
			"alpha_mask": alpha_mask,
			"fixed": True
		})

		self._record_generate_params(seed=seed, prompt=prompt, ignore_special_tkns=True)

		seed_everything(seed)
		output = self.pipe(
			prompt=prompt,
			image=control_img
		).images[0]

		return output

	def inference_SemanticControl(self, prompt: str, reference: str, ref_subj: str, prmpt_subj: str, seed: int, mask_prompt, focus_tokens, save_attn: bool = False, **kwargs):
		use_attn_bias = "use_attn_bias" in kwargs and kwargs["use_attn_bias"] is True
		filename_prefix = "" if "filename_prefix" not in kwargs else kwargs["filename_prefix"]

		if self._is_already_generated(ref_subj, prmpt_subj, prompt, seed, prefix=filename_prefix):
			# print(f"{self.filename} is already generated. Overwriting.")
			print(f"{self.filename} is already generated. Skipping.")
			return


		control_img = self._prepare_control(reference)

		pipe_options = {
			"ignore_special_tkns": True,
			"ref_subj": ref_subj,
			"prmpt_subj": prmpt_subj
		}
		self.pipe.options = pipe_options

		init_store_attn_map(self.pipe, use_attn_bias=use_attn_bias)
		register_unet(
			pipe=self.pipe,
			mask_options={
				"alpha_mask": [1.0],
				"fixed": True
			}
		)

		seed_everything(seed)

		spatial_sample = self.pipe(
			prompt=mask_prompt,
			image=control_img,
			prepare_phase=True,
		).images[0]
		assert_path(self.save_dir)
		spatial_sample.save(self.save_dir + "/spatial_sample.png")

		if save_attn:
			save_attention_maps(
				self.pipe.unet.attn_maps,
				self.pipe.tokenizer,
				base_dir=f"{os.getcwd()}/log/attn_maps/{self.modelType.name}/{prompt}/{mask_prompt}",
				prompts=[mask_prompt],
				options={
					"prefix": "",
					"return_dict": False,
					"ignore_special_tkns": True,
					"enabled_editing_prompts": 0
			})

		self._record_generate_params(seed=seed, prompt=prompt, ignore_special_tkns=True)

		register_unet(
			pipe=self.pipe,
			mask_options={
				"alpha_mask": [1.0],
				"fixed": True
			},
			reset_masks=False
		)
		output = self.pipe(
			prompt=prompt,
			mask_prompt=mask_prompt,
			focus_prompt = focus_tokens,
			image=control_img,
			prepare_phase=False,
			use_attn_bias=use_attn_bias
		).images[0]

		if save_attn:
			save_alpha_masks(self.pipe.unet.alpha_masks, f'{os.getcwd()}/log/alpha_masks/{self.modelType.name}/{prompt}')

		return output

	def postprocess(self, image, save_attn: bool = False):
		if image is None:
			return

		assert_path(self.save_dir)
		image.resize((512, 512)).save(self.filename)
		self.control_img.resize((512, 512)).save(f"{self.save_dir}/{self.control} condition.png")
		comparison = image_grid([
		  		self.reference.resize((512, 512)),
					self.control_img.resize((512, 512)),
					image.resize((512, 512))
		   ], 1, 3)
		comparison.save(f"{self.save_dir}/{self.control} control result - seed {self.generate_param['seed']}.png")

		if save_attn:
			save_attention_maps(
			self.pipe.unet.attn_maps,
			self.pipe.tokenizer,
			base_dir=f"{os.getcwd()}/log/attn_maps/{self.modelType.name}/{self.generate_param['prompt']}",
			prompts=[self.generate_param["prompt"]],
			options={
				"prefix": "",
				"return_dict": False,
				"ignore_special_tkns": self.generate_param["ignore_special_tkns"],
				"enabled_editing_prompts": 0
			})

		print(f"Saved results for {self.filename}")

import argparse
import os
from test.inference import EvalModel
from test.test_set import (subjects, seeds, test)
from test.types import ModelType
from typing import Callable, Optional

from tqdm import tqdm

def sample(
	eval: EvalModel,
	inference: Callable,
	output_dir: str = f"{os.getcwd()}/outputs",
	alpha_mask: str = "1",
	save_attn: bool = False,
	use_attn_bias: bool = True,
	filename_prefix: Optional[str] = None,
):
	eval.set_output_dir(output_dir)

	for seed in tqdm(seeds, desc="For all seeds"):
					for situation in test:
						for subject in subjects:
							prompt = situation['prompt'].format(subject=subject)
							mask_prompt = situation['mask_prompt']
							focus_tokens = situation['focus_tokens']
							reference = f"{os.getcwd()}/assets/test/{situation['ref']}"

							output = inference(prompt, reference, ref_subj=situation["ref_subj"], prmpt_subj=subject,
									seed=seed, alpha_mask=alpha_mask, mask_prompt=mask_prompt, focus_tokens=focus_tokens,
									save_attn=save_attn, use_attn_bias=use_attn_bias, filename_prefix=filename_prefix
							)
							eval.postprocess(output, save_attn=save_attn)


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default="0")
	parser.add_argument('--control', type=str, required=True, help="depth | pose | canny | ...")
	parser.add_argument('--model', type=str, default="SemanticControl", help="ControlNet | SemanticControl")
	parser.add_argument('--alpha_mask', nargs="*", type=float, default=[1], help="Mask applied on inferred alpha. [1, 0, 0, 0] means only upper left is used with 1.")
	parser.add_argument('--save_attn', action='store_true', default=False)

	args = parser.parse_args()

	args.modelType = ModelType.str2enum(args.model)

	return args

def main():
	args = parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

	eval = EvalModel(args.control)
	inference = eval.get_inference_func(args.modelType)

	sample(
		eval=eval,
		inference=inference,
		alpha_mask=args.alpha_mask,
		save_attn=args.save_attn
	)

if __name__ == "__main__":
    main()

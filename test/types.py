from enum import Enum
from typing import List, TypedDict


class ModelType(Enum):
	ControlNet = 1
	SemanticControl = 2

	@classmethod
	def str2enum(cls, model: str):
		assert model in cls._member_names_, f"The model type should be one of {cls._member_names_}, but you gave {model}"

		if model == "ControlNet":
			modelType = ModelType.ControlNet
		elif model == "SemanticControl":
			modelType = ModelType.SemanticControl
		return modelType

class SemanticControlArgs(TypedDict):
	ref: str
	ref_subj: str
	prompt: str
	mask_prompt: str
	focus_tokens: str


from typing import Dict, List

from .types import SemanticControlArgs

seeds = [42]

subjects = ["a dog plushie"]

test: List[SemanticControlArgs] = [
        {
            "ref": "a man is holding the guitar.jpg",
            "ref_subj": "a man",
            "prompt": "{subject} is holding the guitar",
            "mask_prompt": "A man is holding the guitar",
            "focus_tokens": "holding guitar",
        },
        {
            "ref": "a man is holding a sword.jpg",
            "ref_subj": "a man",
            "prompt": "{subject} is holding a sword",
            "mask_prompt": "a man is holding a sword",
            "focus_tokens": "holding sword"
        },
        {
            "ref": "a woman is riding a bike.jpg",
            "ref_subj": "a woman",
            "prompt": "{subject} is riding a bike",
            "mask_prompt": "a woman is riding a bike",
            "focus_tokens": "riding bike"
        }
    ]

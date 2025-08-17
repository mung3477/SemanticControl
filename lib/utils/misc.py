from typing import Any, Dict, Optional


def push_key_value(obj: Optional[Dict], key: str, value: Any):
    if obj is None:
        obj = dict()

    obj[key] = value

    return obj

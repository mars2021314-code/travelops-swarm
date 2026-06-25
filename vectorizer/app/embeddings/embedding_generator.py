import hashlib
import math
from typing import Union, List

VECTOR_SIZE = 1536


def _generate_one(content: str) -> List[float]:
    text = content or ""
    vector = [0.0] * VECTOR_SIZE
    tokens = text.lower().split()
    if not tokens:
        tokens = [text]

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % VECTOR_SIZE
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def generate_embedding(content: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    if isinstance(content, str):
        return _generate_one(content)
    elif isinstance(content, list):
        return [_generate_one(item) for item in content]
    else:
        raise ValueError("Content must be either a string or a list of strings")

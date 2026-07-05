import numpy as np
from typing import List, Union

def cosine_similarity(emb1: Union[List[float], np.ndarray], emb2: Union[List[float], np.ndarray]) -> float:
    """
    Computes the cosine similarity between two embedding vectors.
    Returns a float similarity in [-1.0, 1.0].
    """
    a = np.asarray(emb1, dtype=np.float32)
    b = np.asarray(emb2, dtype=np.float32)
    
    # Flatten if 2D
    if len(a.shape) > 1:
        a = a.flatten()
    if len(b.shape) > 1:
        b = b.flatten()
        
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return float(dot_product / (norm_a * norm_b))

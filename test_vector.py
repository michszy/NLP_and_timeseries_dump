from vector import scalar_product, vector_norm, cosinus_similarity
import numpy as np

def test_scallar_product():
    v1, w1 = np.array([1, 1]), np.array([-1, 1])
    assert scalar_product(v1,w1) == 0
    v = np.array([1, 1])
    assert vector_norm(v) == 1
    v1, w1 = np.array([1, 1]), np.array([-1, 1])
    assert cosinus_similarity(v1,w1) == 0
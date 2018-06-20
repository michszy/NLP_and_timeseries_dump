"""Exo 2 Coder une fonction qui calculer le produit scalaire de deux vecteur"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import math


def scalar_product(v, w):
    r = v * w
    r = sum(r)
    print(r)
    return  r





v, w = np.array([1,1,1,1,1]), np.array([2,2,2,2,2])
scalar_product(v,w )
origin = [0], [0]
v1, w1 = np.array([1,1]), np.array([-1,1])
v2, w2 = np.array([1,1]), np.array([2,2])
#scalar_product(v1,w1 )
#scalar_product(v2,w2 )



#Norme d'un vecteur

# sqrt ( x² + y²)

def vector_norm(v):
    x =  v[0]
    y = v[1]
    r = math.sqrt(pow(x,2) + pow(y,2))
    print(r)
    return r

#vector_norm(v1)

def cosinus_similarity(a,b):
    r = (scalar_product(a, b)) / (vector_norm(a) * vector_norm(b))
    print (r)
    return r
"""
reduire de l'overfitting
reduire la complexité du model
faire de la regularisation
avoir plus de donné"""
cosinus_similarity(v1,v2)

plt.quiver(origin, v1)
plt.show()
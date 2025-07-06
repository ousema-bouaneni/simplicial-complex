import numpy as np
from math import isclose
from ..src.project import distance
from itertools import combinations

def compare_to_expected_result(circumcenter, radius, expected_circumcenter, expected_radius, abs_tol=1e-06):
    assert(isclose(radius, expected_radius,  abs_tol=abs_tol))
    assert(np.allclose(circumcenter, expected_circumcenter))

def random_dimension(max_dim, min_dim=1):
    return np.random.randint(min_dim, max_dim)

def random_point_in_cube(center, bound, dim):
    return center + np.random.uniform(-bound, bound, dim)

def random_point_on_sphere(center, radius, dim):
    # Marsaglia (1972)
    x = np.random.normal(0,1,dim)
    return center+radius*x/distance(x,0)

def random_point_in_ball(center, radius, dim):
    # Voelker, Gossman and Stewart (2017)
    x = random_point_on_sphere(0,1,dim+2)
    return center+radius*x[:-2]

def random_point_in_space(dim):
    return np.random.normal(np.zeros(dim), 1)

def diameter(set):
    res = 0
    for pair in combinations(set, 2):
        a, b = tuple(pair)
        a = np.array(a)
        b = np.array(b)
        dist = distance(a,b)
        if dist > res:
            res = dist
    return res
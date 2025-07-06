import pytest
import numpy as np
from math import isclose
from ..src.project import cech_complex, distance
from .helper import random_dimension, random_point_in_cube, diameter, compare_to_expected_result

def test_one_point(max_dim=5, space_bound=1):
    dim = random_dimension(max_dim)
    expected_point = random_point_in_cube(np.zeros(dim), space_bound, dim)
    res = cech_complex([expected_point], dim, 1)
    assert(len(res) == 1)
    simplex, filtration_value = res[0]
    point = list(simplex)[0]
    compare_to_expected_result(point, filtration_value, expected_point, 0)

def test_two_points(max_dim=5, space_bound=1):
    dim = random_dimension(max_dim)
    a = random_point_in_cube(np.zeros(dim), space_bound, dim)
    b = random_point_in_cube(np.zeros(dim), space_bound, dim)
    res = cech_complex([a,b], dim, 2)
    assert(len(res) == 3)
    for simplex, value in res:
        if len(simplex) == 1:
            assert(isclose(value, 0, abs_tol=1e-7))
        elif len(simplex) == 2:
            assert(isclose(value, distance(a,b)/2))

def test_aligned_points_no_limit(max_dim=5, space_bound=1, number_of_points=10):
    dim = random_dimension(max_dim, 3)
    points = [np.pad([i], (0, dim-1), 'constant', constant_values=(4, 6)) for i in range(number_of_points)]
    complex = cech_complex(points, dim, number_of_points)
    assert(len(complex) == pow(2, number_of_points)-1)
    for simplex, value in complex:
        assert(isclose(value, diameter(simplex)/2, abs_tol=1e-6))

def test_aligned_points_with_limit(max_dim=5, space_bound=1, number_of_points=10, limit = 5):
    dim = random_dimension(max_dim, 3)
    points = [np.pad([i], (0, dim-1), 'constant', constant_values=(4, 6)) for i in range(number_of_points)]
    complex = cech_complex(points, dim, number_of_points)
    for simplex, value in complex:
        assert(isclose(value, diameter(simplex)/2, abs_tol=1e-6))
        assert(value < limit)
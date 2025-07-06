#%% Module imports
import pytest
import numpy as np
from math import isclose
from ..src.project import in_ball, lp_type_alpha_complex_filtration_value, alpha_complex
from .helper import random_dimension, random_point_in_cube, random_point_in_space, random_point_on_sphere, diameter, compare_to_expected_result, distance

@pytest.fixture(params=[lp_type_alpha_complex_filtration_value])
def alpha_complex_filtration(request):
    return request.param


def test_two_points(alpha_complex_filtration):
    a = np.array([0,0])
    b = np.array([1,0])
    points = [a,b]
    dim = 2
    simplex = set()
    simplex.add(tuple(a))
    filtration_value = alpha_complex_filtration(points, simplex, dim)
    assert(isclose(filtration_value, 0))

def test_three_points(alpha_complex_filtration):
    a = np.array([4,0])
    b = np.array([-4,0])
    c  = np.array([0,2])
    points = [a,b,c]
    dim = 2
    simplex = {tuple(p) for p in [a,b]}
    filtration_value = alpha_complex_filtration(points, simplex, dim)
    assert(isclose(filtration_value, 5))

def test_triangle(alpha_complex_filtration):
    dim = 3
    a = np.array([0,5,0])
    b = np.array([3,4,0])
    c = np.array([-3,4,0])
    points = [a,b,c]
    simplex = {tuple(p) for p in points}
    filtration_value = alpha_complex_filtration(points, simplex, dim)
    assert(isclose(filtration_value, 5))

def test_triangle2(alpha_complex_filtration):
    dim = 3
    a = np.array([0,5,0])
    b = np.array([3,4,0])
    c = np.array([-3,4,0])
    d = np.array([0,0,4])
    points = [a,b,c]
    simplex = {tuple(p) for p in points}
    points.append(d)
    filtration_value = alpha_complex_filtration(points, simplex, dim)
    assert(filtration_value > 5)

def test_triangle_not_in_simplex(alpha_complex_filtration):
    dim = 3
    a = np.array([0,5,0])
    b = np.array([3,4,0])
    c = np.array([-3,4,0])
    d = np.array([0,0,4])
    e = np.array([0,0,-4])
    points = [a,b,c,d,e]
    simplex = {tuple(p) for p in [a,b,c]}
    filtration_value = alpha_complex_filtration(points, simplex, dim)
    assert(filtration_value == np.inf)

def test_segment_not_in_complex(alpha_complex_filtration):
    a = np.array([4,0])
    b = np.array([-4,0])
    c  = np.array([0,2])
    d = np.array([0,-2])
    points = [a,b,c,d]
    dim = 2
    simplex = {tuple(p) for p in [a,b]}
    filtration_value = alpha_complex_filtration(points, simplex, dim)
    assert(filtration_value == np.inf)

def test_cube_minus_ball_random(alpha_complex_filtration, number_of_points=1000, number_of_tests=10, max_dimension=100):
    for _ in range(number_of_tests):
        dim = random_dimension(max_dim=max_dimension)
        center = random_point_in_space(dim)
        points = [random_point_on_sphere(center, 1, dim) for i in range(dim+1)]
        simplex = {tuple(p) for p in points}
        while len(points) < number_of_points:
            point = random_point_in_cube(center=center, bound=2, dim=dim)
            if not in_ball(point, (center, 1), epsilon=-1e-14):
                points.append(point)
        filtration_value = alpha_complex_filtration(points, simplex, dim)
        assert(isclose(filtration_value, 1))

def test_one_point(alpha_complex_filtration,max_dim=5, space_bound=1):
    dim = random_dimension(max_dim)
    expected_point = random_point_in_cube(np.zeros(dim), space_bound, dim)
    res = alpha_complex([expected_point], dim, 1, np.inf, alpha_complex_filtration)
    assert(len(res) == 1)
    simplex, filtration_value = res[0]
    point = list(simplex)[0]
    compare_to_expected_result(point, filtration_value, expected_point, 0)

def test_two_points2(alpha_complex_filtration,max_dim=5, space_bound=1):
    dim = random_dimension(max_dim)
    a = random_point_in_cube(np.zeros(dim), space_bound, dim)
    b = random_point_in_cube(np.zeros(dim), space_bound, dim)
    res = alpha_complex([a,b], dim, 2, np.inf, alpha_complex_filtration)
    assert(len(res) == 3)
    for simplex, value in res:
        if len(simplex) == 1:
            assert(isclose(value, 0, abs_tol=1e-7))
        elif len(simplex) == 2:
            assert(isclose(value, distance(a,b)/2))

def test_square(alpha_complex_filtration):
    a = np.array([1,0])
    b = np.array([0,1])
    c = np.array([0,-1])
    d = np.array([-1,0])
    points = [a,b,c,d]
    complex = alpha_complex(points, 2, 4,2,alpha_complex_filtration)
    assert(len(complex) == pow(2,4)-1)
    for simplex, value in complex:
        assert(value == diameter(simplex)/2)
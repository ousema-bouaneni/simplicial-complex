#%% Module imports
import pytest
import numpy as np
from ..src.project import sphere_from_points, lp_type_meb, welzl_meb, distance
from .helper import compare_to_expected_result, random_dimension, random_point_on_sphere, random_point_in_cube
import sys

sys.setrecursionlimit(1500)

@pytest.fixture(params=[sphere_from_points, lp_type_meb, welzl_meb])
def circumcenter_function(request):
    return request.param

#%% Manual cases for low dimension
def test_no_points(circumcenter_function):
    with pytest.raises(AssertionError):
        circumcenter_function([])

def test_one_point_manual(circumcenter_function):
    a = np.zeros(3)
    circumcenter, radius = circumcenter_function([a])
    compare_to_expected_result(circumcenter, radius, a, 0)

def test_segment_manual(circumcenter_function):
    a = np.array([0,0])
    b = np.array([0,2])
    circumcenter, radius = circumcenter_function([a,b], 2)
    compare_to_expected_result(circumcenter, radius, np.array([0,1]), 1)

def test_right_triangle(circumcenter_function):
    a = np.array([2,0])
    b = np.array([0,2])
    origin = np.array([0,0])
    expected_center = np.array([1,1])
    circumcenter, radius = circumcenter_function([a,b,origin], 2)
    compare_to_expected_result(circumcenter, radius, expected_center, distance(origin, expected_center))

def test_right_triangle2(circumcenter_function):
    points = [np.array([-5,0,0]), np.array([3,-4,0]), np.array([3,4,0])]
    center, radius = circumcenter_function(points)
    compare_to_expected_result(center, radius, np.array([0,0,0]), 5)

#%% Simple randomized tests
def test_one_point_random(circumcenter_function, number_of_tests = 100, max_dim = 100, space_bound = 1):
    for _ in range(number_of_tests):
        dim = random_dimension(max_dim)
        point = random_point_in_cube(np.zeros(dim), space_bound, dim)
        circumcenter, radius = circumcenter_function([point], dim)
        compare_to_expected_result(circumcenter, radius, point, 0)

def test_segment_random(circumcenter_function, number_of_tests = 100, max_dim = 100, space_bound = 1):
    for _ in range(number_of_tests):
        dim = random_dimension(max_dim)
        a = random_point_in_cube(np.zeros(dim), space_bound, dim)
        b = random_point_in_cube(np.zeros(dim), space_bound, dim)
        circumcenter, radius = circumcenter_function([a,b], dim)
        compare_to_expected_result(circumcenter, radius, (a+b)/2, distance(a,b)/2)

def test_unit_sphere_random(circumcenter_function, dimension=4, number_of_points=100):
    if circumcenter_function == sphere_from_points:
        number_of_points = 0
    points = [random_point_on_sphere(np.zeros(dimension),1,dimension) for i in range(max(number_of_points, dimension+1))]
    circumcenter, radius = circumcenter_function(points, dimension)
    compare_to_expected_result(circumcenter, radius, 0, 1)
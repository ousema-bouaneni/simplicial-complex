import pytest
import numpy as np
from math import sqrt
from ..src.project import naive_meb, lp_type_meb, welzl_meb
from .helper import compare_to_expected_result,random_point_in_ball, random_point_in_cube
import sys

sys.setrecursionlimit(1500)

@pytest.fixture(params=[lp_type_meb, welzl_meb])
def meb(request):
    return request.param


def test_isoceles_triangle(meb):
    points = [np.array([-10,0,0]), np.array([10,0,0]), np.array([0,1,0])]
    center, radius = meb(points)
    compare_to_expected_result(center, radius, np.array([0,0,0]), 10)

def test_4points(meb):
    a = np.array([5,0,1])
    b = np.array([-1,-3,4])
    c = np.array([-1,-4,-3])
    d = np.array([-1,4,-3])
    points = [a,b,c,d]
    center, radius = meb(points)
    compare_to_expected_result(center, radius, np.array([0,0,0]), sqrt(26))

def test_unit_ball_random(meb, dim=4, number_of_points=100):
    points = [random_point_in_ball(np.zeros(dim),1,dim) for i in range(number_of_points-2)]
    a = np.zeros(dim)
    a[0] = 1
    b = np.zeros(dim)
    b[0] = -1
    points.append(a)
    points.append(b)
    circumcenter, radius = meb(points, dim)
    compare_to_expected_result(circumcenter, radius, 0, 1)

def test_cube_random(meb, dim=4, number_of_points=100, bound=1):
    points = [random_point_in_cube(np.zeros(dim), bound, dim) for p in range(number_of_points-2)]
    points.append(np.ones(dim))
    points.append(-np.ones(dim))
    c, r = meb(points, dim)
    compare_to_expected_result(c, r, np.zeros(dim), sqrt(dim))

def test_arbitrary_pointset(meb):
    number_of_points=5
    bound = 10
    dim = 3
    points = [random_point_in_cube(np.zeros(dim), bound, dim) for p in range(number_of_points)]
    c,r = meb(points, dim)
    expected_c, expected_r = naive_meb(points, dim)
    compare_to_expected_result(c, r, expected_c, expected_r)
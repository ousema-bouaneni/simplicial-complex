#%% Importing modules

import numpy as np
from random import shuffle
import itertools

#%% Geometry helper functions
def norm2(x):
    return np.sum(x**2)

def distance(x,y):
    return np.sqrt(norm2(x-y))

def in_ball(point, ball, epsilon = 1e-10):
    # epsilon needs to be bigger than 1e-15 in magnitude to avoid floating point errors
    center,radius = ball
    return distance(center,point) < radius + epsilon


#%% Sphere construction

def sphere_from_points(points, dim=3):
    """
    Finds the circumcenter of a set of points

    Input : points is a non-empty list of at most dim+1 ndarrays each of which encodes a point in dim 
    
    Output : circumcenter, radius

    Complexity : O(dim^3)
    """
    assert(type(dim) == int)
    assert(dim > 0)
    assert(len(points) > 0)
    if len(points) > dim + 1:
        points = points[:dim+1]
    points_as_array = np.array(points)

    # See report to find a derivation of A and B

    A = np.block([[2 * points_as_array @ points_as_array.T, np.ones((len(points),1))],[np.ones((1,len(points))),0]])

    #      _________________________
    #     |                     |   |
    #     | 2 Gram(p1, ..., pn) | 1 |
    # A = |_____________________|___|
    #     |__________1__________|_0_|
    #
    # where p1, ..., pn are the elements of the list points

    B = np.sum(points_as_array**2,axis=1)
    B = np.append(B,1)

    # B = [ ||p1||² , ..., ||pn||² , 1]

    solution = np.linalg.solve(A,B)

    # solution = [c1, ..., cn, u]
    # circumcenter = c1 * p1 + ... + cn * pn and u = radius² - ||circumcenter||²

    circumcenter = np.dot(solution[:-1], points_as_array)

    # We use an absolute value here to avoid having an exception when the radius is 0
    # (in which case the value could be around -1e-12 due to floating point arithmetic)

    radius = np.sqrt(np.abs(solution[-1] + norm2(circumcenter)))
    
    return circumcenter, radius


sphere_from_points.number_of_points = 0

#%% MEB naive approach

def naive_meb(points,dim=3):
    """
    Finds the minimal enclosing ball (MEB) of a non-empty set of points by iterating over all subsets of at most dim+1 elements

    Input : non-empty set of points of length n and dimension d

    Output : center and radius of the MEB

    Complexity : O(d^3 * n^{d+1})
    """
    assert(len(points) > 0)
    center = np.zeros(dim)
    radius = np.inf
    for size in range(1,dim + 2):
        for subset in itertools.combinations(points,size):
            new_sphere = sphere_from_points(list(subset), dim)
            new_center, new_radius = new_sphere
            if new_radius < radius and all([in_ball(point, new_sphere) for point in points]):
                radius = new_radius
                center = new_center
    return center, radius

#%% MEB with LP-type solver

def lp_type_solver(constraint_set, violation_test, current_basis, combinatorial_dimension):
    """
    Implementation of Seidel's algorithm to solve LP-type problems

    Input :
            - a list of length n representing constraints

            - violation_test allowing to test if basis changes after addition of a constraint

            - initially empty list representing a basis of the already done constraints

            - combinatorial_dimension d of the LP-type problem

    Output : an arbitrary basis of the constraint set

    Complexity : O(d! * n) expected calls to violation_test
    """
    if len(current_basis) >= combinatorial_dimension:
        return current_basis
    done_constraints = []
    updated_basis = current_basis
    for constraint in np.random.permutation(constraint_set):
        if violation_test(updated_basis, constraint):
            updated_basis = lp_type_solver(done_constraints, violation_test, [*current_basis, constraint], combinatorial_dimension)
        done_constraints.append(constraint)
    return updated_basis

def lp_type_meb(points, dim=3):
    """
    Finds the minimal enclosing ball (MEB) of a non-empty set of points using an LP-type solver

    Input : non-empty set of points of length n and dimension d

    Output : center and radius of the MEB

    Complexity : O(d^4 * d! * n)
    """
    def violation_test(boundary_points, new_point):
            return len(boundary_points) == 0 or not in_ball(new_point, sphere_from_points(boundary_points, dim))
    return sphere_from_points(lp_type_solver(points,violation_test,[], dim+1),dim)

#%%  MEB with Welzl's algorithm

def sphere_from_points_noerror(points, dim=3):
    """
    Variant of the previous circumcenter function which doesn't throw an error on empty pointsets for use with Welzl's recursive algorithm
    """
    if len(points) == 0:
        return np.inf*np.ones(dim), 0
    else:
        return sphere_from_points(points, dim)

def welzl_aux(points,boundary_points,dim=3):
    """
    Auxiliary function used for Welzl's algorithm with accumulator for known boundary points
    """
    if len(points)==0 or len(boundary_points)==dim+1:
        return sphere_from_points_noerror(boundary_points,dim)
    point = points.pop()
    solution_candidate = welzl_aux(points[:],boundary_points[:],dim)
    if in_ball(point,solution_candidate):
        return solution_candidate
    boundary_points.append(point)
    return welzl_aux(points[:],boundary_points[:],dim)

def welzl_meb(points,dim=3):
    """
    Finds the minimal enclosing ball (MEB) of a non-empty set of points using recursion

    Input : non-empty set of points of length n and dimension d

    Output : center and radius of the MEB

    Complexity : O(d! * n)
    """
    assert(len(points) > 0)
    shuffle(points)
    return welzl_aux(points[:],[],dim)

#%% Cech and alpha complexes

def complex(points, space_dim, max_simplex_dim, filtration_limit, complex_filtration_value):
    """
    Finds all simplexes in a simplicial complex that have dimension less than
     or equal to max_simplex_dim and filtration value less than filtration_limit

    Input :
        - non-empty list of points of length n
        - the dimension d
        - the maximum dimension for simplexes (k in the problem statement)
        - the limit on filtration values
        - a filtration_value calculator

    Output : list of (simplex, filtration_value) couples

    Complexity : O(number_of_simplexes_in_the_complex * n) calls to complex_filtration_value
    """
    if len(points) == 0:
        return []
    points_as_tuples = [tuple(point) for point in points]
    res = []
    current_simplex_set = {frozenset()}
    for i in range(1, max_simplex_dim+1):
        new_simplex_set = set()
        for simplex in current_simplex_set:
            for point in points_as_tuples:
                if point not in simplex :
                    new_simplex = set(simplex)
                    new_simplex.add(point)
                    new_simplex = frozenset(new_simplex)
                    if new_simplex in new_simplex_set:
                        continue
                    filtration_value = complex_filtration_value(points, new_simplex, space_dim)
                    if filtration_value < filtration_limit:
                        new_simplex_set.add(new_simplex)
                        res.append((new_simplex,filtration_value))
        current_simplex_set = new_simplex_set
    return res

def cech_complex(points, space_dim, max_simplex_dim, filtration_limit=np.inf, meb=lp_type_meb):
    """
    Finds all simplexes in the cech complex

    Input :

            - non-empty list of points of length n
            - the dimension d
            - the maximum dimension for simplexes (k in the problem statement)
            - an optional limit on filtration values (infinity by default)
            - an MEB problem solver (lp_type_meb by default)

    Output : list of (simplex, filtration_value) couples

    Complexity : O(n^{k+2}) calls to meb, so O(d^4 * d! * n^{k+3}) by default
    """
    def cech_filtration_value(points, simplex, space_dim):
        _,value=meb(list([np.array(point) for point in simplex]), space_dim)
        return value
    return complex(points, space_dim, max_simplex_dim, filtration_limit, cech_filtration_value)

def lp_type_alpha_complex_filtration_value(points, simplex, dim=3):
    """
    Finds filtration value of simplex in the alpha complex using an LP-type solver

    Input : non-empty list of points of length n, a simplex, and the dimension d

    Output : infinity if the simplex isn't in the alpha complex, its filtration value otherwise

    Complexity : O(d^4 * d! * n)
    """

    def violation_test(boundary_points, new_point):
        return len(boundary_points) == 0 or in_ball(new_point, sphere_from_points(boundary_points, dim), epsilon = -1e-10)
    
    outside_points = [x for x in points if tuple(x) not in simplex]
    ball = sphere_from_points(lp_type_solver(outside_points, violation_test, [np.array(point) for point in simplex], dim+1), dim)
    _, filtration_value = ball
    for point in outside_points:
        if in_ball(point, ball, epsilon=-1e-10):
            filtration_value = np.inf
    return filtration_value

def alpha_complex(points, space_dim, max_simplex_dim, filtration_limit=np.inf, alpha_complex_filtration_value=lp_type_alpha_complex_filtration_value):
    """
    Finds all simplexes in the alpha complex

    Input :
        - non-empty list of points of length n
        - the dimension d
        - the maximum dimension for simplexes (k in the problem statement)
        - an optional limit on filtration values (infinity by default)
        - a filtration_value calculator (lp_type_alpha_complex_filtration_value by default)

    Output : list of (simplex, filtration_value) couples

    Complexity : O(size_of_alpha_complex * n) calls to alpha_complex_filtration_value, so O(size_of_alpha_complex * d^4 * d! * n^2) by default
    """
    return complex(points, space_dim, max_simplex_dim, filtration_limit, alpha_complex_filtration_value)
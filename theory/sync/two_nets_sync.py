import numpy as np
from itertools import product as iproduct
from scipy.optimize import fsolve
from numdifftools import Jacobian
from numpy.linalg import det

import sync.single_net_sync as sns

"""
This file contains methods for calculating R_1 and R_2 in two networks
"""
verbose = False

def interaction_factory(interaction_type,q = 0):
    if interaction_type == "interdependent":
        return lambda x : q + (1 - q ) * x
    elif interaction_type == "competitive":
        return lambda x: q + (1 - q) * (1 - x)

def combined_R1R2(intfunc, lam, frac, interaction_type="interdependent",q=0):
    if interaction_type == "interdependent" or interaction_type == "competitive":
        return lambda R1R2: np.array([
            (frac * intfunc(lam[0] * R1R2[0]* interaction_factory(interaction_type,q)(R1R2[1])) + (1 - frac) * intfunc(lam[0] * R1R2[0])) - R1R2[0],
            (frac * intfunc(lam[1] * R1R2[1]* interaction_factory(interaction_type,q)(R1R2[0])) + (1 - frac) * intfunc(lam[1] * R1R2[1])) - R1R2[1]])
    # elif interaction_type == "competitive":
    #     return lambda R1R2: np.array([
    #         (frac * intfunc(lam[0] * R1R2[0] * (1 - R1R2[1])) + (1 - frac) * intfunc(lam[0] * R1R2[0])) - R1R2[0],
    #         (frac * intfunc(lam[1] * R1R2[1] * (1 - R1R2[0])) + (1 - frac) * intfunc(lam[1] * R1R2[1])) - R1R2[1]])
    elif interaction_type == "hybrid":
        return lambda R1R2: np.array([
            (frac * intfunc(lam[0] * R1R2[0] * interaction_factory("interdependent",q)(R1R2[1])) + (1 - frac) * intfunc(lam[0] * R1R2[0])) - R1R2[0],
            (frac * intfunc(lam[1] * R1R2[1] * interaction_factory("competitive",q)(R1R2[0])) + (1 - frac) * intfunc(lam[1] * R1R2[1])) - R1R2[1]])
    elif interaction_type == "mixed":
        return lambda R1R2: np.array([
            (frac * intfunc(lam[0] * R1R2[0] * interaction_factory("interdependent",q)(R1R2[1])) + (1 - frac) * intfunc(lam[0] * R1R2[0]* interaction_factory("competitive",q)(R1R2[1]))) - R1R2[0],
            (frac * intfunc(lam[1] * R1R2[1] * interaction_factory("interdependent",q)(R1R2[0])) + (1 - frac) * intfunc(lam[1] * R1R2[1]* interaction_factory("competitive",q)(R1R2[0]))) - R1R2[1]])
    elif interaction_type == "halfhalf":
        return lambda R1R2: np.array([
            (frac * 0.5 * (intfunc(lam[0] * R1R2[0] * R1R2[1]) + intfunc(lam[0] * R1R2[0] * (1 - R1R2[1]))) +
             (1 - frac) * intfunc(lam[0] * R1R2[0])) -  R1R2[0],
            (frac * 0.5 * (intfunc(lam[1] * R1R2[1] * R1R2[0]) + intfunc(lam[1] * R1R2[1] * (1 - R1R2[0]))) +
            (1 - frac) * intfunc(lam[1] * R1R2[1])) - R1R2[1]])
    raise ValueError("Unknown interaction type received: " + str(interaction_type))


def combined_R1R2_different_systems(intfunc1,intfunc2,lambeta,frac,interaction_type="interdependent",q=0):
    if interaction_type == "interdependent" or interaction_type == "competitive":
        return lambda R1R2: np.array([
            (frac * intfunc1(lambeta[0] * R1R2[0]* interaction_factory(interaction_type,q)(R1R2[1])) + (1 - frac) * intfunc1(lambeta[0] * R1R2[0])) - R1R2[0],
            (frac * intfunc2(lambeta[1] * R1R2[1]* interaction_factory(interaction_type,q)(R1R2[0])) + (1 - frac) * intfunc2(lambeta[1] * R1R2[1])) - R1R2[1]])
    # elif interaction_type == "competitive":
    #     return lambda R1R2: np.array([
    #         (frac * intfunc(lam[0] * R1R2[0] * (1 - R1R2[1])) + (1 - frac) * intfunc(lam[0] * R1R2[0])) - R1R2[0],
    #         (frac * intfunc(lam[1] * R1R2[1] * (1 - R1R2[0])) + (1 - frac) * intfunc(lam[1] * R1R2[1])) - R1R2[1]])
    elif interaction_type == "hybrid":
        return lambda R1R2: np.array([
            (frac * intfunc1(lambeta[0] * R1R2[0] * interaction_factory("interdependent",q)(R1R2[1])) + (1 - frac) * intfunc1(lambeta[0] * R1R2[0])) - R1R2[0],
            (frac * intfunc2(lambeta[1] * R1R2[1] * interaction_factory("competitive",q)(R1R2[0])) + (1 - frac) * intfunc2(lambeta[1] * R1R2[1])) - R1R2[1]])
    elif interaction_type == "mixed":
        return lambda R1R2: np.array([
            (frac * intfunc1(lambeta[0] * R1R2[0] * interaction_factory("interdependent",q)(R1R2[1])) + (1 - frac) * intfunc1(lambeta[0] * R1R2[0]* interaction_factory("competitive",q)(R1R2[1]))) - R1R2[0],
            (frac * intfunc2(lambeta[1] * R1R2[1] * interaction_factory("interdependent",q)(R1R2[0])) + (1 - frac) * intfunc2(lambeta[1] * R1R2[1]* interaction_factory("competitive",q)(R1R2[0]))) - R1R2[1]])
    elif interaction_type == "halfhalf":
        return lambda R1R2: np.array([
            (frac * 0.5 * (intfunc1(lambeta[0] * R1R2[0] * R1R2[1]) + intfunc1(lambeta[0] * R1R2[0] * (1 - R1R2[1]))) +
             (1 - frac) * intfunc1(lambeta[0] * R1R2[0])) -  R1R2[0],
            (frac * 0.5 * (intfunc2(lambeta[1] * R1R2[1] * R1R2[0]) + intfunc2(lambeta[1] * R1R2[1] * (1 - R1R2[0]))) +
            (1 - frac) * intfunc2(lambeta[1] * R1R2[1])) - R1R2[1]])


def solve_R1R2(intfunc, lam, frac,eps=1e-9,interaction_type="interdependent"):
    to_solve = combined_R1R2(intfunc, lam, frac,interaction_type=interaction_type)
    r1solutions = []
    r2solutions = []
    if interaction_type == "interdependent":
        init_list = [(i,i) for i in np.arange(0,1,0.1)]
    else:
        init_list = [i for i in iproduct(np.arange(0, 1, 0.2),np.arange(0, 1, 0.2))]
    for init in init_list:
        sol, info, ier, mesg = fsolve(to_solve, init, full_output=True, xtol=1.49012e-10, maxfev=1000)
        if ier == 1 and sol[0] >= 0 and sol[1] >= 0:
            if all([abs(sol[0] - i) > eps for i in r1solutions]):
                r1solutions.append(float(sol[0]))
            if all([abs(sol[1] - i) > eps for i in r2solutions]):
                r2solutions.append(float(sol[1]))
    return sorted(r1solutions),sorted(r2solutions)


def solve_R1R2_double_solutions(intfunc, lam, frac,eps=1e-9,interaction_type="interdependent",q=0):
    to_solve = combined_R1R2(intfunc, lam, frac,interaction_type=interaction_type,q=q)
    solutions = [[0,0]]
    stabilities = [get_solution_stability(to_solve,[0,0])]
    #if interaction_type == "interdependent":
    #    init_list = [(i,i) for i in np.arange(0,1,0.1)]
    #else:
    init_list = [i for i in iproduct(np.arange(0, 1, 0.14), np.arange(0, 1, 0.1))]
    for init in init_list:
        sol, info, ier, mesg = fsolve(to_solve, init, full_output=True, xtol=1.49012e-10, maxfev=1000)
        if ier == 1:
            if abs(sol[0]) < eps:
                sol[0] = 0
            if abs(sol[1]) < eps:
                sol[1] = 0
            if sol[0] < 0 or sol[1] < 0 or sol[0] > 1 or sol[1] > 1:
                continue
            if all([abs(sol[0] - i[0]) > eps or abs(sol[1] - i[1]) > eps for i in solutions]):
                solutions.append([float(sol[0]),float(sol[1])])
                stabilities.append(get_solution_stability(to_solve,sol))
    if sum(stabilities) == 0:
        for i,sol in enumerate(solutions):
            stabilities[i] = get_solution_stability(to_solve,sol,maxorder=16)
    return sorted(zip(solutions,stabilities))


def solve_R1R2_different_systems_double_solutions(intfunc1, intfunc2, lambeta, frac,eps=1e-9,interaction_type="interdependent",q=0):
    to_solve = combined_R1R2_different_systems(intfunc1, intfunc2, lambeta, frac,interaction_type=interaction_type,q=q)
    solutions = [[0,0]]
    stabilities = [get_solution_stability(to_solve,[0,0])]
    #if interaction_type == "interdependent":
    #    init_list = [(i,i) for i in np.arange(0,1,0.1)]
    #else:
    init_list = [i for i in iproduct(np.arange(0, 1, 0.14), np.arange(0, 1, 0.1))]
    for init in init_list:
        sol, info, ier, mesg = fsolve(to_solve, init, full_output=True, xtol=1.49012e-10, maxfev=1000)
        if ier == 1:
            if abs(sol[0]) < eps:
                sol[0] = 0
            if abs(sol[1]) < eps:
                sol[1] = 0
            if sol[0] < 0 or sol[1] < 0 or sol[0] > 1 or sol[1] > 1:
                continue
            if all([abs(sol[0] - i[0]) > eps or abs(sol[1] - i[1]) > eps for i in solutions]):
                solutions.append([float(sol[0]),float(sol[1])])
                stabilities.append(get_solution_stability(to_solve,sol))
    if sum(stabilities) == 0:
        for i,sol in enumerate(solutions):
            stabilities[i] = get_solution_stability(to_solve,sol,maxorder=16)
    return sorted(zip(solutions,stabilities))


def get_solution_stability(to_solve,sol,eps=1e-9,maxorder=4):

    if abs(sol[0]) < eps or abs(sol[1]) < eps:
        methods = ["forward"]
    else:
        methods = ["central","backward"]
    outputs=[]
    for method in methods:
        for order in range(2,maxorder,2):
            jac = Jacobian(to_solve, method=method)
            jfp = jac(sol)
            thisdet, tra = [det(jfp), np.trace(jfp)]
            if thisdet < 0 or thisdet > 0 and tra > 0:
                stable = 0
            else:
                stable = 1
            outputs.append(stable)
    if verbose and len(outputs)>1 and not np.prod(outputs):
        print("Warning [%.4f,%.4f] : Disagreement found between central and backward"%tuple(sol))
    return max(outputs)


#these functions assume the intfunc has a parameter for k (ie interp2d from sns)

def combined_R1R2_debug1(func, k1, k2, lam, frac):
    to_solve = lambda R1, R2: [(frac * func(k1, lam * R1 * R2) + (1 - frac) * func(k1, lam * R1)) - R1,
                               (frac * func(k2, lam * R1 * R2) + (1 - frac) * func(k2, lam * R2)) - R2]
    return to_solve

def combined_R1R2_debug2(func, k1, k2, lam, frac):
    return lambda R1R2: [
        (frac * func(k1, lam * np.product(R1R2), grid=False) + (1 - frac) * func(k1, lam * R1R2[0], grid=False)) - R1R2[
            0],
        (frac * func(k2, lam * np.product(R1R2), grid=False) + (1 - frac) * func(k2, lam * R1R2[1], grid=False)) - R1R2[
            1]]


def combined_R1R2_debug3(func, k1, k2, lam, frac, as_array=False):
    R1diff = lambda R1, R2: (frac * func(k1, lam * R1 * R2, grid=False) + (1 - frac) * func(k1, lam * R1,
                                                                                            grid=False)) - R1
    R2diff = lambda R1, R2: (frac * func(k2, lam * R1 * R2, grid=False) + (1 - frac) * func(k2, lam * R2,
                                                                                            grid=False)) - R2
    if as_array:
        return lambda R1R2: np.array([R1diff(R1R2[0], R1R2[1]), R2diff(R1R2[0], R1R2[1])])
    return lambda R1R2: [R1diff(R1R2[0], R1R2[1]), R2diff(R1R2[0], R1R2[1])]


# only implemented for k1 == k2
def plot_stability(func, lam, frac, k1, eps=1e-11):
    import matplotlib.pyplot as plt
    Rvec = np.arange(0, 1, 0.001)
    f_of_r = combined_R1R2(func, k1, k1, lam, frac)
    plt.plot(Rvec, [f_of_r([R_, R_])[0] for R_ in Rvec], label="RR")
    # plt.plot(Rvec,Rvec,label="R")
    plt.axhline(0)

    plt.xlabel("f(R) - R")
    plt.ylabel("R")


def get_R1R2_surface(func, k1, k2, lam, frac, dR=1e-4):
    Rvec = np.arange(0, 1 + dR, dR)
    Nr = len(Rvec)
    Z1 = np.zeros((Nr, Nr))
    Z2 = 0 * Z1
    to_plot = combined_R1R2(func, k1, k2, lam, frac)
    R1R2_pairs = iproduct(Rvec, Rvec)
    R1R2_vals = map(to_plot, R1R2_pairs)
    R1R2_idx = iproduct(range(len(Rvec)), range(len(Rvec)))
    for (i, j), (val1, val2) in zip(R1R2_idx, R1R2_vals):
        Z1[i, j] = val1
        Z2[i, j] = val2
    return Rvec, Z1, Z2


def geometric_R1R2_solutions(Rvec, Z1, Z2, steps=1000, eps=1e-9):
    import matplotlib.pyplot as plt
    cs1 = plt.contour(Rvec, Rvec, Z1, levels=[0])
    cs2 = plt.contour(Rvec, Rvec, Z2, levels=[0])
    p1 = cs1.collections[0].get_paths()[0]
    p2 = cs2.collections[0].get_paths()[0]
    from scipy import interp
    interp_x = np.linspace(0, 1, steps)

    ip1 = interp(interp_x, p1.vertices[:, 0], p1.vertices[:, 1])
    ip2 = interp(interp_x, p2.vertices[:, 0], p2.vertices[:, 1])
    from shapely.geometry import LineString
    ls1 = LineString(zip(interp_x, ip1))
    ls2 = LineString(zip(interp_x, ip2))
    points = ls1.intersection(ls2)
    if points.geom_type == "Point":
        return [[0 if x < eps else x for x in [points.y, points.x]]]
    solutions = []
    for point in points:
        if point.geom_type != "Point":
            print("Non-Point solution found, ignoring. Type found: %s" % point.geom_type)
            continue
        solutions.append([0 if x < eps else x for x in [point.y, point.x]])  # switching order! axis of contour are T
    return solutions
    # np.abs(ip1[:,0]) - np.abs(ip2[:,0])


def algebraic_R1R2_solutions(r1r2_func, guesses, eps=1e-9):
    from scipy.optimize import newton_krylov as nk
    solutions = []
    for guess in guesses:
        try:
            nk_sol = nk(r1r2_func, guess, x_tol=eps)
            solutions.append([0 if x < eps else x for x in nk_sol])
        except ValueError:
            print("Unable to calculate algebraic solution, using geometric guess")
            solutions.append(guess)
    return solutions


def jacobian(r1r2_func):
    import numdifftools as ndt
    return ndt.Jacobian(r1r2_func)


def check_solution_stability(J, solution, eps=1e-9):
    solution = [eps if i < eps else i for i in solution]
    jmat = J.jacobian(solution)
    from scipy.linalg import det
    jdet = det(jmat)
    jtr = jmat[0, 0] + jmat[1, 1]
    if jtr < 0 and jdet > 0:
        return True
    else:
        return False


def check_solution_list_stability(J, solution_list):
    return [check_solution_stability(J, sol) for sol in solution_list]


def get_curves(intfunc, lamvec, k1, k2, frac, dR=1e-3, eps=1e-9, debug=False):
    solutions = []
    for lam in lamvec:
        if debug:
            print(lam, end=" ")
        r1r2 = combined_R1R2(intfunc, k1, k2, lam, frac, True)
        J = jacobian(r1r2)
        Rvec, Z1, Z2 = get_R1R2_surface(intfunc, k1, k2, lam, frac, dR)
        gsols = geometric_R1R2_solutions(Rvec, Z1, Z2, 1 / dR, eps)
        asols = algebraic_R1R2_solutions(r1r2, gsols, eps)
        stability = check_solution_list_stability(J, asols)
        solutions.append({"solutions": asols, "stable": stability})
    return solutions


def plot_curves(lamvec, solutions, color="red"):
    from matplotlib import pyplot as plt
    for idx in range(len(lamvec)):
        lam = lamvec[idx]
        sols = solutions[idx]
        for sidx, [R1, R2] in enumerate(sols["solutions"]):
            alpha = 1 if sols["stable"][sidx] else 0.6
            mew = 1 if sols["stable"][sidx] else 0.1
            plt.plot(lam, R1, 'o', color=color, alpha=alpha, mew=mew)
            plt.plot(lam, R2, 'v', color=color, alpha=alpha, mew=mew)


def plot_R1R2_surface(Rvec, Z1, Z2):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(121)
    plt.pcolormesh(Rvec, Rvec, Z1)
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(Rvec, Rvec, Z2)
    plt.colorbar()

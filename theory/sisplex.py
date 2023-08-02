from __future__ import division, print_function
import numpy as np
from scipy.integrate import quad, dblquad, IntegrationWarning
from scipy.special import hyp2f1,zetac, factorial
from scipy.interpolate import interp2d, RectBivariateSpline, interp1d
from scipy.optimize import fsolve, brentq
from itertools import product as iproduct
from numdifftools import Jacobian
from numpy.linalg import det
import json
kmin = 0
kmax = 1000

def gaussweight(xy):
    x,y = xy
    A = 7/(np.pi*2)
    sx = .3
    sy = .3
    x0 = 0.5
    y0 = 0.5
    return A * np.exp( -(x-x0)**2/(2*sx**2) - (y-y0)**2/(2*sy**2))

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

#
# def solve_R1R2(intfunc, lam, frac,eps=1e-9,interaction_type="interdependent"):
#     to_solve = combined_R1R2(intfunc, lam, frac,interaction_type=interaction_type)
#     r1solutions = []
#     r2solutions = []
#     if interaction_type == "interdependent":
#         init_list = [(i,i) for i in np.arange(0,1,0.1)]
#     else:
#         init_list = [i for i in iproduct(np.arange(0, 1, 0.2),np.arange(0, 1, 0.2))]
#     for init in init_list:
#         sol, info, ier, mesg = fsolve(to_solve, init, full_output=True, xtol=1.49012e-10, maxfev=1000)
#         if ier == 1 and sol[0] >= 0 and sol[1] >= 0:
#             if all([abs(sol[0] - i) > eps for i in r1solutions]):
#                 r1solutions.append(float(sol[0]))
#             if all([abs(sol[1] - i) > eps for i in r2solutions]):
#                 r2solutions.append(float(sol[1]))
#     return sorted(r1solutions),sorted(r2solutions)


def solve_R1R2_double_solutions(intfunc, lam, frac,eps=1e-9,interaction_type="interdependent",q=0):
    to_solve = combined_R1R2(intfunc, lam, frac,interaction_type=interaction_type,q=q)
    solutions = [[0,0]]
    stabilities = [get_solution_stability(to_solve,[0,0])]
    #if interaction_type == "interdependent":
    #    init_list = [(i,i) for i in np.arange(0,1,0.1)]
    #else:
    init_list = [i for i in iproduct(np.arange(0, 1, 0.035), np.arange(0, 1, 0.041))]
    for init in init_list:
        if gaussweight(init) > np.random.rand():
            continue
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
    return sorted(zip(solutions,stabilities))


def get_solution_stability(to_solve,sol,eps=1e-9):

    if abs(sol[0]) < eps or abs(sol[1]) < eps:
        method = "forward"
    else:
        method = "central"
    jac = Jacobian(to_solve, method=method)
    jfp = jac(sol)
    thisdet, tra = [det(jfp), np.trace(jfp)]
    if thisdet < 0 or thisdet > 0 and tra > 0:
        return 0
    else:
        return 1


def make_Pk_ER(kbar, crossover_k=12):# this is k *Pk!!!
    """
    This function returns a k*P(k) for ER nets.
     It uses the full factorial function for low values of k and the Stirling approx for higher values.
    :param kbar:
    :param crossover_k:
    :return:
    """
    def f(k):
        if k < crossover_k:
            return kbar ** (k) * np.exp(-kbar) * k / np.float128(factorial(k))
        else:
            lny = k * np.log(kbar) + (0.5 - k) * np.log(k) + k - kbar - 0.91893853320467267 - 1 / (12.0 * k) + 1 / (
            360 * k ** 3) - 1 / (1260 * k ** 5) + 1 / (1680 * k ** 7)
            return np.exp(lny)

    return f


def make_integrand(kbar, alpha,crossover_k=12):
    kPk = make_Pk_ER(kbar,crossover_k)
    return lambda k: k*kPk(k) / (1 + k*alpha)


def make_integrand_sf(gamma, alpha):
    return lambda k : k**(2-gamma) / (1 + k*alpha)


def integrate(kbar,alpha,dist="er",supplied_kmax=None):
    if dist=="er":
        val, err = quad(make_integrand(kbar, alpha), 1, kmax, epsabs=1.49e-9, epsrel=1.49e-9, limit=500)
    elif dist=="sf":
        gamma = kbar
        val, err = quad(make_integrand_sf(gamma, alpha), 1, kmax, epsabs=1.49e-9, epsrel=1.49e-9, limit=500)
        kbar = get_sf_kbar(gamma)
    return val * alpha / kbar


def get_sf_integral(gamma,forcekbar=None):
    oneOverKbar = 1/forcekbar if forcekbar else 1/get_sf_kbar(gamma)
    if gamma == 4:
        f = lambda alpha:  oneOverKbar * alpha * ( 1 - alpha * np.log(1 + (1 / alpha))) if alpha > 0 else 0
    elif gamma == 3:
        f = lambda alpha: oneOverKbar * alpha * np.log(1 + ( 1 / alpha )) if alpha > 0 else 0
    else:
        f= lambda alpha : oneOverKbar * hyp2f1(1,gamma-2,gamma-1,-1/alpha) / ( (gamma -2)) if alpha > 0 else 0
    return np.vectorize(f, otypes=[np.float64])


def get_ba_integral(m):
    f= lambda alpha: (1/2*m) * alpha * np.log(1 + 1 / (alpha * m)) if alpha > 0 else 0
    return np.vectorize(f, otypes=[np.float64])

def get_rr_integral(kbar):
    f = lambda alpha : alpha*kbar/(1 + alpha*kbar) if alpha > 0 else 0
    return np.vectorize(f, otypes=[np.float64])


def get_sf_kbar(gamma):
    """
    assuming integration from 1 to inf in continuum approx
    :param gamma:
    :return:
    """
    if gamma <= 2:
        raise ValueError
    return (zetac(gamma-1) + 1) / ( zetac(gamma) + 1)


def get_interpolated_integral(kbar, mode="r", fname="sis_interpolation_values.json",gamma=3,dist="er",supplied_kmax=None):
    global W
    if mode == "r":
        d = json.load(open(fname))
        alpha = np.array(d['alpha'])
        inta = d["integral"]
        if "dist" in d:
            if d["dist"] == "sf":
                assert(abs(gamma - d["gamma"]) < 1e-6)
            elif d["dist"] == "er":
                assert (abs(kbar - d["kbar"]) < 1e-6)
        else:
            assert (abs(kbar - d["kbar"]) < 1e-6)
        print(" _|_ loaded  %s -|- "%fname, end='')
    else:
        alpha = np.arange(0, 3, 0.00001)
        inta = []
        for a in alpha:
            try:
                val = integrate(kbar, a,dist,supplied_kmax)
            except OverflowError:
                print("Encountered overflow error for alpha = %.8f , using 0" % a)
                val = 0
            except IntegrationWarning as warn:
                print("Check value obtained for alpha = %.6f, received message : %s " %(a,warn.message))
            inta.append(val)
        # inta = [0] + [get_int(kbar, a) for a in alpha[1:]]
        # inta = [0] + [full_integrate(kbar, a) for a in alpha[1:]]
        print(" _|_ calced -|- ", end='')
        if mode == "w":
            d = {"alpha": [i for i in alpha], "integral": inta, "kbar": kbar, "dist": dist,}
            if dist == "sf":
                d["gamma"] = gamma
            json.dump(d, open(fname, "w"))
            print(" _|_ written -|- ", end='')
    print("")
    try:
        return interp1d(alpha, inta, bounds_error=False, fill_value="extrapolate", kind="linear")
    except:
        print("Warning: edges not extrapolated")
        return interp1d(alpha, inta, bounds_error=False, fill_value=0, kind="cubic")

def find_fc(kbar,beta,integral_fname=None,eps=1e-7):
    kPk = make_Pk_ER(kbar)
    if integral_fname:
        intfunc =  get_interpolated_integral(kbar,mode="r",fname=integral_fname)
    else:
        intfunc =  get_interpolated_integral(kbar,mode="x")
    def to_solve(theta_f):
        theta,frac = theta_f
        tangent_integrand = lambda k : (2*frac*theta / (1 + beta*k*theta**2)**2 + (1-frac) / (1 + beta*k*theta)**2) * k*kPk(k)
        val, err = quad(tangent_integrand, 1, kmax, epsabs=1.49e-9, epsrel=1.49e-9, limit=500)
        tangent_condition = (beta / kbar) * val - 1
        xsect_condition = frac * intfunc(beta * theta**2) + (1 - frac) * intfunc(beta*theta) - theta
        return np.array([tangent_condition,xsect_condition])
    guess = np.array([0.5,0.5])
    solutions=[]
    init_list = [i for i in iproduct(np.arange(0, 1, 0.24), np.arange(0, 1, 0.23))]
    for guess in init_list:
        sol, info, ier, mesg = fsolve(to_solve, guess, full_output=True, xtol=1.49012e-10, maxfev=1000)
        if ier == 1:
            if abs(sol[0]) < eps:
                sol[0] = 0
            if abs(sol[1]) < eps:
                sol[1] = 0
            if sol[0] < 0 or sol[1] < 0 or sol[0] > 1 or sol[1] > 1:
                continue
            if all([abs(sol[0] - i[0]) > eps or abs(sol[1] - i[1]) > eps for i in solutions]):
                solutions.append([float(sol[0]),float(sol[1])])
    return solutions
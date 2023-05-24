import numpy as np
from scipy.integrate import quad, dblquad, IntegrationWarning
from scipy.special import factorial
from scipy.interpolate import interp2d, RectBivariateSpline, interp1d
from scipy.optimize import fsolve, brentq
from itertools import product as iproduct
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Unable to load matplotlib plots will fail.")
    plt = None
import json

kmin = 0
kmax = 1000
kmin_sf = 0.99999
W=2

def make_Pk_sf(gamma,supplied_kmax=None):
    global kmin_sf
    if supplied_kmax:
        normterm = (1 / (1 - gamma)) * (supplied_kmax ** (1 - gamma) - (kmin_sf)**(1 - gamma))
    else:
        normterm = (1 / (gamma - 1) ) * (kmin_sf)**(1 - gamma)


    return lambda k: (1/normterm) * k ** ( - gamma)


def make_integrand_i1_sf(gamma,supplied_kmax=None):

    Pk = make_Pk_sf(gamma,supplied_kmax)
    return lambda k : k**2 * Pk(k)

def make_integrand_i1(kbar,crossover_k=12):
    def f(k):
        if k<crossover_k:
            return kbar ** (k) * np.exp(-kbar) * k * k / factorial(k)
        else:
            lny = k * np.log(kbar) + (1.5 - k) * np.log(k) + k - kbar - 0.91893853320467267 - 1 / (12.0 * k) + 1 / (
            360 * k ** 3) - 1 / (1260 * k ** 5) + 1 / (1680 * k ** 7)
            return np.exp(lny)
    return f


def integrate_i1(kbar, alpha, dist="er",gamma=3,supplied_kmax=None):
    global kmin,kmin_sf,W
    if dist == "sf":
        if alpha*kmin_sf >= W / 2:
            return 0
        val, err = quad(make_integrand_i1_sf(gamma, supplied_kmax), kmin_sf, W / (2*alpha), epsabs=1.49e-9, epsrel=1.49e-9, limit=500)
    elif dist == "er":
        val, err = quad(make_integrand_i1(kbar), kmin, W / (2*alpha), epsabs=1.49e-9, epsrel=1.49e-9, limit=500)
    return val * alpha * np.pi / 2


def make_integrand_i2(kbar, alpha):
    global W
    # Pk = lambda k: kbar ** (k) * np.exp(-kbar) * k / np.float128(factorial(k))
    Pk = hybrid_Pk(kbar) #this is k*Pk
    return lambda k: Pk(k) * ( (W / 2) *np.sqrt(1 - (W / (2*alpha * k)) ** 2) + alpha * k * np.arcsin(np.float128 (W / (2*alpha * k))))

def make_integrand_i2_sf(gamma,alpha,supplied_kmax=None):
    global W
    Pk = make_Pk_sf(gamma,supplied_kmax)#this is just Pk
    return lambda k: k*Pk(k) * ( (W / 2) *np.sqrt(1 - (W / (2*alpha * k)) ** 2) + alpha * k * np.arcsin(np.float128 (W / (2*alpha * k))))


def hybrid_Pk(kbar, crossover_k=12):# this is k *Pk!!!
    def f(k):
        if k < crossover_k:
            return kbar ** (k) * np.exp(-kbar) * k / np.float128(factorial(k))
        else:
            lny = k * np.log(kbar) + (0.5 - k) * np.log(k) + k - kbar - 0.91893853320467267 - 1 / (12.0 * k) + 1 / (
            360 * k ** 3) - 1 / (1260 * k ** 5) + 1 / (1680 * k ** 7)
            return np.exp(lny)

    return f


def stirling_Pk(kbar):
    def f(k):
        lny = k * np.log(kbar) + (0.5 - k) * np.log(k) + k - kbar - 0.91893853320467267 - 1 / (12.0 * k) + 1 / (
        360 * k ** 3) - 1 / (1260 * k ** 5) + 1 / (1680 * k ** 7)
        return np.exp(lny)

    return f


def get_kmax(kbar, pkeps=1e-20, crossover_k=12):
    # Pk = lambda k: kbar ** (k) * np.exp(-kbar) * k / np.float128(factorial(k))
    Pk = hybrid_Pk(kbar, crossover_k=crossover_k)
    k = 1
    while True:
        try:
            nonzero = Pk(k) > pkeps
        except OverflowError:
            break
        if k > kbar and not nonzero:
            break
        k += 1
    if k < kbar:
        raise ValueError("Warning: values too large")
    return k


def integrate_i2(kbar, alpha,dist="er",gamma=3,supplied_kmax=None):
    global W
    if dist == "sf":
        global kmin_sf
        if supplied_kmax:
            actualkmax = supplied_kmax
        else:
            global kmax
            actualkmax = kmax
        if alpha == 0 or alpha*actualkmax <= W / 2 :
            return 0
        actualkmin = W / (2*alpha) if W / (2*alpha) >  kmin_sf else kmin_sf
        val, err = quad(make_integrand_i2_sf(gamma, alpha,actualkmax), actualkmin, actualkmax, epsabs=1.49e-9, epsrel=1.49e-9,
                    limit=500)
        return val
    actualkmax = get_kmax(kbar)
    actualkmax = 5 * kbar
    if alpha == 0 or actualkmax < W / (2*alpha):
        return 0
    val, err = quad(make_integrand_i2(kbar, alpha), W / (2*alpha), actualkmax, epsabs=1.49e-9, epsrel=1.49e-9, limit=500)
    return val


def get_kbar_sf(gamma=3,supplied_kmax=None):
    Pk = make_Pk_sf(gamma,supplied_kmax)
    global kmax,kmin_sf
    actualkmax = supplied_kmax if supplied_kmax else np.inf
    val, err = quad( lambda k : k * Pk(k), kmin_sf, actualkmax,epsabs=1.49e-9, epsrel=1.49e-9, limit=500)
    return val


def get_int(kbar, alpha,dist="er",gamma=3,supplied_kmax=None):
    global W
    # factor of 0.5 is to normalize the frequency distribution (g(w) = 1/2 between -1 and 1)
    if dist == "sf":
        kbar = get_kbar_sf(gamma, supplied_kmax)
    return (1 / W) * (1 / kbar) * (integrate_i1(kbar,alpha,dist,gamma,supplied_kmax) + integrate_i2(kbar,alpha,dist,gamma,supplied_kmax))


alpha = lambda R, f, lam: lam * R * (R * f + 1 - f)

alphaf1 = lambda R, lam: lam * R * R

alphaf0 = lambda R, lam: lam * R


def make_full_integrand(kbar, alpha):
    Pk = lambda k: kbar ** (k) * np.exp(-kbar) * k / np.float128(factorial(k))
    g = lambda w: 0 if w > W/2 or w < -W/2 else 1 / W
    return lambda w, k: Pk(k) * g(w) * np.sqrt(1 - (w / (k * alpha)) ** 2)


def full_integrate(kbar, alpha):
    gfun = lambda k: -alpha * k
    hfun = lambda k: alpha * k
    print("alpha = %.4f ... " % alpha, end='')
    val, err = dblquad(make_full_integrand(kbar, alpha), kmin, 5 * kbar, gfun, hfun)
    print("complete.")
    return 0.5 * val / kbar


###
# Move on to interpolation
###

def get_interpolated_integral(kbar, mode="r", fname="interpolation_values.json",gamma=3,dist="er",supplied_kmax=None):
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
        print(" _|_ loaded -|- ", end='')
    else:
        alpha = np.arange(0, 3, 0.001)
        if dist == "sf":
            kbar = get_kbar_sf(gamma,supplied_kmax)
        inta = []
        for a in alpha:
            try:
                val = get_int(kbar, a,dist,gamma,supplied_kmax)
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
            d = {"alpha": [i for i in alpha], "integral": inta, "kbar": kbar, "dist": dist, "width" : W}
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


def get_interpolated_integral2(toalpha=1, mode="w", fname="interpolation_values_2d.json"):
    if mode == "r":
        d = json.load(open(fname))
        k_vec = d["k_vec"]
        alpha = np.array(d["alpha"])
        z = np.array(d["z"])
    else:
        k_vec = list(range(10, 15))
        alpha = np.arange(0, toalpha, 0.004)
        z = np.zeros((len(k_vec), len(alpha)))
        for i, j in iproduct(range(len(k_vec)), range(len(alpha) - 1)):
            try:
                z[i, j + 1] = get_int(k_vec[i], alpha[j + 1])
                # z[i, j + 1] = full_integrate(k_vec[i], alpha[j + 1])
            except OverflowError:
                print("Unable to calculate (%i,%.4f); using 0" % (k_vec[i], alpha[j + 1]))
        # return interp2d(k_vec,alpha,z)
        if mode == "w":
            d = {"k_vec": k_vec, "alpha": [i for i in alpha], "z": [[i for i in row] for row in z]}
            json.dump(d, open(fname, "w"))
    return RectBivariateSpline(k_vec, alpha, z)


###
# Move on to solution
###

def combined_integral(intfunc, kbar, frac, lam, R):
    try:
        return frac * intfunc(alphaf1(R, lam)) + (1 - frac) * intfunc(alphaf0(R, lam))
    except:
        return frac * intfunc(kbar, alphaf1(R, lam), grid=False) + (1 - frac) * intfunc(kbar, alphaf0(R, lam),
                                                                                        grid=False)


def solve_R(intfunc, lam, frac, kbar, eps=5e-10):
    to_solve = lambda R: combined_integral(intfunc, kbar, frac, lam, R) - R
    checkpoints = np.linspace(0, 1, 15)
    # to_solve(checkpoints)
    # Rmin,Rmax = [0,1]
    # while True:
    #    try:
    #        root = brentq(to_solve,Rmin,Rmax)
    #    except ValueError:

    all_sols = [0]
    for point in checkpoints:
        sol, info, ier, mesg = fsolve(to_solve, point, full_output=True, xtol=1.49012e-8, maxfev=1000)
        if ier == 1:
            if all([abs(sol - i) > eps for i in all_sols]):
                all_sols.append(float(sol))
    return sorted(all_sols)
    # all_sols = np.unique(np.array(sorted(np.abs(all_sols))).astype(str)).astype(np.float)
    # d = np.diff(all_sols)
    # return all_sols[np.concatenate([[True], d > eps])]
    # return all_sols


def guan_stability_check(intfunc, lam, frac, kbar, r_value=0, stability_threshold=0.5, epsilon_init=1e-4, eps=1e-11,
                         full_output=False):
    if abs(combined_integral(intfunc, kbar, frac, lam, r_value) - r_value) > eps:
        print("%.4f is not a solution, cannot check stability", r_value)
    epsilon_vec = [epsilon_init]
    max_length = 10
    while len(epsilon_vec) < max_length:
        new_r = r_value + epsilon_vec[-1]
        new_epsilon = abs(combined_integral(intfunc, kbar, frac, lam, new_r) - r_value)
        epsilon_vec.append(new_epsilon)
        if new_epsilon < eps:
            break
    stable = sum(map(lambda x: 1 if x < 0 else 0, np.diff(epsilon_vec))) / (len(epsilon_vec) - 1)
    return stable > stability_threshold if not full_output else epsilon_vec


def get_energy_level(intfunc, lam, frac, kbar, r_value=0):
    if abs(combined_integral(intfunc, kbar, frac, lam, r_value) - r_value) > 1e-4:
        print("[f=%.4f, lambda=%.8f, k=%.2f] %.5f is not a solution, cannot check stability" % (frac,lam,kbar,r_value))
    f_of_r = lambda r: r - combined_integral(intfunc, kbar, frac, lam, r)
    return quad(f_of_r, 0, r_value)[0]


def plot_energy_function(intfunc, lam, frac, kbar):
    rvec = np.linspace(0, 1, num=1000)
    fofrvec = [0]
    f_of_r = lambda r: r - combined_integral(intfunc, kbar, frac, lam, r)
    for r1 in rvec[1:]:
        r0 = r1 - 0.001
        fofrvec.append(fofrvec[-1] + quad(f_of_r, r0, r1)[0])
    fofrvec = np.array(fofrvec)
    plt.plot(rvec, fofrvec)


def sign_change(x, func, eps=1e-6, maxeps=0.01):
    step = eps
    while func(x - step) * func(x + step) > 1:
        step += eps
        if step > maxeps:
            raise ValueError("There's no zero at %.5f!" % x)
    # print("step = %.10f"%step)
    return func(x + step) / abs(func(x + step))


def plot_stability(intfunc, lam, frac, kbar, eps=1e-11):
    all_sols = solve_R(intfunc, lam, frac, kbar, eps)
    to_solve = lambda R: combined_integral(intfunc, kbar, frac, lam, R) - R

    import matplotlib.pyplot as plt
    Rvec = np.arange(0, 1, 0.001)
    plt.plot(Rvec, [combined_integral(intfunc, kbar, frac, lam, R_) - R_ for R_ in Rvec], label="RR")
    # plt.plot(Rvec,Rvec,label="R")
    plt.axhline(0)
    for i in all_sols:
        plt.scatter(i, 0)
    plt.xlabel("f(R) - R")
    plt.ylabel("R")


def check_stability(intfunc, lam, frac, kbar, eps=1e-11):
    all_sols = solve_R(intfunc, lam, frac, kbar, eps)
    to_solve = lambda R: combined_integral(intfunc, kbar, frac, lam, R) - R
    return [sign_change(i, to_solve) for i in all_sols]


def get_bifurcation_point(intfunc, frac, kbar, lam_interval, dl=1e-10):
    lmin, lmax = lam_interval
    lmaxsol = solve_R(intfunc, lmax, frac, kbar)
    lminsol = solve_R(intfunc, lmin, frac, kbar)
    if len(lminsol) != 1 or len(lmaxsol) == 1:
        print("(f=%.4f) This is not going to work: %i sols at %.3f and %i sols at %.3f" % (
            frac, len(lminsol), lmin, len(lmaxsol), lmax))

    while lmax - lmin > dl:
        lam = (lmax + lmin) / 2
        lsol = solve_R(intfunc, lam, frac, kbar, eps=dl)
        if len(lsol) >= 2:
            lmax = lam
            lmaxsol = lsol
        elif len(lsol) == 1:
            lmin = lam
            lminsol = lsol
            break
    return lam, lmaxsol


def get_R_curve(intfunc, f, kbar, lam_vec):
    Rsolved = [solve_R(intfunc, lam, f, kbar) for lam in lam_vec]
    return Rsolved


def plot_Rsolved(lam_vec, Rsolved, color='k', label="0.1"):
    from matplotlib import pyplot as plt
    for l_, r_ in zip(lam_vec, Rsolved):
        try:
            plt.plot([l_ for i in range(len(r_))], r_, '.', color=color)
        except:
            plt.plot(l_, r_, '.', color=color)


def get_d(intfunc, kbar, frac, lam_vec, eps=1e-11):
    found_jump = False
    for lam in lam_vec:
        all_sols = solve_R(intfunc, lam, frac, kbar, eps)
        signs = check_stability(intfunc, lam, frac, kbar)
        if signs[0] > 0:
            lmax = lam
            break
        if not found_jump:
            for ind, s in enumerate(signs):
                if s == -1 and ind > 0:
                    found_jump = True
                    lmin = lam
    try:
        return lmax - lmin
    except:
        print("(f = %.3f) Solution not found." % frac)
        return 0


def load_energy_array(fname="/home/micha/phd/oscillators/zhang_results_energy.json"):
    from itertools import product as iproduct
    d = json.load(open(fname))
    lamvec = np.array(d["lambda_vec"])
    fvec = np.array(sorted(map(float, d["energy_level"].keys())))
    A = np.zeros((len(fvec), len(lamvec)))
    for [(fidx, f), (lamidx, lam)] in iproduct(enumerate(sorted(d["energy_level"].keys(), key=float)),
                                               enumerate(d["lambda_vec"])):
        if len(d["energy_level"][f][lamidx]) == 3 and lam < 0.59:
            A[fidx, lamidx] = d["energy_level"][f][lamidx][1]
        else:
            A[fidx, lamidx] = np.nan
    return lamvec, fvec, np.ma.masked_array(data=A, mask=np.isnan(A))


def plot_energy_barrier_height(f_str, fname="/home/micha/phd/oscillators/zhang_results_energy.json"):
    d = json.load(open(fname))
    lamvec = np.array(d["lambda_vec"])
    lamvec_ltd, energy_barrier = zip(*[(l, e[1]) for l, e in zip(lamvec, d["energy_level"][f_str]) if len(e) == 3])
    plt.figure()
    plt.plot(lamvec_ltd, energy_barrier)
    plt.show()

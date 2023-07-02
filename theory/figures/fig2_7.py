from __future__ import division,print_function
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm,colors as mplcolors
from matplotlib.collections import LineCollection
from sys import argv,exit
from os.path import basename,join
try:
    import sisplex,intersis,two_nets_sync as tns
except ImportError:
    from oscillators import sisplex,intersis,two_nets_sync as tns

from resistornets import theory


def get_lambdac(intfunc,lambdamin=0,lambdamax=1,precision=1e-10,function = tns.solve_R1R2_double_solutions):
    while lambdamax - lambdamin > precision:
        thislambda = 0.5 * (lambdamax + lambdamin)
        nsols = len(function(intfunc,[thislambda,thislambda],1,interaction_type="interdependent"))
        if nsols > 1:
            lambdamax = thislambda
        else:
            lambdamin = thislambda
    return lambdamax



kbar=50
integral_fname = "/home/micha/phd/oscillators/interpolation_values_combined_1d_k50_HD.json"
intfunc = tns.sns.get_interpolated_integral(kbar,"r",integral_fname)
sis_integral_fname = "/home/micha/phd/oscillators/sis_interpolation_values_combined_1d_k50_HD.json"
sisintfunc = sisplex.get_interpolated_integral(kbar,"r",sis_integral_fname)
rrsisintfunc = sisplex.get_rr_integral(kbar)
lambdac = get_lambdac(intfunc)
betac = get_lambdac(intfunc=sisintfunc,function=sisplex.solve_R1R2_double_solutions)
betacrr = get_lambdac(intfunc = rrsisintfunc,function=sisplex.solve_R1R2_double_solutions)

dlambdavec = np.logspace(-9,-1,20)
Rvec = []
Thetavec,Thetavec2 = [[],[]]
for dlambda in dlambdavec:
    thislambda = (lambdac+dlambda,lambdac+dlambda)
    Rvec.append(tns.solve_R1R2_double_solutions(intfunc,thislambda,1,interaction_type="interdependent")[-1][0][0])
    thisbeta = (betac+dlambda,betac+dlambda)
    Thetavec.append(sisplex.solve_R1R2_double_solutions(sisintfunc,thisbeta,1,interaction_type="interdependent")[-1][0][0])
    thisbeta = (betacrr + dlambda, betacrr + dlambda)
    Thetavec2.append(sisplex.solve_R1R2_double_solutions(rrsisintfunc, thisbeta, 1, interaction_type="interdependent")[-1][0][0])

pc = 0
#p_vec pc + dlambdavec
#theory.get_interdep_pinf(p_vec, theory.P_inf_factory(kbar))


Rvec = np.array(Rvec)
Thetavec = np.array(Thetavec)
Thetavec2 = np.array(Thetavec2)

betac = get_lambdac(intfunc=sisintfunc,function=sisplex.solve_R1R2_double_solutions)

plt.figure()
plt.loglog(dlambdavec,Rvec,'.-')
plt.loglog(dlambdavec,Thetavec,'.-')
plt.loglog(dlambdavec,Thetavec2,'.-')
plt.loglog(dlambdavec,np.sqrt(dlambdavec),'.-')
plt.show()


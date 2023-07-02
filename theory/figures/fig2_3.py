from __future__ import division,print_function
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm,colors as mplcolors
from matplotlib.collections import LineCollection
from sys import argv,exit
from os.path import basename,join
from os import environ
from numdifftools import Jacobian
from numpy.linalg import det
import h5py
try:
    import two_nets_sync as tns
except ImportError:
    from oscillators import two_nets_sync as tns
from itertools import product as iproduct


def fp_type(d,t):
    if d<0:
        return "saddle"
    if d>0 and t>0:
        return "unstable"
    if d>0 and t<0:
        return "stable"


def runname2metadata(runname):
    d = {}
    for s in basename(runname).strip(".hdf5").split("_")[1:]:
        i=0
        key=''
        val=''

        while  i < len(s) and not s[i].isnumeric():
            key+=s[i]
            i+=1
        while  i < len(s) and (s[i].isnumeric() or s[i] == "."):
            val+=s[i]
            i+=1
        d[key] = val

    return d


def make_varying_color_collection(xdata,ydata,num_segments=100,cmap=cm.viridis,linspace=True):
    linesegments = np.array_split(np.vstack((xdata,ydata)).T,num_segments)
    for idx, linesegment in enumerate(linesegments[:-1]):
        linesegments[idx] = np.vstack((linesegments[idx], linesegments[idx + 1][0]))
        if linspace:
            coll = LineCollection(linesegments,colors=cmap(np.linspace(0,1,num_segments)))
        else:
            coll = LineCollection(linesegments, colors=cmap(np.logspace(np.log10(1/num_segments), 0, num_segments)))
    return coll
randid=''
#argv[1] = "/home/micha/cudamoto_results/single_run_N131072_k50_f1lambda1_0.03_lambda2_0.05_id2507.hdf5"
integral_fname = None

if len(argv) <= 3 and "hdf5" in argv[1]:
    hd5_fname = argv[1]
    for runname, run in h5py.File(hd5_fname).items():
        break
    d = runname2metadata(hd5_fname)
    for key,val in runname2metadata(runname).items():
        d[key] = val
    print(d)
    kbar = float(d["k"])
    frac = float(d["f"])
    interaction_id = int(d["interaction"]) - 1
    lam1 = float(d["lambdaA"])
    lam2 = float(d["lambdaB"])
    randid = d["id"]
    if len(argv) == 3 and "interp" in argv[2]:
        integral_fname = argv[2]

elif len(argv) < 5:
    print("kbar frac interaction_id lam1 lam2")
    exit()

else:
    kbar = int(argv[1])
    frac = float(argv[2])
    interaction_id = int(argv[3])
    lam1 = float(argv[4])
    lam2 = float(argv[5])


if len(argv) > 6 and "interp" in argv[6]:
    integral_fname = argv[6]
elif not integral_fname:
    integral_fname = "/home/micha/phd/oscillators/interpolation_values_combined_1d_k%i_HD.json"%kbar
if len(argv) > 6 and "hdf5" in argv[6]:
    hd5_fname = argv[6]
print("Loading integral from %s ... "%integral_fname, end = "")
intfunc = tns.sns.get_interpolated_integral(kbar,fname=integral_fname)


lam = (lam1,lam2)

interaction_type=["interdependent","competitive","hybrid", "mixed"][interaction_id]
to_solve = tns.combined_R1R2(intfunc, lam, frac, interaction_type=interaction_type)

x = np.linspace(0,1,2000)
y = np.linspace(0,1,2000)
u,v = to_solve(np.meshgrid(x,y))
plt.figure()
plt.streamplot(x,y,u,v,linewidth=2*np.hypot(u,v),density=2.5)
plt.xlabel("$R_1$")
plt.ylabel("$R_2$")

sols = tns.solve_R1R2_double_solutions(intfunc, lam, frac,interaction_type=interaction_type)
eps = 1e-11
jfp_list=[]
print("(lambda1, lambda2) \t : fptype    det(J)   tra(J)   tra(J)^2 - 4*det(J)")

for sol in sols:
    if type(sol[0]) is list:
        sol = sol[0]
    if abs(sol[0]) < eps or abs(sol[1]) < eps:
        method = "forward"
    else:
        method = "backward"
    jac = Jacobian(to_solve,method=method)
    jfp = jac(sol)
    jfp_list.append(jfp)
    thisdet,tra = [det(jfp),np.trace(jfp)]
    print("(%.6f, %.6f)"%tuple(sol),end="\t")
    print(" : %s %.6f %.6f %.6f"%(fp_type(thisdet,tra),thisdet,tra,tra*tra - 4*thisdet))
    if thisdet<0 or thisdet>0 and tra>0:
        plt.scatter(sol[0],sol[1],marker='o',s=60,color="darkred")
    else:
        plt.scatter(sol[0],sol[1],marker='x',s=60,color="darkgreen")
plt.title("$\\bar{k} = %i$ , $f = %.2f$ , $\lambda = (%.4f,%.4f)$ - %s"%(kbar,frac,lam1,lam2,interaction_type))

try:
    import h5py
    with h5py.File(hd5_fname) as f:
        for ds_k,ds in f.items():
            r1,r2 = zip(*ds)
            #plt.plot(r1,r2,color="green",lw=2)
            coll = make_varying_color_collection(r1,r2)
            plt.gca().add_collection(coll)
            #plt.scatter(r1[0],r2[0],color="red",marker="^")

except:
    pass


plt.axis([-0.01, 1.01, -0.01, 1.01])
plt.tight_layout()
plt.savefig("/tmp/phase_flow_k%i_f%.2f_x%i_la%.6f_lb%.6f_id%s.pdf"%(kbar,frac,interaction_id,lam1,lam2,randid))


if False:
    Z1 = np.zeros((len(x),len(y)))
    Z2 = np.zeros((len(x),len(y)))
    for i,j in iproduct(range(len(x)),range(len(y))):
        Z1[i,j] = to_solve([x[i],y[j]])[0]
        Z2[i,j] = to_solve([x[i],y[j]])[1]

    plt.figure()

    plt.contour(x,y,Z1.T,[0],colors="red",label="$\dot{R}_1 = 0$")
    plt.contour(x,y,Z2.T,[0],colors="blue",label="$\dot{R}_2 = 0$")
    plt.legend(loc="best")



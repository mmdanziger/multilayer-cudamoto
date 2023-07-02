from __future__ import division,print_function
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm,colors as mplcolors
from matplotlib.collections import LineCollection
from sys import argv,exit
from os.path import basename,join
from os import environ
import h5py

from itertools import product as iproduct

sis=False



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


runlist = []


def get_lifetime(fname):
    global sis,runlist
    if fname.split(".")[-1] == "hdf5":
        f = h5py.File(fname)
        sis = False
        betalist=[]
        for run in f:
            betalist.append(runname2metadata(run)["lambdaA"])

        lifetime = np.array(sorted([(float(beta),0.01*len(f.get(x))) for beta,x in zip(betalist,f) if f.get(x)[-1][0] < 0.05]))

    elif fname.split(".")[-1] == "json":
        sis = True
        betalist,runlist = json.load(open(fname))
        lifetime = np.array(sorted([(float(beta),0.01*len(run)) for beta,run in zip(betalist,runlist) if run[-1] < 0.005]))
    else:
        raise ValueError("Unknown filetype: '%s' supported filetypes: 'hd5' (sync) 'json' (sis)")
    betac_lowerbound = lifetime.max(axis=0)[0]
    betac_upperbound = min(filter(lambda x: float(x)>betac_lowerbound,betalist))
    betac = betac_upperbound#0.5*(betac_upperbound + betac_lowerbound)
    print("test")
    return betac,betalist,lifetime

if __name__ == "__main__":

    betac,betalist,lifetime = get_lifetime(argv[1])

    beta_above_to_plot = sorted(filter(lambda x: x >= betac, betalist), key=lambda x: abs(float(x) - float(betac)))
    beta_below_to_plot = sorted(filter(lambda x: x < betac, betalist), key=lambda x: abs(float(x) - float(betac)))

    beta_to_plot = sorted(list(beta_above_to_plot[:2]) + list(beta_below_to_plot[:2]), reverse=True)


    dt=0.01



    plt.figure()

    paramstring = "\\beta" if sis else "\lambda"
    orderparamstring = "\\Theta" if sis else "R"
    for beta in beta_to_plot:
        if sis:
            for beta_,run in zip(betalist,runlist):
                if "%.6f"%beta_ == "%.6f"%beta:
                    plt.plot([dt * i for i in range(len(run))], run, lw=2, label="$%s = %.6f$"%(paramstring, float(beta)))
        else:
            f = h5py.File(argv[1])
            for run in f:
                if runname2metadata(run)["lambdaA"] == beta:

                    r1, r2 = zip(*f.get(run))
                    runlist.append(r1)
                    plt.plot([dt*i for i in range(len(r1)) ],r1,lw=2,label="$%s = %.6f$"%(paramstring,float(beta)))
    plt.legend(loc="best")
    plt.xlabel("$t$")
    plt.ylabel("$"+orderparamstring+"$")
    plt.figure()
    betac = float(betac)
    plt.loglog(betac - lifetime[:,0], lifetime[:,1]/max(lifetime[:,1]),'.-')
    plt.xlabel("$%s_c - %s$"%(paramstring,paramstring))
    plt.ylabel("$T/T_{\\rm max}$")
    plt.loglog(betac - lifetime[:,0], (betac - lifetime[:,0])**(-0.5 ) / max((betac - lifetime[:,0])**(-0.5 )),'--')



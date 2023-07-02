from __future__ import division,print_function
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from sys import argv
from os.path import basename,join
try:
    from pyNoN import logbin
except ImportError:
    from sys import path
    path.append("/home/micha/phd")
    from pyNoN import logbin
import h5py

def fname2metadata(fname):
    d = {}
    for s in basename(fname).rstrip(".hdf5").split("_")[1:]:
        i=0
        key=''
        while not s[i].isnumeric():
            key+=s[i]
            i+=1
        d[key] = s[i:]
    return d

fname = argv[1]
lambda1 = 0.135
lambda2 = 0.2175
frac = 0.5
with h5py.File(fname) as f:
    ds = list(f.values())[0]
    r1,r2 = zip(*ds)
r1 = np.array(r1)
r2 = np.array(r2)
t = np.array([0.01*i for i in range(len(r1))])

fig1 = plt.figure()
ax1 = fig1.add_subplot(221)
plt.plot(t,r1,label="r1")
plt.plot(t,r2,label="r2")
plt.legend(loc="best")

ax2 = fig1.add_subplot(222)
plt.plot(t,lambda1 *(1 - frac + frac*np.array(r2)),label="lam1")
plt.plot(t,lambda2 *(1 - frac + frac*np.array(r1)),label="lam2")
plt.axhline(y=0.1,ls="--",color="red",label="lamcrit")
plt.legend(loc="best")

ax3 = fig1.add_subplot(223)
dr1dt = np.diff(r1)/np.diff(t)
dr2dt = np.diff(r2)/np.diff(t)
plt.plot(t[1:],dr1dt,label="$dr_1/dt$")
plt.plot(t[1:],dr2dt,label="$dr_2/dt$")
plt.legend(loc="best")

lb1 = logbin.LogBin(zip(t[1:],dr1dt),type="linear")
lb2 = logbin.LogBin(zip(t[1:],dr2dt),type="linear")
ddr1 = np.diff(lb1.yavg)/np.diff(lb1.xavg)
ddr2 = np.diff(lb2.yavg)/np.diff(lb2.xavg)

ax4 = fig1.add_subplot(224)
plt.plot(lb1.xavg[:-1],ddr1,label="$d^2r_1/dt^2$")
plt.plot(lb2.xavg[:-1],ddr2,label="$d^2r_2/dt^2$")
plt.legend(loc="best")

plt.show()
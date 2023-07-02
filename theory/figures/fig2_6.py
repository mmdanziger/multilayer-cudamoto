from __future__ import division,print_function
import json
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from matplotlib import cm,colorbar as cb,colors as mplcolors
from matplotlib.collections import LineCollection
from sys import argv,exit
from os.path import basename,join
from os import environ
from numdifftools import Jacobian
from numpy.linalg import det
import h5py


def make_varying_color_collection(xdata,ydata,num_segments=100,cmap=cm.viridis,linspace=True):
    linesegments = np.array_split(np.vstack((xdata,ydata)).T,num_segments)
    for idx, linesegment in enumerate(linesegments[:-1]):
        linesegments[idx] = np.vstack((linesegments[idx], linesegments[idx + 1][0]))
        if linspace:
            coll = LineCollection(linesegments,colors=cmap(np.linspace(0,1,num_segments)))
        else:
            coll = LineCollection(linesegments, colors=cmap(np.logspace(np.log10(1/num_segments), 0, num_segments)))
    return coll

hd5_fname = argv[1]
h=0.01
runs = []
plt.figure(1)
with h5py.File(hd5_fname) as f:
    for ds_k,ds in f.items():
        r1,r2 = zip(*ds)
        runs.append([r1,r2])
        #plt.plot(r1,r2,color="green",lw=2)
        coll = make_varying_color_collection(r1,r2)
        plt.gca().add_collection(coll)
        #plt.scatter(r1[0],r2[0],color="red",marker="^")
plt.savefig("/tmp/"+basename(hd5_fname[:-4])+".pdf")
plt.figure(2)

ims = plt.imshow( np.outer(np.linspace(0,np.sqrt(h*len(r1))),np.linspace(0,np.sqrt(h*len(r1)))), cmap=cm.viridis)
plt.figure(1)
cb1 = plt.colorbar(mappable=ims,ax=plt.gca(),)
cb1.set_label('$t$')

plt.xlabel("$R_1$")
plt.ylabel("$R_2$")

plt.close(2)

plt.figure()

for i in range(len(runs)):
    for j in range(len(runs)):
        if i < j:
            t = [h*i for i in range(len(runs[i][0]))]
            X = np.array(list(runs[i]))
            Y = np.array(list(runs[j]))
            plt.plot(t, (norm((X-Y),axis=0) / norm(0.5*(X+Y),axis=0)) )
plt.xlabel("$t$")
plt.ylabel("$|R_i - R_j| / <R> $")

plt.show()

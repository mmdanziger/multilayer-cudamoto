from __future__ import division,print_function
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm,colors as mplcolors
from sys import argv


fname = "/storage/home/micha/cudamoto_results/2/theory/two_net_result_competitive_k12_f0.5000.json" if len(argv) <2 else argv[1]

d = json.load(open(fname))
lamvec = sorted(d.keys(), key=float)
lamvec2 = sorted(d[list(d.keys())[0]].keys(), key=float)
nsols = np.zeros((len(lamvec), len(lamvec2)))
nstablesols = np.zeros((len(lamvec), len(lamvec2)))

for l1 in lamvec:
    for l2 in lamvec2:
        nsols[lamvec.index(l1), lamvec2.index(l2)] = len(d[l1][l2])
        nstablesols[lamvec.index(l1), lamvec2.index(l2)] = sum([i[1] for i in d[l1][l2]])

plt.figure()
norm = mplcolors.BoundaryNorm([i+0.5 for i in range(int(nsols.max()) + 1)], cm.viridis.N)
plt.pcolor(np.array(lamvec).astype(float), np.array(lamvec2).astype(float), nsols.T, cmap=cm.viridis, norm=norm,
           vmin=.5, vmax=np.max(nsols) + .5)
plt.colorbar(ticks=np.arange(np.min(nsols), np.max(nsols) + 1))
plt.show()

plt.figure()
norm = mplcolors.BoundaryNorm([i+0.5 for i in range(int(nstablesols.max()) + 1)], cm.viridis.N)
plt.pcolor(np.array(lamvec).astype(float), np.array(lamvec2).astype(float), nstablesols.T, cmap=cm.viridis, norm=norm,
           vmin=.5, vmax=np.max(nstablesols) + .5)
plt.colorbar(ticks=np.arange(np.min(nstablesols), np.max(nstablesols) + 1))
plt.show()
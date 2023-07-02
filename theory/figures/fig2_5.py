from __future__ import division,print_function
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm,colors as mplcolors
from sys import argv
import h5py
f = h5py.File(argv[1])
guide_dot_count=15
make_subplots=False
r1,r2 = zip(*f.get("rvalues"))
l1,l2=zip(*f.get("lamvalues"))

def reverse_direction(x):
    halfway = len(x)//2
    return x[halfway:] + x[:halfway]

halfway = len(r1)//2
if l1[halfway] < l1[0]:
    l1 = reverse_direction(l1)
    l2 = reverse_direction(l2)
    r1 = reverse_direction(r1)
    r2 = reverse_direction(r2)
r1,r2 = np.array(r1),np.array(r2)

this_cm = cm.gist_rainbow
marker_sequence = ['o', '^', 's', 'D']

if make_subplots:
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
else:
    plt.figure(1)
plt.plot(l1[:halfway],l2[:halfway],zorder=20,lw=2)
marker_count=0
for idx in range(0,halfway+1,halfway//guide_dot_count):
    plt.scatter(l1[idx],l2[idx],c=this_cm(idx/halfway),s=60,zorder=21,marker=marker_sequence[marker_count%len(marker_sequence)])
    marker_count+=1
plt.xlabel(r"$\lambda_1$")
plt.ylabel(r"$\lambda_2$")
plt.grid(1)
plt.axis("equal")
plt.tight_layout()
if make_subplots:
    plt.subplot(1,2,2)
else:
    plt.figure()
plt.plot(r1[:halfway],".-",color="darkblue",label="$R_1$",lw=2)
plt.plot(list(reversed(r1[halfway:])),".--",color="darkblue",lw=2)
plt.plot(r2[:halfway],".-",color="darkred",label="$R_2$",lw=2)
plt.plot(list(reversed(r2[halfway:])),".--",color="darkred",lw=2)
marker_count=0
for idx in range(0,halfway+1,halfway//guide_dot_count):
    plt.scatter(idx,-0.05, c=this_cm(idx/halfway),s=60,zorder=21,marker=marker_sequence[marker_count%len(marker_sequence)], clip_on=False)
    marker_count+=1
plt.axis("tight")
plt.ylim(ymin=0)
plt.xticks([])
plt.xlabel("$s$")
plt.ylabel("$R$")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
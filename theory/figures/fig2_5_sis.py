from __future__ import division,print_function
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm,colors as mplcolors
from sys import argv
import h5py
bv,hv = np.load(argv[1])
guide_dot_count=15
make_subplots=False
r1,r2 = np.hsplit(hv,2)
l1,l2= np.hsplit(bv,2)

orig_color_list = [ "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#fabebe", "#008080", "#e6beff", "#aa6e28", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000080", "#808080"]
color_list = orig_color_list[:]
color_dict={}
color1 = color_list.pop(0)
color2 = color_list.pop(0)
color3 = color_list.pop(0)
color4 = color_list.pop(0)


def reverse_direction(x):
    halfway = len(x)//2
    return np.array(list(x[halfway:]) + list(x[:halfway]))


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
plt.xlabel(r"$\beta_1$")
plt.ylabel(r"$\beta_2$")
#plt.grid(1)
plt.axis("equal")
plt.tight_layout()
plt.axis([0.007,0.5,0.007,0.5])
topspace = plt.gcf().subplotpars.top
bottomspace = plt.gcf().subplotpars.bottom

if make_subplots:
    plt.subplot(1,2,2)
else:
    plt.figure(3,figsize=(8,5))
if "LS" in argv[1]:
    label_tag="LS"
    thismarker="o"
    color1=color3
    color2=color4
else:
    label_tag="LF"
    thismarker="^"
plt.plot(r1[:halfway],"-",color=color1,label="$\Theta_1$ "+label_tag,lw=2,marker=thismarker)
plt.plot(list(reversed(r1[halfway:])),"--",color=color1,lw=2,marker=thismarker)
plt.plot(r2[:halfway],"-",color=color2,label="$\Theta_2 $"+label_tag,lw=2,marker=thismarker)
plt.plot(list(reversed(r2[halfway:])),"--",color=color2,lw=2,marker=thismarker)
marker_count=0
for idx in range(0,halfway+1,halfway//guide_dot_count):
    plt.scatter(idx,-0.05, c=this_cm(idx/halfway),s=60,zorder=21,marker=marker_sequence[marker_count%len(marker_sequence)], clip_on=False)
    marker_count+=1
plt.axis([0,halfway,0,1.01*max((max(r1),max(r2)))])
plt.ylim(ymin=0)
plt.xticks([])
#plt.xlabel("$s$")
plt.ylabel("$\Theta$")
plt.legend(loc="best")
plt.tight_layout()
plt.subplots_adjust(bottom=bottomspace,top=topspace)
plt.show()
from __future__ import division,print_function
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm,colors as mplcolors
from sys import argv
import h5py
from os.path import basename
data = {}


pretty = {"only" : "$|cos|$ ","rcos" : "$r_i |cos|$", "r_i" : "$r_i$", "sqrtrcos" : "$\sqrt{r_i cos}$"}
labelled_yet={}

orig_color_list = [ "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#fabebe", "#008080", "#e6beff", "#aa6e28", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000080", "#808080"]



color_list = orig_color_list[:]
color_dict={}


def getcoslabel(fname):
    fname = basename(fname)
    if "cos" not in fname:
        return "r_i"
    idx1=fname.index("cos")
    idx2=fname.index("_",idx1)
    return fname[idx1+3:idx2]

start_figures_from=4
show_global = False
for fname in sorted(argv[1:]):
    if not "hdf5" in fname:
        continue
    f = h5py.File(fname)
    guide_dot_count=15
    make_subplots=False
    iscos = "cos" in basename(fname)
    r1_k, r2_k = zip(*f.get("rvalues_kweight"))
    r1_g, r2_g = zip(*f.get("rvalues_global"))
    l1, l2=zip(*f.get("lamvalues"))
    thisdata = {"lambda" : [l1,l2], "r_g" : [r1_g,r2_g], "r_k" : [r1_k,r2_k]}
    thiskey = getcoslabel(fname)
    data[thiskey] = thisdata
    if not thiskey in labelled_yet:
        labelled_yet[thiskey] = False

    def reverse_direction(x):
        halfway = len(x)//2
        return x[halfway:] + x[:halfway]

    halfway = len(r1_k) // 2
    if l1[halfway] < l1[0]:
        l1 = reverse_direction(l1)
        l2 = reverse_direction(l2)
        r1_k = reverse_direction(r1_k)
        r2_k = reverse_direction(r2_k)
        r1_g = reverse_direction(r1_g)
        r2_g = reverse_direction(r2_g)

    r1_k, r2_k = np.array(r1_k), np.array(r2_k)
    r1_g, r2_g = np.array(r1_g), np.array(r2_g)


    this_cm = cm.gist_rainbow
    marker_sequence = ['o', '^', 's', 'D']

    if make_subplots:
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
    else:
        plt.figure(1)
    plt.plot(l1[:halfway], l2[:halfway], zorder=20, lw=2)
    marker_count=0
    for idx in range(0,halfway+1,halfway//guide_dot_count):
        plt.scatter(l1[idx], l2[idx], c=this_cm(idx / halfway), s=60, zorder=21, marker=marker_sequence[marker_count % len(marker_sequence)])
        marker_count+=1
    plt.xlabel(r"$\lambda_1$")
    plt.ylabel(r"$\lambda_2$")
    plt.grid(1)
    plt.axis("equal")
    plt.tight_layout()
    if make_subplots:
        plt.subplot(1,2,2)
    else:
        plt.figure(start_figures_from)
    coslabel = " " + pretty[getcoslabel(fname)]
    thismarker = '^' if iscos else 'o'
    if len(color_list) < 2:
        color_list = orig_color_list[:]

    if not labelled_yet[thiskey]:
        color1 = color_list.pop(0)
        color2 = color_list.pop(0)
        color_dict[thiskey] = [color1, color2]

        if show_global:
            color3 = color_list.pop(0)
            color4 = color_list.pop(0)
            color_dict[thiskey] += [color3,color4]
        plt.figure(start_figures_from)
        plt.plot(r1_k[:halfway], ".-", color=color1, label="$R_1$"+coslabel, lw=2,alpha=0.8,marker=thismarker,ms=10)
        plt.plot(list(reversed(r1_k[halfway:])), ".--", color=color1, lw=2,alpha=0.8,marker=thismarker,ms=10)
        #plt.figure(start_figures_from+1)
        plt.plot(r2_k[:halfway], ".-", color=color2, label="$R_2$"+coslabel, lw=2,alpha=0.8,marker=thismarker,ms=10)
        plt.plot(list(reversed(r2_k[halfway:])), ".--", color=color2, lw=2,alpha=0.8,marker=thismarker,ms=10)
        if show_global:
            plt.figure(start_figures_from)
            plt.plot(r1_g[:halfway], ".-", color=color3, label="$R_1^{\\rm global}$"+coslabel, lw=2,alpha=0.8,marker=thismarker,ms=10)
            plt.plot(list(reversed(r1_g[halfway:])), ".--", color=color3, lw=2,alpha=0.8,marker=thismarker,ms=10)
            plt.figure(start_figures_from+1)
            plt.plot(r2_g[:halfway], ".-", color=color4, label="$R_2^{\\rm global}$"+coslabel, lw=2,alpha=0.8,marker=thismarker,ms=10)
            plt.plot(list(reversed(r2_g[halfway:])), ".--", color=color4, lw=2,alpha=0.8,marker=thismarker,ms=10)
    else:
        if show_global:
            color1,color2,color3,color4 = color_dict[thiskey]
        else:
            color1,color2 = color_dict[thiskey]
        plt.figure(start_figures_from)
        plt.plot(r1_k[:halfway], ".-", color=color1, lw=2,alpha=0.8,marker=thismarker,ms=10)
        plt.plot(list(reversed(r1_k[halfway:])), ".--", color=color1, lw=2,alpha=0.8,marker=thismarker,ms=10)
        plt.figure(start_figures_from+1)
        plt.plot(r2_k[:halfway], ".-", color=color2,  lw=2,alpha=0.8,marker=thismarker,ms=10)
        plt.plot(list(reversed(r2_k[halfway:])), ".--", color=color2, lw=2,alpha=0.8,marker=thismarker,ms=10)

        plt.figure(start_figures_from)
        plt.plot(r1_g[:halfway], ".-", color=color3, lw=2,alpha=0.8,marker=thismarker,ms=10)
        plt.plot(list(reversed(r1_g[halfway:])), ".--", color=color3, lw=2,alpha=0.8,marker=thismarker,ms=10)
        plt.figure(start_figures_from+1)
        plt.plot(r2_g[:halfway], ".-", color=color4, lw=2,alpha=0.8,marker=thismarker,ms=10)
        plt.plot(list(reversed(r2_g[halfway:])), ".--", color=color4, lw=2,alpha=0.8,marker=thismarker,ms=10)


plt.figure(start_figures_from)
marker_count=0
for idx in range(0,halfway+1,halfway//guide_dot_count):
    plt.scatter(idx,-0.05, c=this_cm(idx/halfway),s=60,zorder=21,marker=marker_sequence[marker_count%len(marker_sequence)], clip_on=False)
    marker_count+=1
plt.axis("tight")
plt.ylim(ymin=0)
plt.xticks([])
plt.xlabel("")
plt.ylabel("$R$")
plt.legend(loc="best")
plt.tight_layout()

plt.figure(start_figures_from+1)
marker_count=0
for idx in range(0,halfway+1,halfway//guide_dot_count):
    plt.scatter(idx,-0.05, c=this_cm(idx/halfway),s=60,zorder=21,marker=marker_sequence[marker_count%len(marker_sequence)], clip_on=False)
    marker_count+=1
plt.axis("tight")
plt.ylim(ymin=0)
plt.xticks([])
plt.xlabel("")
plt.ylabel("$R$")
plt.legend(loc="best")
plt.tight_layout()



plt.figure()
for key in data:
    labelkey = key if len(data) > 1 else ""
    plt.plot( (np.array(data[key]["r_g"][0]) - data[key]["r_k"][0]), label = labelkey + " $ r_g - r_k (1)$")
    plt.plot( (np.array(data[key]["r_g"][1]) - data[key]["r_k"][1]), label = labelkey + " $ r_g - r_k (2)$")
plt.legend(loc="best")
plt.xlabel("step")
plt.ylabel("absolute deviation")
plt.tight_layout()

plt.figure()
for key in data:
    plt.plot( (np.array(data[key]["r_g"][0]) - data[key]["r_k"][0])/data[key]["r_k"][0], label = labelkey + "$ (r_g - r_k)/r_k (1)$")
    plt.plot( (np.array(data[key]["r_g"][1]) - data[key]["r_k"][1])/data[key]["r_k"][0], label = labelkey + "$ (r_g - r_k)/r_k (2)$")
plt.legend(loc="best")
plt.xlabel("step")
plt.ylabel("relative deviation")
plt.tight_layout()







plt.show()

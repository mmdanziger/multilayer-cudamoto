import json
from os import stat
from os.path import basename, join
from sys import argv

import numpy as np
from matplotlib import cm
from matplotlib import colors as mplcolors
from matplotlib import pyplot as plt

plt.ioff()


def fname2metadata(fname):
    d = {}
    for s in basename(fname).rstrip(".json").split("_")[1:]:
        i = 0
        key = ""
        try:
            while not s[i].isnumeric():
                key += s[i]
                i += 1
            d[key] = s[i:]
        except IndexError:
            continue
    return d


def first_nonzero(sols, idx=0, eps=1e-10):
    for sol in sorted(sols, key=lambda x: (x[idx], x[(idx + 1) % 2])):
        if sol[idx] > eps:
            return sol[idx]
    return np.nan


def max_solution(sols, idx=0, eps=1e-10):
    for sol in sorted(sols, key=lambda x: (x[idx], x[(idx + 1) % 2]), reversed=True):
        if sol[idx] > eps:
            return sol[idx]
    return np.nan


def nearest_saddle_point(sols):
    if len([i for i in sols if i[1]]) == 1:
        return np.nan
    if sols[0][1] == 0 and sum([i[1] for i in sols]) == 1:
        return 0
    try:
        low_fp = sorted([sol for sol, stab in sols if stab], key=lambda x: np.sqrt(np.dot(x, x)))[0]
        high_fp = sorted([sol for sol, stab in sols if stab], key=lambda x: np.sqrt(np.dot(x, x)))[-1]
    except IndexError:
        msg = "No stable points found"
        raise ValueError(msg)

    try:
        nearest_saddle_to_low = \
            sorted([sol for sol, stab in sols[1:] if not stab], key=lambda x: np_sorting_dist(x, low_fp))[0]
        nearest_saddle_to_high = \
            sorted([sol for sol, stab in sols[1:] if not stab], key=lambda x: np_sorting_dist(x, high_fp))[0]
    except IndexError:
        msg = "No saddle points found"
        raise ValueError(msg)

    return min(np_jump_dist(nearest_saddle_to_low, low_fp), np_jump_dist(nearest_saddle_to_high, high_fp))

    # if not saddles_dist:


def zero_stable(sols):
    if sols[0][1] == 1:
        return 1
    else:
        return np.nan


def both_stable(sols, rthresh1=0.01, rthresh2=0.51):
    for sol, stab in sols:
        if stab and sol[0] > rthresh1 and sol[1] > rthresh1:
            if sol[0] > rthresh2 and not sol[1] > rthresh2:
                continue
            if sol[1] > rthresh2 and not sol[0] > rthresh2:
                continue
            return 1
    return np.nan


def only_one_stable(sols, idx=0, rthresh1=1e-23, rthresh2=0.51):
    for sol, stab in sols:
        if stab and sol[idx] > rthresh1 and sol[(idx + 1) % 2] < rthresh1:
            return 1
    return np.nan


def a_over_b_stable(sols, idx=0, rthresh1=0.001, rthresh2=0.51):
    for sol, stab in sols:
        if stab and sol[idx] > rthresh2 and rthresh1 < sol[(idx + 1) % 2] < rthresh2:
            #if sol[idx] > rthresh1 and rthresh1 < sol[(idx + 1) % 2] < sol[idx]:
            return 1
    return np.nan


def npnorm(vec):
    return np.sqrt(np.dot(vec, vec))


def np_sorting_dist(vec1, vec2):
    d = np.array(vec1) - np.array(vec2)
    return np.sqrt(np.dot(d, d ))


def np_jump_dist(vec1, vec2):
    d = np.array(vec1) - np.array(vec2)
    return min(np.abs(d))


fname = argv[1]
figwidth,figheight = [5.3,5]
savestacked = False
interaction_idx = 0
if ("x2" in basename(fname)) or ("competitive" in basename(fname)):
    interaction_idx = 1
elif ("x3" in basename(fname)) or ("hybrid" in basename(fname)):
    interaction_idx = 2
elif ("x4" in basename(fname)) or ("mixed" in basename(fname)):
    interaction_idx = 3
elif ("x5" in basename(fname)) or ("halfhalf" in basename(fname)):
    interaction_idx = 4

interaction_type = ["interdependent", "competitive", "hybrid", "mixed", "halfhalf"][interaction_idx]

theory_only = "ScanLambda" not in basename(fname)
metadata = fname2metadata(fname)

if basename(fname)[:3] == "sis":
    param1_string = r"$\beta_1$"
    param2_string = r"$\beta_2$"
else:
    param1_string = r"$\lambda_1$"
    param2_string = r"$\lambda_2$"

if "k" not in metadata:
    metadata["k"] = metadata["gamma"]  # horribly patchy!!!

if not theory_only:
    d = json.load(open(fname))

    if interaction_type == "competitive":
        fig1 = plt.figure(figsize=(9, 9))
    else:
        fig1 = plt.figure(figsize=(7, 9))

    result_type_list = ["r_00", "r_10", "r_11"] if interaction_type == "competitive" else ["r_00", "r_11"]
    subplot_rows = 3
    subplot_cols = 3 if interaction_type == "competitive" else 2
    plot_id = 0
    for result_type in result_type_list:
        for netid_to_show in [0, 1]:
            x, y, z = zip(*zip((x["lambda"][0] for x in d), (x["lambda"][1] for x in d),
                               (x[result_type][netid_to_show] for x in d)))
            xlist = sorted(set(x))
            ylist = sorted(set(y))
            zarr = np.outer(xlist, ylist) * np.nan
            for x_, y_, z_ in zip(x, y, z):
                zarr[xlist.index(x_), ylist.index(y_)] = z_

            ma_zarr = np.ma.masked_invalid(zarr)
            plot_id += 1
            ax = fig1.add_subplot(subplot_rows, subplot_cols, plot_id)

            cf = ax.contourf(xlist, ylist, ma_zarr.T, 25, vmin=0, vmax=1, cmap=cm.viridis)
            ax.set_xlabel(param1_string)
            ax.set_ylabel(param2_string)
            fig1.colorbar(cf)
            fig1.tight_layout()
        if interaction_type == "competitive":
            x, y, z = zip(*zip((x["lambda"][0] for x in d), (x["lambda"][1] for x in d),
                               (min(x[result_type][0], x[result_type][1]) for x in d)))
            xlist = sorted(set(x))
            ylist = sorted(set(y))
            zarr = np.outer(xlist, ylist) * np.nan
            for x_, y_, z_ in zip(x, y, z):
                zarr[xlist.index(x_), ylist.index(y_)] = z_

            ma_zarr = np.ma.masked_invalid(zarr)
            plot_id += 1
            ax = fig1.add_subplot(subplot_rows, subplot_cols, plot_id)

            cf = ax.contourf(xlist, ylist, ma_zarr.T, 25, vmin=0, vmax=1, cmap=cm.viridis)
            ax.set_xlabel(param1_string)
            ax.set_ylabel(param2_string)
            fig1.colorbar(cf)
            fig1.tight_layout()

    if interaction_type != "competitive":
        for netid_to_show in [0, 1]:
            x, y, z = zip(*zip((x["lambda"][0] for x in d), (x["lambda"][1] for x in d),
                               (x["r_11"][netid_to_show] - x["r_00"][netid_to_show] for x in d)))
            xlist = sorted(set(x))
            ylist = sorted(set(y))
            zarr = np.outer(xlist, ylist) * np.nan
            for x_, y_, z_ in zip(x, y, z):
                zarr[xlist.index(x_), ylist.index(y_)] = z_

            ma_zarr = np.ma.masked_invalid(zarr)
            plot_id += 1
            ax = fig1.add_subplot(subplot_rows, subplot_cols, plot_id)

            cf = ax.contourf(xlist, ylist, ma_zarr.T, 25, vmin=0, vmax=1, cmap=cm.viridis)
            ax.set_xlabel(param1_string)
            ax.set_ylabel(param2_string)
            fig1.colorbar(cf)
            fig1.tight_layout()

    plt.tight_layout()
    plt.savefig(basename(fname).rstrip("json") + "pdf")

    theory_fname = "two_net_result_%s_k%i_f%.4f.json" % (interaction_type, int(metadata["k"]), float(metadata["f"]))

    basepath = "/storage/home/micha/cudamoto_results/2/theory"
    try:
        stat(join(basepath, theory_fname))
    except:
        basepath = "/home/micha/phd/oscillators"

if theory_only:
    theory_fname = fname
    basepath = "."
d2 = json.load(open(join(basepath, theory_fname)))
lamvec = sorted(d2.keys(), key=float)
lamvec2 = sorted(next(iter(d2.values())).keys(), key=float)
R1_1 = np.zeros((len(lamvec), len(lamvec2)))
R1_2 = np.zeros((len(lamvec), len(lamvec2)))
R2_1 = np.zeros((len(lamvec), len(lamvec2)))
R2_2 = np.zeros((len(lamvec), len(lamvec2)))

minR1R2_1 = np.zeros((len(lamvec), len(lamvec2)))
nsols = np.zeros((len(lamvec), len(lamvec2)))
nstablesols = np.zeros((len(lamvec), len(lamvec2)))
nstablenonzerosols = np.zeros((len(lamvec), len(lamvec2)))

wherezero = np.zeros((len(lamvec), len(lamvec2)))
whereboth = np.zeros((len(lamvec), len(lamvec2)))
where1only = np.zeros((len(lamvec), len(lamvec2)))
where2only = np.zeros((len(lamvec), len(lamvec2)))
where1over2 = np.zeros((len(lamvec), len(lamvec2)))
where2over1 = np.zeros((len(lamvec), len(lamvec2)))

rthresh2 = .001#   if interaction_type == "hybrid" else min(float(metadata["f"]) + 0.01,0.51)
for l1 in lamvec:
    l1ind = lamvec.index(l1)
    for l2 in lamvec2:
        l2ind = lamvec2.index(l2)
        sols = d2[l1][l2]
        nsols[l1ind, l2ind] = len(sols)
        nstablesols[l1ind, l2ind] = sum([i[1] for i in sols])
        nstablenonzerosols[l1ind, l2ind] = sum([i[1] for idx, i in enumerate(sols) if idx > 0])
        wherezero[l1ind, l2ind] = zero_stable(sols)
        whereboth[l1ind, l2ind] = both_stable(sols, rthresh1=rthresh2,rthresh2=rthresh2)
        where1only[l1ind, l2ind] = only_one_stable(sols, 0)
        where2only[l1ind, l2ind] = only_one_stable(sols, 1)
        where1over2[l1ind, l2ind] = a_over_b_stable(sols, 0, rthresh2=rthresh2)
        where2over1[l1ind, l2ind] = a_over_b_stable(sols, 1, rthresh2=rthresh2)

        try:
            R1_1[l1ind, l2ind] = nearest_saddle_point(sols)
        except ValueError as ve:
            R1_1[l1ind, l2ind] = np.nan
            print(f"({l1} , {l2}) : {ve}")

            # if len(sols) > 1:

i = 0
if not True:
    fig2 = plt.figure()
    subplot_rows2 = 1
    subplot_cols2 = 2
    dr = 0.37

    for subplot_idx, arr in enumerate([R1_1]):
        ax = fig2.add_subplot(subplot_rows2, subplot_cols2, subplot_idx + 1)
        cf = ax.contourf(lamvec, lamvec2, np.ma.masked_invalid(arr.T), 25, vmin=0, vmax=1, cmap=cm.viridis)
        ax.set_xlabel(param1_string)
        ax.set_ylabel(param2_string)
        fig2.colorbar(cf)
    cmap = mplcolors.ListedColormap(["red", "blue", "yellow"])
    bounds = [.5, 1.5, 2.5, 3.5]
    norm = mplcolors.BoundaryNorm(bounds, cmap.N)

    ax = fig2.add_subplot(subplot_rows2, subplot_cols2, subplot_idx + 2)
    for arr, _this_hatch, this_color in zip([wherezero, whereboth, where1only, where2only], ["o", "/", "-", "|"],
                                           [1, 2, 3, 3]):
        nsolspc = plt.contourf(np.array(lamvec).astype(float), np.array(lamvec2).astype(float),
                               this_color * np.ma.masked_invalid(arr.T),
                               zorder=i, cmap=cmap, alpha=0.6, norm=norm)
        i += 1
else:
    fig2 = plt.figure(figsize=(figwidth,figheight))
    subplot_rows2 = 1
    subplot_cols2 = 1
    cmap = mplcolors.ListedColormap(["red", "blue", "yellow", "darkviolet", "darkgreen", "cyan"])
    bounds = [.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    norm = mplcolors.BoundaryNorm(bounds, cmap.N)
    name_array = ["wherezero", "where1only", "where2only", "where1over2", "where2over1", "whereboth"]
    ax = fig2.add_subplot(subplot_rows2, subplot_cols2, 1)
    for arr, _this_hatch, this_color in zip([wherezero, where1only, where2only, where1over2, where2over1, whereboth],
                                           ["o", "-", "|", "|", "|", "/"], [1, 2, 3, 4, 5, 6]):
        nsolspc = plt.contourf(np.array(lamvec).astype(float), np.array(lamvec2).astype(float),
                               this_color * np.ma.masked_invalid(arr.T),
                               zorder=i, cmap=cmap, alpha=0.6, norm=norm)
        plt.xlabel(param1_string)
        plt.ylabel(param2_string)
        fig2.set_size_inches(figwidth,figheight)
        fig2.tight_layout()
        if savestacked:
            plt.savefig("ScanLambdaTheoryStacked_k%i_f%.2f_%s_%s.pdf" % (
                int(metadata["k"]), float(metadata["f"]), interaction_type, name_array[i]), bbox_inches="tight")
        i += 1
        if not theory_only:
            xmax = float(min([max(i, key=float) for i in [lamvec, xlist, ylist, lamvec2]], key=float))
            plt.xlim(xmax=xmax)
            plt.ylim(ymax=xmax)
plt.xlabel(param1_string)
plt.ylabel(param2_string)
plt.tight_layout()
if not theory_only:
    xmax = float(min([max(i, key=float) for i in [lamvec, xlist, ylist, lamvec2]], key=float))
    plt.xlim(xmax=xmax)
    plt.ylim(ymax=xmax)
plt.savefig("ScanLambdaTheory_k%i_f%.2f_%s.pdf" % (int(metadata["k"]), float(metadata["f"]), interaction_type),
            bbox_inches="tight")

fig3 = plt.figure(figsize=(figwidth,figheight))
subplot_rows2 = 1
subplot_cols2 = 1
cmap = mplcolors.ListedColormap(["red", "blue", "yellow", "darkviolet", "darkgreen", "cyan"])
bounds = [.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
norm = mplcolors.BoundaryNorm(bounds, cmap.N)

ax3 = fig3.add_subplot(1, 1, 1)
i = 0

for arr, _this_hatch, this_color in zip([wherezero, where1only, where2only, where1over2, where2over1, whereboth],
                                       ["o", "-", "|", "|", "|", "/"], [1, 2, 3, 4, 5, 6]):
    nsolspc = plt.contourf(np.array(lamvec).astype(float), np.array(lamvec2).astype(float),
                           this_color * np.ma.masked_invalid(arr.T),
                           zorder=i, cmap=cmap, alpha=0.6, norm=norm)

    plt.xlabel(param1_string)
    plt.ylabel(param2_string)
    fig3.set_size_inches(figwidth,figheight)
    fig3.tight_layout()
    plt.savefig("ScanLambdaTheoryIndividual_k%i_f%.2f_%s_%s.pdf" % (
        int(metadata["k"]), float(metadata["f"]), interaction_type, name_array[i]), bbox_inches="tight")
    plt.cla()
    i += 1

fig3 = plt.figure(figsize=(figwidth,figheight))
subplot_rows2 = 1
subplot_cols2 = 1
cmap = cm.viridis  # mplcolors.ListedColormap(["red", "blue", "yellow", "darkviolet", "darkgreen", "cyan"])
bounds = [i - 0.5 for i in sorted(set(nstablesols.flatten()))]
bounds.append(bounds[-1] + 0.5)
norm = mplcolors.BoundaryNorm(bounds, cmap.N)

ax3 = fig3.add_subplot(1, 1, 1)
i = 0
nsolspc = plt.contourf(np.array(lamvec).astype(float), np.array(lamvec2).astype(float),
                       np.ma.masked_invalid(nstablesols.T),
                       cmap=cmap, alpha=0.6, norm=norm)
fig3.colorbar(nsolspc, ticks=np.arange(np.min(nstablesols), np.max(nstablesols) + 1))

plt.xlabel(param1_string)
plt.ylabel(param2_string)
fig3.set_size_inches(figwidth,figheight)
fig3.tight_layout()
plt.savefig(
    "ScanLambdaTheoryMultistability_k%i_f%.2f_%s.pdf" % (int(metadata["k"]), float(metadata["f"]), interaction_type),
    bbox_inches="tight")


fig4 = plt.figure(figsize=(figwidth,figheight))
subplot_rows2 = 1
subplot_cols2 = 1
cmap = cm.viridis  # mplcolors.ListedColormap(["red", "blue", "yellow", "darkviolet", "darkgreen", "cyan"])
bounds = [i - 0.5 for i in sorted(set(nsols.flatten()))]
bounds.append(bounds[-1] + 0.5)
norm = mplcolors.BoundaryNorm(bounds, cmap.N)

ax4 = fig4.add_subplot(1, 1, 1)
i = 0
nsolspc = plt.contourf(np.array(lamvec).astype(float), np.array(lamvec2).astype(float),
                       np.ma.masked_invalid(nsols.T),
                       cmap=cmap, alpha=0.6, norm=norm)
fig4.colorbar(nsolspc, ticks=np.arange(np.min(nsols), np.max(nsols) + 1))

plt.xlabel(param1_string)
plt.ylabel(param2_string)
fig4.set_size_inches(figwidth,figheight)
fig4.tight_layout()
plt.savefig(
    "ScanLambdaTheoryFixedPointCount_k%i_f%.2f_%s.pdf" % (int(metadata["k"]), float(metadata["f"]), interaction_type),
    bbox_inches="tight")

if not theory_only:
    netid_to_show = 0
    xmax = float(min([max(i, key=float) for i in [lamvec, xlist, ylist, lamvec2]], key=float))
    fig1tag = plt.figure(figsize=(6, 5))
    dr = 0.18
    x, y, z = zip(*zip((x["lambda"][0] for x in d), (x["lambda"][1] for x in d),
                       (x["r_11"][netid_to_show] - x["r_00"][netid_to_show] for x in d)))
    xlist = sorted(set(x))
    ylist = sorted(set(y))
    zarr = np.outer(xlist, ylist) * np.nan
    for x_, y_, z_ in zip(x, y, z):
        zarr[xlist.index(x_), ylist.index(y_)] = z_

    ma_zarr = np.ma.masked_invalid(zarr)
    ax = fig1tag.add_subplot(1, 1, 1)

    cf = ax.contourf(xlist, ylist, ma_zarr.T, 25, vmin=0, vmax=1, cmap=cm.viridis)
    CS = plt.contour(np.array(lamvec).astype(float), np.array(lamvec2).astype(float),
                     R1_1.T, levels=[dr], linewidths=3, colors="orange")

    ax.set_xlabel(param1_string)
    ax.set_ylabel(param2_string)
    fig1tag.colorbar(cf)
    fig1tag.tight_layout()
    plt.xlim(xmax=xmax)
    plt.ylim(ymax=xmax)
    plt.savefig("ScanLambdaMetastable_k%i_f%.2f_dr%.2f_%s.pdf" % (
    int(metadata["k"]), float(metadata["f"]), dr, interaction_type),
                bbox_inches="tight")
    fig2.savefig("ScanLambdaTheory_k%i_f%.2f_%s.pdf" % (int(metadata["k"]), float(metadata["f"]), interaction_type),
            bbox_inches="tight")


# nsolspc = plt.pcolormesh(np.array(lamvec).astype(float), np.array(lamvec2).astype(float), nsols.T, cmap=cm.viridis, norm=norm,
#            vmin=.5, vmax=np.max(nsols) + .5)
#
# nstablesolspc = plt.pcolormesh(np.array(lamvec).astype(float), np.array(lamvec2).astype(float), nstablesols.T, cmap=cm.viridis, norm=norm,
#            vmin=.5, vmax=np.max(nstablesols) + .5)
#
# nstablenonzerosolspc = plt.pcolormesh(np.array(lamvec).astype(float), np.array(lamvec2).astype(float), nstablenonzerosols.T, cmap=cm.viridis, norm=norm,
#            vmin=.5, vmax=np.max(nstablenonzerosols) + .5)
#
#


#
# plt.pcolormesh(np.array(lamvec).astype(float), np.array(lamvec2).astype(float), nsols.T, cmap=cm.viridis, norm=norm,
#            vmin=.5, vmax=np.max(nsols) + .5)

#
#
# for l1 in lamvec:
#     for l2 in lamvec2:
#         if len(d2[l1][l2]["r1"]) > 1 and len(d2[l1][l2]["r2"]) > 1:
#
# for subplot_idx,arr in enumerate([R1_1, R2_1,  minR1R2_1, R1_2,R2_2]):
#
plt.ion()

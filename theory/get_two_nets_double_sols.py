import sync.two_nets_sync as tns
from sys import argv
from collections import defaultdict
import json
from glob import glob
import os
from os.path import basename

"""
 Example: get_two_nets_double_sols.py master 12 0.5 0 0.001 1 0.3

Script to generate solutions for a given value of f.
It can run as either the master or the slave.
The master has no args, and launches slaves with args indicating which values to cover.
This way multiprocessing can be used cleanly and efficiently.
After all of the slaves write their output to /tmp, the master compiles all the outputs into one json.
"""

def num(string_input):
    try:
        return int(string_input)
    except:
        return float(string_input)
force_fname = False
mode = argv[1]
try:
    k = num(argv[2])#or gamma if sf
except:
    integral_fname = argv[2]
    force_fname = True
    k = [num(i[1:]) for i in basename(integral_fname).split("_") if i[0]=="k"][0]
frac = float(argv[3])
interaction_idx = int(argv[4])
lambda_i = float(argv[5])
lambda_f = float(argv[6])
lambda_step = float(argv[7])
degree_dist = "er"
q = float(os.environ["OSCILLATORS_Q"]) if "OSCILLATORS_Q" in os.environ else 0

interaction_type=["interdependent","competitive", "hybrid","mixed","halfhalf"][interaction_idx]

if mode == "slave":
    """
    Slave
    """
    lambda1_i = lambda_i
    lambda1_f = lambda_f
    lambda2_i = float(argv[8])
    lambda2_f = float(argv[9])
    res = defaultdict(dict)
    if force_fname:
        integral_fname = integral_fname
    elif degree_dist == "er":
        integral_fname = "interpolation_values_combined_1d_k%i_HD.json" % k
    elif degree_dist == "sf":
        k_string = "%i"%k if type(k) is int else "%.1f"%k
        integral_fname = "interpolation_values_combined_1d_gamma%s_HD.json" % k_string

    intfunc = tns.sns.get_interpolated_integral(kbar=k, mode="r",fname=integral_fname,gamma=k)
    for lambda1 in tns.np.arange(lambda1_i, lambda1_f, step=lambda_step):
        for lambda2 in tns.np.arange(lambda2_i, lambda2_f, step=lambda_step):
            sols = tns.solve_R1R2_double_solutions(intfunc, [lambda1, lambda2], frac,interaction_type=interaction_type,q=q)
            res["%.8f" % lambda1]["%.8f" % lambda2] = sols
    json.dump(res,
              open("/tmp/two_net_result_%s_k%i_f%.4f_lambdai%.8f_lambdaf%.8f.json" % (interaction_type, k, frac, lambda1_i, lambda1_f), "w"))
else:
    """
    Master
    """
    from subprocess import call

    cores = int(argv[8]) if len(argv) > 8 else None


    def run(li, lf):
        global force_fname
        k_arg = integral_fname if force_fname else str(k)
        call(["python", argv[0], "slave", k_arg, "%.4f"%frac, str(interaction_idx), str(li), str(lf),str(lambda_step),str(lambda_i),str(lambda_f)])
        print("complete.")


    import multiprocessing

    processes = multiprocessing.cpu_count() if not cores else cores
    jobs = []
    for i in range(processes):
        li = lambda_i + i * (lambda_f - lambda_i) / processes
        lf = lambda_i + (i + 1) * (lambda_f - lambda_i) / processes
        p = multiprocessing.Process(target=run, args=(li, lf))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()
    """
    Collect and save data from slaves' outputs.
    """
    d = {}
    for fname in glob("/tmp/two_net_result_%s_k%i_f%.4f_*.json" % (interaction_type,k, frac)):
        subd = json.load(open(fname))
        for key,v in subd.items():
            d[key] = v
        call(["rm" , fname])
    if degree_dist == "er":
        json.dump(d,open("two_net_result_%s_k%i_f%.4f_q%.4f.json"%(interaction_type,k,frac,q),"w"))
    else:
        json.dump(d, open("two_net_result_%s_gamma%i_f%.4f_q%.4f.json" % (interaction_type, k, frac,q), "w"))

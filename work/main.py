import os, sys, numpy, scipy
from pyscf.lib import chkfile

from util import solve_bs_noci

if __name__ == "__main__":
    m        = "h2"
    basis    = "sto-3g"
    dir_path = f"./data/{m}/{basis}"
    h5_path  = f"./data/{m}/{basis}/data.h5"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    if os.path.exists(h5_path):
        os.remove(h5_path)

    for x in numpy.linspace(0.8, 2.4, 41):
        data_dict = solve_bs_noci(x, basis=basis, m=m)
        chkfile.save(h5_path, "%12.8f" % x, data_dict)

    m        = "h4-line"
    basis    = "sto-3g"
    dir_path = f"./data/{m}/{basis}"
    h5_path  = f"./data/{m}/{basis}/data.h5"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    if os.path.exists(h5_path):
        os.remove(h5_path)

    for x in numpy.linspace(0.8, 2.4, 41):
        data_dict = solve_bs_noci(x, basis=basis, m=m)
        chkfile.save(h5_path, "%12.8f" % x, data_dict)

    m        = "h4-square"
    basis    = "sto-3g"
    dir_path = f"./data/{m}/{basis}"
    h5_path  = f"./data/{m}/{basis}/data.h5"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    if os.path.exists(h5_path):
        os.remove(h5_path)

    for x in numpy.linspace(0.8, 2.4, 41):
        data_dict = solve_bs_noci(x, basis=basis, m=m)
        chkfile.save(h5_path, "%12.8f" % x, data_dict)

    m        = "n2"
    basis    = "sto-3g"
    dir_path = f"./data/{m}/{basis}"
    h5_path  = f"./data/{m}/{basis}/data.h5"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    if os.path.exists(h5_path):
        os.remove(h5_path)

    for x in numpy.linspace(0.8, 2.4, 41):
        data_dict = solve_bs_noci(x, basis=basis, m=m)
        chkfile.save(h5_path, "%12.8f" % x, data_dict)

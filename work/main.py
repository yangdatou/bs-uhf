import os, sys, numpy, scipy
from pyscf.lib import chkfile

from util import solve_bs_noci

if __name__ == "__main__":
    m        = "n2"
    basis    = "sto-3g"
    dir_path = f"/Users/yangjunjie/work/bs-uhf/work/{m}/{basis}"
    h5_path  = f"/Users/yangjunjie/work/bs-uhf/work/{m}/{basis}/data.h5"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    if os.path.exists(h5_path):
        os.remove(h5_path)

    for x in numpy.linspace(0.4, 4.0, 41):
        data_dict = solve_bs_noci(x, basis=basis, m=m)
        chkfile.save(h5_path, "%12.8f" % x, data_dict)

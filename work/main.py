import os, sys, numpy, scipy
from pyscf.lib import chkfile

from util import solve_bs_noci

def main(m, basis, dir_path, h5_path, x_list):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    if os.path.exists(h5_path):
        os.remove(h5_path)

    print("\n")
    print("#" * 20)
    print("m        = %s" % m)
    print("basis    = %s" % basis)
    print("dir_path = %s" % dir_path)
    print("h5_path  = %s" % h5_path)
    for x in x_list:
        data_dict = solve_bs_noci(x, basis=basis, m=m)
        chkfile.save(h5_path, "%12.8f" % x, data_dict)

if __name__ == "__main__":
    # m        = "h2"
    basis    = "sto-3g"
    dir_path = f"./data/"
    # h5_path  = os.path.join(dir_path, f"{m}-{basis}.h5")
    # x_list   = numpy.linspace(0.5, 3.0, 10)
    # main(m, basis, dir_path, h5_path, x_list)

    # m       = "h4-line"
    # h5_path = os.path.join(dir_path, f"{m}-{basis}.h5")
    # x_list  = numpy.arange(0.5, 3.0, 0.1)
    # main(m, basis, dir_path, h5_path, x_list)

    # m       = "h4-square"
    # h5_path = os.path.join(dir_path, f"{m}-{basis}.h5")
    # x_list  = numpy.arange(0.5, 3.0, 0.1)
    # main(m, basis, dir_path, h5_path, x_list)

    m       = "n2"
    h5_path = os.path.join(dir_path, f"{m}-{basis}.h5")
    x_list  = numpy.arange(0.5, 3.0, 0.1)
    main(m, basis, dir_path, h5_path, x_list)

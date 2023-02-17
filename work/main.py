import os, sys, numpy, scipy
from pyscf.lib import chkfile

from util import solve_bs_noci

def main(m, basis, h5_path, tmp_dir, x_list, is_scf=False):
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)

    if os.path.exists(h5_path):
        os.remove(h5_path)

    print("\n")
    print("#" * 20)
    print("m        = %s" % m)
    print("basis    = %s" % basis)
    print("tmp_dir  = %s" % tmp_dir)
    print("h5_path  = %s" % h5_path)
    print("is_scf   = %s" % is_scf)

    for x in x_list:
        data_dict = solve_bs_noci(x, tmp_dir=tmp_dir, basis=basis, m=m, is_scf=is_scf)
        chkfile.save(h5_path, "%12.8f" % x, data_dict)

if __name__ == "__main__":
    dir_path = f"./data/"
    # m        = "h2"
    # basis    = "sto-3g"
    # h5_path  = os.path.join(dir_path, f"{m}-{basis}-scf.h5")
    # x_list   = numpy.linspace(0.5, 3.0, 41)
    # main(m, basis, dir_path, h5_path, x_list, is_scf=True)

    # basis    = "cc-pvdz"
    # h5_path  = os.path.join(dir_path, f"{m}-{basis}.h5")
    # x_list   = numpy.linspace(0.5, 3.0, 41)
    # main(m, basis, dir_path, h5_path, x_list)

    # h5_path  = os.path.join(dir_path, f"{m}-{basis}-scf.h5")
    # x_list   = numpy.linspace(0.5, 3.0, 41)
    # main(m, basis, dir_path, h5_path, x_list, is_scf=True)

    # m       = "h4-line"
    # basis    = "sto-3g"
    # h5_path = os.path.join(dir_path, f"{m}-{basis}.h5")
    # x_list  = numpy.linspace(0.5, 3.0, 41)
    # main(m, basis, dir_path, h5_path, x_list)

    # basis    = "cc-pvdz"
    # h5_path  = os.path.join(dir_path, f"{m}-{basis}.h5")
    # x_list   = numpy.linspace(0.5, 3.0, 41)
    # main(m, basis, dir_path, h5_path, x_list)

    # m       = "h4-square"
    # basis   = "sto-3g"
    # h5_path = os.path.join(dir_path, f"{m}-{basis}.h5")
    # x_list  = numpy.linspace(0.5, 3.0, 41)
    # main(m, basis, dir_path, h5_path, x_list)

    # basis    = "cc-pvdz"
    # h5_path  = os.path.join(dir_path, f"{m}-{basis}.h5")
    # x_list   = numpy.linspace(0.5, 3.0, 41)
    # main(m, basis, dir_path, h5_path, x_list)

    m        = "n2"
    basis    = "sto-3g"
    dir_path = f"./data/"
    tmp_dir  = dir_path + f"/{m}-{basis}/"
    h5_path  = os.path.join(dir_path, f"{m}-{basis}.h5")
    
    x_list   = numpy.linspace(0.8, 3.0, 41)
    main(m, basis, h5_path, tmp_dir, x_list, is_scf=False)
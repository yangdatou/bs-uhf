from functools import reduce
from itertools import combinations

import os, sys
from sys import stdout
import numpy, scipy
from scipy.linalg import sqrtm

import pyscf
from pyscf import gto, scf, lo, mp
from pyscf.tools.dump_mat import dump_rec

def localize_mo(mo, mol_obj=None, ovlp_ao=None, method="iao"):
    if method == "iao":
        c = lo.iao.iao(mol_obj, mo)
        c = lo.vec_lowdin(c, ovlp_ao)
        mo = c

    elif method == "boys":
        c = lo.Boys(mol_obj, mo).kernel()
        mo = c

    else:
        raise NotImplementedError
    
    return mo

def solve_he2_mp2(x=1.0, spin=0, basis="ccpvdz"):
    mol = gto.Mole()
    mol.verbose = 0
    # mol.output = './tmp/he2-ump2-%.2f.out' % x
    mol.atom = f"""
        He 0.00000 0.00000 0.00000
        He 0.00000 0.00000 {x:12.8f}
    """ 
    mol.basis = basis
    mol.spin  = spin
    mol.build()

    rhf_obj = scf.RHF(mol)

    rhf_obj.verbose   = 0
    rhf_obj.max_cycle = 0
    rhf_obj.kernel()

    rmp2_obj = mp.mp2.RMP2(rhf_obj)
    rmp2_obj.conv_tol = 1e-12
    rmp2_obj.verbose  = 5
    ene_rmp2_1, t2_rmp2_1 = rmp2_obj.kernel()

    rhf_obj.verbose   = 0
    rhf_obj.max_cycle = 100
    rhf_obj.kernel()
    
    rmp2_obj = mp.mp2.RMP2(rhf_obj)
    rmp2_obj.conv_tol = 1e-12
    rmp2_obj.verbose  = 5
    ene_rmp2_2, t2_rmp2_2 = rmp2_obj.kernel()

    print("RMP2 energy (1): %20.12f" % ene_rmp2_1)
    print("RMP2 energy (2): %20.12f" % ene_rmp2_2)

    ovlp_ao = rhf_obj.get_ovlp()
    coeff_iao = localize_mo(rhf_obj.mo_coeff, mol_obj=mol, ovlp_ao=ovlp_ao, method="iao")

    uhf_obj = scf.UHF(mol)
    uhf_obj.verbose = 4
    uhf_obj.max_cycle = 0
    uhf_obj.kernel()
    uhf_obj.converged = False

    ump2_obj = mp.ump2.UMP2(uhf_obj)
    ump2_obj.verbose = 100
    ump2_obj._scf.converged = False
    ene_ump2, t2_ump2 = ump2_obj.kernel()

    assert 1 == 2

if __name__ == "__main__":
    x = 3.0
    solve_he2_mp2(x=x, spin=0, basis="ccpvtz")
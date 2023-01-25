from functools import reduce
import os, sys, numpy, scipy
from numpy import linalg

from pyscf import gto, scf, fci, mp, ao2mo, ci
from pyscf.tools.dump_mat import dump_rec
from pyscf.fci import cistring, spin_op

from spin_utils import coeff_rhf_to_ghf, coeff_uhf_to_ghf
from spin_utils import rotate_coeff_ghf, get_spin_avg

def solve_h2_bs_uhf(r, basis="sto-3g", f=None):
    mol = gto.Mole()
    mol.atom = f"""
    H1 0.000 0.000 {( r/2.0): 12.8f}
    H2 0.000 0.000 {(-r/2.0): 12.8f}
    """
    mol.basis = basis
    mol.build()

    rhf_obj = scf.RHF(mol)
    rhf_obj.verbose = 0
    rhf_obj.kernel(dm0=None)
    assert rhf_obj.converged

    coeff_rhf = coeff_rhf_to_ghf(rhf_obj.mo_coeff, rhf_obj.mo_energy)

    rdm1_rhf = rhf_obj.make_rdm1()

    dm0 = numpy.zeros((2, mol.nao, mol.nao))
    dm0[0, 0, 0] = 1.0
    dm0[1, 1, 1] = 1.0

    uhf_obj = scf.UHF(mol)
    uhf_obj.verbose = 0
    uhf_obj.kernel(dm0=dm0)
    assert uhf_obj.converged

    coeff_uhf = coeff_uhf_to_ghf(uhf_obj.mo_coeff, uhf_obj.mo_energy)

    ghf_obj = scf.GHF(mol)
    ghf_obj.verbose = 0
    ovlp_ao = ghf_obj.get_ovlp()
    mo_occ = numpy.asarray([1, 1, 0, 0])

    rdm1_uhf = ghf_obj.make_rdm1(coeff_uhf, mo_occ)
    s_uhf    = get_spin_avg(rdm1_ghf=rdm1_uhf, ovlp_ao=ovlp_ao, ao_idx=[0]).real

    for gamma in numpy.linspace(0.0, numpy.pi / 2, 21):
        for beta in numpy.linspace(0.0, numpy.pi, 21):
            l_matrix_y = numpy.array([[0.0,  0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
            l_matrix_z = numpy.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [ 0.0, 0.0, 0.0]])

            rot_matrix_2 = scipy.linalg.expm(beta  * l_matrix_y)
            rot_matrix_3 = scipy.linalg.expm(gamma * l_matrix_z)

            rot_matrix = reduce(numpy.dot, [rot_matrix_2, rot_matrix_3])
            s_uhf_rot = numpy.dot(rot_matrix, s_uhf)
            
            coeff_beta_gamma = rotate_coeff_ghf(coeff_uhf, beta=beta, gamma=gamma)
            rdm1_beta_gamma  = ghf_obj.make_rdm1(coeff_beta_gamma, mo_occ)
            s_beta_gamma     = get_spin_avg(rdm1_ghf=rdm1_beta_gamma, ovlp_ao=ovlp_ao, ao_idx=[0]).real

            ss_uhf_rot    = f"[{s_uhf_rot[0]: 6.4f}, {s_uhf_rot[1]: 6.4f}, {s_uhf_rot[2]: 6.4f}]"
            ss_beta_gamma = f"[{s_beta_gamma[0]: 6.4f}, {s_beta_gamma[1]: 6.4f}, {s_beta_gamma[2]: 6.4f}]"
            print(f"\ngamma = {gamma: 12.8f}, beta = {beta: 12.8f}")
            print(f"ss_uhf_rot    = {ss_uhf_rot}")
            print(f"ss_beta_gamma = {ss_beta_gamma}")
            assert numpy.linalg.norm(s_uhf_rot - s_beta_gamma) < 1e-10


if __name__ == "__main__":
    basis = "sto-3g"
    with open(f"/Users/yangjunjie/work/bs-uhf/data/h2/bs-uhf-{basis}.csv", "w") as f:
        for x in numpy.linspace(0.4, 2.0, 40):
            if x == 2.0:
                solve_h2_bs_uhf(x, basis=basis, f=f)
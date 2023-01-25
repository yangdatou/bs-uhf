from functools import reduce
import os, sys, numpy, scipy
from numpy import linalg

from pyscf import gto, scf, fci, mp, ao2mo, ci
from pyscf.tools.dump_mat import dump_rec
from pyscf.fci import cistring, spin_op

from spin_utils import coeff_rhf_to_ghf, coeff_uhf_to_ghf
from spin_utils import rotate_coeff_ghf, get_spin_avg

def get_fci_ham(ghf_obj, coeff, )

def solve_h4_bs_uhf(r, basis="sto-3g", f=None):
    mol = gto.Mole()
    mol.atom = f"""
    H1 { r/2.0: 12.8f} { r/2.0: 12.8f} 0.00000
    H2 { r/2.0: 12.8f} {-r/2.0: 12.8f} 0.00000
    H3 {-r/2.0: 12.8f} { r/2.0: 12.8f} 0.00000
    H4 {-r/2.0: 12.8f} {-r/2.0: 12.8f} 0.00000
    """
    mol.basis = basis
    mol.build()

    rhf_obj = scf.RHF(mol)
    rhf_obj.verbose = 0
    rhf_obj.kernel(dm0=None)
    assert rhf_obj.converged

    coeff_rhf = coeff_rhf_to_ghf(rhf_obj.mo_coeff, rhf_obj.mo_energy)
    rdm1_rhf  = rhf_obj.make_rdm1()

    dm0 = numpy.zeros((2, mol.nao, mol.nao))
    dm0[0, 0, 0] = 1.0
    dm0[0, 3, 3] = 1.0
    dm0[1, 1, 1] = 1.0
    dm0[1, 2, 2] = 1.0

    uhf_obj = scf.UHF(mol)
    uhf_obj.verbose = 0
    uhf_obj.conv_tol = 1e-12
    uhf_obj.kernel(dm0=dm0)
    assert uhf_obj.converged

    coeff_uhf, mo_occ_uhf = coeff_uhf_to_ghf(uhf_obj.mo_coeff, uhf_obj.mo_energy, uhf_obj.mo_occ)

    from util import show_h4_spin
    from pyscf.tools import cubegen

    ghf_obj = scf.GHF(mol)
    ghf_obj.verbose = 0
    ovlp_ao = ghf_obj.get_ovlp()

    from pyscf.fci import cistring, addons
    norb_alph, norb_beta   = uhf_obj.mo_coeff[0].shape[1], uhf_obj.mo_coeff[1].shape[1]
    nelec_alph, nelec_beta = uhf_obj.nelec[0], uhf_obj.nelec[1]
    ndet     = cistring.num_strings(norb_alph + norb_beta, nelec_alph + nelec_beta)
    vfci_rhf = numpy.zeros((ndet, 1))

    vfci_list = []
    
    rdm1_uhf = ghf_obj.make_rdm1(coeff_uhf, mo_occ_uhf)
    s_uhf    = get_spin_avg(rdm1_ghf=rdm1_uhf, ovlp_ao=ovlp_ao, ao_idx=[0]).real
    vfci_uhf = numpy.zeros((ndet, 1))
    vfci_uhf[0, 0] = 1
    u = reduce(numpy.dot, [vfci_uhf.T, ovlp_ao, coeff_rhf])
    vfci_uhf = addons.transform_ci_for_orbital_rotation(vfci_uhf, norb_alph + norb_beta, (nelec_alph + nelec_beta, 0), u)

    for beta in numpy.linspace(0.0, numpy.pi, 21):
        # l_matrix_y = numpy.array([[0.0,  0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        # l_matrix_z = numpy.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [ 0.0, 0.0, 0.0]])

        # rot_matrix_2 = scipy.linalg.expm(beta  * l_matrix_y)
        # rot_matrix_3 = scipy.linalg.expm(gamma * l_matrix_z)

        # rot_matrix = reduce(numpy.dot, [rot_matrix_2, rot_matrix_3])
        # s_uhf_rot = numpy.dot(rot_matrix, s_uhf)
        
        coeff_beta = rotate_coeff_ghf(coeff_uhf, beta=beta, gamma=gamma)
        rdm1_beta  = ghf_obj.make_rdm1(coeff_beta, mo_occ_uhf)

        err = ghf_obj.get_grad(coeff_beta_gamma, mo_occ_uhf)
        err = numpy.linalg.norm(err)

        u = reduce(numpy.dot, [coeff_beta.conj().T, ovlp_ao, coeff_rhf])
        vfci_beta = numpy.zeros((ndet, 1))
        vfci_beta[0, 0] = 1
        vfci_beta = addons.transform_ci_for_orbital_rotation(vfci_beta, norb_alph + norb_beta, (nelec_alph + nelec_beta, 0), u)

        print(vfci_beta)



if __name__ == "__main__":
    basis = "sto-3g"
    with open(f"/Users/yangjunjie/work/bs-uhf/data/h2/bs-uhf-{basis}.csv", "w") as f:
        for x in numpy.linspace(0.4, 2.0, 40):
            if x == 2.0:
                solve_h4_bs_uhf(x, basis=basis, f=f)
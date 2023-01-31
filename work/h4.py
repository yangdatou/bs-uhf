from functools import reduce
import os, sys, numpy, scipy
from sys import stdout
from numpy import linalg

from pyscf import gto, scf, fci, mp, ao2mo, ci
from pyscf.fci.direct_spin1 import absorb_h1e, contract_2e
from pyscf.tools.dump_mat import dump_rec
from pyscf.fci import cistring, spin_op

from spin_utils import coeff_rhf_to_ghf, coeff_uhf_to_ghf
from spin_utils import rotate_coeff_ghf, get_spin_avg

def get_ghf_fci_ham(ghf_obj, coeff_ghf, nelec):
    hcore = ghf_obj.get_hcore()
    nao   = hcore.shape[0] // 2
    norb  = coeff_ghf.shape[1]

    assert hcore.shape == (nao * 2, nao * 2)
    assert coeff_ghf.shape == (nao * 2, norb)

    h1e = reduce(numpy.dot, (coeff_ghf.T, ghf_obj.get_hcore(), coeff_ghf))

    coeff_alph = coeff_ghf[:nao, :]
    coeff_uhf_beta = coeff_ghf[nao:, :]
    h2e_aabb = ao2mo.kernel(ghf_obj._eri, (coeff_alph, coeff_alph, coeff_uhf_beta, coeff_uhf_beta))

    h2e  = ao2mo.kernel(ghf_obj._eri, coeff_alph)
    h2e += ao2mo.kernel(ghf_obj._eri, coeff_uhf_beta)
    h2e += h2e_aabb + h2e_aabb.T

    return h1e, h2e, absorb_h1e(h1e, h2e, norb, nelec, 0.5)

def solve_h4_bs_uhf(r, basis="sto-3g", f=None):
    atoms = ""
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
    rhf_obj.conv_tol = 1e-12
    ovlp_ao = rhf_obj.get_ovlp()
    hcore   = rhf_obj.get_hcore()
    mo_energy, mo_coeff = rhf_obj.eig(hcore, ovlp_ao)
    coeff_rhf = coeff_rhf_to_ghf(mo_coeff, mo_energy)
    ene_rhf   = rhf_obj.energy_elec()[0]

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
    ene_uhf = uhf_obj.energy_elec()[0]
    dm_uhf = uhf_obj.make_rdm1()
    coeff_uhf, mo_occ_uhf = coeff_uhf_to_ghf(uhf_obj.mo_coeff, uhf_obj.mo_energy, uhf_obj.mo_occ)

    from pyscf.fci import cistring, addons
    norb_alph, norb_beta   = uhf_obj.mo_coeff[0].shape[1], uhf_obj.mo_coeff[1].shape[1]
    nelec_alph, nelec_beta = uhf_obj.nelec[0], uhf_obj.nelec[1]
    ndet     = cistring.num_strings(norb_alph + norb_beta, nelec_alph + nelec_beta)
    vfci_rhf = numpy.zeros((ndet, 1))
    vfci_rhf[0, 0] = 1

    vfci_uhf_list = []
    vfci_mp2_list = []

    ghf_obj = scf.GHF(mol)
    ghf_obj.verbose = 0
    rdm1_uhf = ghf_obj.make_rdm1(coeff_uhf, mo_occ_uhf)
    ghf_obj.kernel(dm0=rdm1_uhf)
    assert ghf_obj.converged
    ene_ghf = ghf_obj.energy_elec()[0]
    
    gmp_obj = mp.GMP2(ghf_obj)
    gmp_obj.verbose = 0

    ovlp_ao = ghf_obj.get_ovlp()

    from pyscf.ci.cisd import tn_addrs_signs
    t2addr, t2sign = tn_addrs_signs(norb_alph + norb_beta, nelec_alph + nelec_beta, 2)

    for beta in numpy.linspace(0.0, numpy.pi, 11):
        # l_matrix_y = numpy.array([[0.0,  0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        # l_matrix_z = numpy.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [ 0.0, 0.0, 0.0]])

        # rot_matrix_2 = scipy.linalg.expm(beta  * l_matrix_y)
        # rot_matrix_3 = scipy.linalg.expm(gamma * l_matrix_z)

        # rot_matrix = reduce(numpy.dot, [rot_matrix_2, rot_matrix_3])
        # s_uhf_rot = numpy.dot(rot_matrix, s_uhf)

        coeff_uhf_beta = rotate_coeff_ghf(coeff_uhf, beta=beta).real
        rdm1_uhf_beta  = ghf_obj.make_rdm1(coeff_uhf_beta, mo_occ_uhf)

        ene_mp2, t2 = gmp_obj.kernel(mo_coeff=coeff_uhf_beta)
        ene_mp2    += ene_uhf

        err = ghf_obj.get_grad(coeff_uhf_beta, mo_occ_uhf)
        err = numpy.linalg.norm(err)
        assert err < 1e-6

        u = reduce(numpy.dot, [coeff_uhf_beta.conj().T, ovlp_ao, coeff_rhf])
        coeff_uhf_beta_ = reduce(numpy.dot, [coeff_rhf, u.T])
        assert numpy.linalg.norm(coeff_uhf_beta - _) < 1e-8

        vfci_uhf_beta = numpy.zeros((ndet, 1))
        vfci_uhf_beta[0, 0] = 1.0

        vfci_uhf_beta = addons.transform_ci_for_orbital_rotation(vfci_uhf_beta, norb_alph + norb_beta, (nelec_alph + nelec_beta, 0), u)
        vfci_uhf_list.append(vfci_uhf_beta)

        vfci_mp2_beta = numpy.zeros((ndet, 1))
        vfci_mp2_beta[0, 0] = 1.0

        nocc = 4
        nvir = norb_alph + norb_beta - nocc
        oo_idx = numpy.tril_indices(nocc, -1)
        vv_idx = numpy.tril_indices(nvir, -1)
        t2_ = t2[oo_idx][:, vv_idx[0], vv_idx[1]]
        vfci_mp2_beta[t2addr, 0] = t2_.ravel() * t2sign
        
        vfci_mp2_beta = addons.transform_ci_for_orbital_rotation(vfci_mp2_beta, norb_alph + norb_beta, (nelec_alph + nelec_beta, 0), u)
        vfci_mp2_list.append(vfci_mp2_beta)

    h1e, h2e, ham = get_ghf_fci_ham(ghf_obj, coeff_rhf, (nelec_alph + nelec_beta, 0))
    ene_fci, vfci = fci.direct_spin1.kernel(h1e, h2e, norb_alph + norb_beta, (nelec_alph + nelec_beta, 0))

    data_dict = {
        "r": r, "ene_nuc": mol.energy_nuc(),
        "RHF": ene_rhf, 
        "UHF": ene_uhf, 
        "GHF": ene_ghf,
        "FCI": ene_fci,
        "MP2": ene_mp2,
    }

    vfci_uhf_list = numpy.asarray(vfci_uhf_list)
    vfci_mp2_list = numpy.asarray(vfci_mp2_list)

    h_dot_v       = [contract_2e(ham, v, norb_alph + norb_beta, (nelec_alph + nelec_beta, 0)) for v in vfci_mp2_list]

    for iv, v_list in enumerate([vfci_uhf_list, vfci_mp2_list]):
        v_dot_v  = numpy.einsum("Imn,Jmn->IJ", v_list.conj(), vfci_mp2_list)
        v_dot_hv = numpy.einsum("Imn,Jmn->IJ", v_list.conj(), h_dot_v)

        e, c = numpy.linalg.eig(v_dot_v)

        mask = abs(e.real) > 1e-8
        print(mask)
        cs = c[:, mask]

        v_dot_v_  = reduce(numpy.dot, [cs.conj().T, v_dot_v, cs])
        v_dot_hv_ = reduce(numpy.dot, [cs.conj().T, v_dot_hv, cs])

        print(v_dot_v_)
        print(v_dot_hv_)

        ene_noci = scipy.linalg.eig(v_dot_hv_, v_dot_v_)[0].real
        data_dict[f"NOCI-{iv}"] = ene_noci[0]


    from pyscf import lib
    os.makedirs(f"/Users/yangjunjie/work/bs-uhf/data/h4", exist_ok=True)
    lib.chkfile.save(f"/Users/yangjunjie/work/bs-uhf/data/h4/bs-uhf-{basis}.h5", f"{r}", data_dict)

if __name__ == "__main__":
    basis = "sto3g"
    with open(f"/Users/yangjunjie/work/bs-uhf/data/h4/bs-uhf-{basis}.csv", "w") as f:
        for x in numpy.linspace(0.4, 3.2, 41):
            if abs(x - 1.4) < 1e-1: # break
                solve_h4_bs_uhf(x, basis=basis, f=f)
                
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

def solve_h6_bs_uhf(r, basis="sto-3g", f=None):
    atoms = ""
    for i in range(6):
        theta  = numpy.pi / 3.0 * i
        coordx = r * numpy.cos(theta)
        coordy = r * numpy.sin(theta)
        atoms += f"H{i+1} {coordx: 12.8f} {coordy: 12.8f} 0.00000\n"
        
    mol = gto.Mole()
    mol.atom  = atoms
    mol.basis = basis
    mol.build()

    rhf_obj = scf.RHF(mol)
    rhf_obj.verbose = 0
    rhf_obj.conv_tol = 1e-12
    rhf_obj.kernel()
    assert rhf_obj.converged
    mo_coeff, mo_energy = rhf_obj.mo_coeff, rhf_obj.mo_energy
    coeff_rhf = coeff_rhf_to_ghf(mo_coeff, mo_energy)
    ene_rhf   = rhf_obj.energy_elec()[0]

    fci_obj = fci.FCI(mol, rhf_obj.mo_coeff)
    fci_obj.verbose = 0
    ene_fci  = fci_obj.kernel()[0]
    ene_fci -= mol.energy_nuc()

    dm0 = numpy.zeros((2, mol.nao, mol.nao))
    dm0[0, 0, 0] = 1.0
    dm0[0, 2, 2] = 1.0
    dm0[0, 4, 4] = 1.0
    dm0[1, 1, 1] = 1.0
    dm0[1, 3, 3] = 1.0
    dm0[1, 5, 5] = 1.0

    uhf_obj = scf.UHF(mol)
    uhf_obj.verbose = 0
    uhf_obj.conv_tol = 1e-12
    uhf_obj.kernel(dm0=dm0)
    assert uhf_obj.converged
    ene_uhf = uhf_obj.energy_elec()[0]
    dm_uhf  = uhf_obj.make_rdm1()
    coeff_uhf, mo_occ_uhf = coeff_uhf_to_ghf(uhf_obj.mo_coeff, uhf_obj.mo_energy, uhf_obj.mo_occ)

    ghf_obj = scf.GHF(mol)
    ghf_obj.verbose = 0
    rdm1_uhf = ghf_obj.make_rdm1(coeff_uhf, mo_occ_uhf)
    ghf_obj.kernel(dm0=rdm1_uhf)
    assert ghf_obj.converged
    ene_ghf = ghf_obj.energy_elec()[0]
    ovlp_ao = ghf_obj.get_ovlp()
    
    gmp_obj = mp.GMP2(ghf_obj)
    gmp_obj.verbose = 0
    ene_gmp2  = gmp_obj.kernel()[0]
    ene_gmp2 += ene_ghf

    from pyscf.fci import cistring, addons
    norb_alph, norb_beta   = uhf_obj.mo_coeff[0].shape[1], uhf_obj.mo_coeff[1].shape[1]
    nelec_alph, nelec_beta = uhf_obj.nelec[0], uhf_obj.nelec[1]
    ndet     = cistring.num_strings(norb_alph + norb_beta, nelec_alph + nelec_beta)

    nocc = nelec_alph + nelec_beta
    norb = norb_alph + norb_beta
    nvir = norb - nocc

    vfci_uhf_list = []
    vfci_mp2_list = []

    from pyscf.ci.cisd import tn_addrs_signs
    t2addr, t2sign = tn_addrs_signs(norb_alph + norb_beta, nelec_alph + nelec_beta, 2)

    for beta in numpy.linspace(0.0, numpy.pi, 21):
        coeff_uhf_beta = rotate_coeff_ghf(coeff_uhf, beta=beta).real
        rdm1_uhf_beta  = ghf_obj.make_rdm1(coeff_uhf_beta, mo_occ_uhf)

        csc  = reduce(numpy.dot, (coeff_uhf_beta.conj().T, ovlp_ao, coeff_uhf_beta))
        is_orth = numpy.linalg.norm(csc - numpy.eye(norb)) < 1e-8
        assert is_orth

        ene_mp2, t2  = gmp_obj.kernel(mo_coeff=coeff_uhf_beta)
        ene_mp2     += ene_uhf
        assert abs(ene_mp2 - ene_gmp2) < 1e-8

        err = ghf_obj.get_grad(coeff_uhf_beta, mo_occ_uhf)
        err = numpy.linalg.norm(err)
        assert err < 1e-6

        u = reduce(numpy.dot, [coeff_uhf_beta.conj().T, ovlp_ao, coeff_rhf])
        coeff_uhf_beta_ = reduce(numpy.dot, [coeff_rhf, u.T])
        assert numpy.linalg.norm(coeff_uhf_beta - coeff_uhf_beta_) < 1e-8

        vfci_uhf_beta = numpy.zeros((ndet, 1))
        vfci_uhf_beta[0, 0] = 1.0
        vfci_uhf_list.append(addons.transform_ci_for_orbital_rotation(vfci_uhf_beta, norb_alph + norb_beta, (nelec_alph + nelec_beta, 0), u))

        # vfci_mp2_beta = numpy.zeros((ndet, 1))
        # vfci_mp2_beta[0, 0] = 1.0

        # oo_idx = numpy.tril_indices(nocc, -1)
        # vv_idx = numpy.tril_indices(nvir, -1)
        # t2_ = t2[oo_idx][:, vv_idx[0], vv_idx[1]]
        # vfci_mp2_beta[t2addr, 0] = t2_.ravel() * t2sign
        
        # vfci_mp2_beta = addons.transform_ci_for_orbital_rotation(vfci_mp2_beta, norb_alph + norb_beta, (nelec_alph + nelec_beta, 0), u)
        # vfci_mp2_list.append(vfci_mp2_beta)

    h1e, h2e, ham = get_ghf_fci_ham(ghf_obj, coeff_rhf, (nelec_alph + nelec_beta, 0))

    data_dict = {
        "r": r, "ene_nuc": mol.energy_nuc(),
        "RHF": ene_rhf, 
        "UHF": ene_uhf, 
        "GHF": ene_ghf,
        "FCI": ene_fci,
        "MP2": ene_gmp2,
    }

    print(f"r = {r: 12.6f}, ene_rhf = {ene_rhf: 12.6f}, ene_uhf = {ene_uhf: 12.6f}, ene_fci = {ene_fci: 12.6f}")

    vfci_uhf_list = numpy.asarray(vfci_uhf_list)
    vfci_mp2_list = numpy.asarray(vfci_mp2_list)

    vhf_dot_vhf = numpy.einsum("Imn,Jmn->IJ", vfci_uhf_list.conj(), vfci_uhf_list)
    s, c        = numpy.linalg.eigh(vhf_dot_vhf)
    print("".join([f"{x: 12.2e}" for x in s]))

    # h_dot_v       = [contract_2e(ham, v, norb_alph + norb_beta, (nelec_alph + nelec_beta, 0)) for v in vfci_mp2_list]

    # for iv, v_list in enumerate([vfci_uhf_list, vfci_mp2_list]):
    #     v_dot_v  = numpy.einsum("Imn,Jmn->IJ", v_list.conj(), vfci_mp2_list)
    #     v_dot_hv = numpy.einsum("Imn,Jmn->IJ", v_list.conj(), h_dot_v)

    #     e, c = numpy.linalg.eig(v_dot_v)

    #     mask = abs(e.real) > 1e-8
    #     cs = c[:, mask]

    #     v_dot_v_  = reduce(numpy.dot, [cs.conj().T, v_dot_v, cs])
    #     v_dot_hv_ = reduce(numpy.dot, [cs.conj().T, v_dot_hv, cs])

    #     print(v_dot_v_)
    #     print(v_dot_hv_)

    #     ene_noci = scipy.linalg.eig(v_dot_hv_, v_dot_v_)[0].real
    #     data_dict[f"NOCI-{iv}"] = ene_noci[0]

    # assert 1 == 2

    from pyscf import lib
    os.makedirs(f"/Users/yangjunjie/work/bs-uhf/data/h6", exist_ok=True)
    lib.chkfile.save(f"/Users/yangjunjie/work/bs-uhf/data/h6/bs-uhf-{basis}.h5", f"{r}", data_dict)

if __name__ == "__main__":
    basis = "sto3g"
    dirpath = f"/Users/yangjunjie/work/bs-uhf/data/h6/"
    os.makedirs(dirpath, exist_ok=True)

    with open(f"{dirpath}/bs-uhf-{basis}.csv", "w") as f:
        for x in numpy.linspace(0.4, 3.2, 41):
            if x > 3.0:
                solve_h6_bs_uhf(x, basis=basis, f=f)
                break

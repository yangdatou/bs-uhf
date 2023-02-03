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
    coeff_beta = coeff_ghf[nao:, :]
    h2e_aabb = ao2mo.kernel(ghf_obj._eri, (coeff_alph, coeff_alph, coeff_beta, coeff_beta))

    h2e  = ao2mo.kernel(ghf_obj._eri, coeff_alph)
    h2e += ao2mo.kernel(ghf_obj._eri, coeff_beta)
    h2e += h2e_aabb + h2e_aabb.T

    return h1e, h2e, absorb_h1e(h1e, h2e, norb, nelec, 0.5)

def solve_hub_bs_uhf(nelecs, nsite, u=1.0, is_pbc=False):
    nelec_alph, nelec_beta = nelecs

    mol = gto.Mole()
    mol.incore_anyway = True
    mol.nelectron = nelec_alph + nelec_beta
    mol.spin      = nelec_alph - nelec_beta
    mol.build()

    h1e = numpy.zeros((nsite, nsite))
    h2e = numpy.zeros((nsite, nsite, nsite, nsite))

    for i in range(nsite - 1):
        h1e[i, i+1] = -1.0
        h1e[i+1, i] = -1.0
        h2e[i, i, i, i] = u

    if is_pbc:
        h1e[0, nsite-1] = -1.0
        h1e[nsite-1, 0] = -1.0

    h2e[nsite-1, nsite-1, nsite-1, nsite-1] = u

    rhf_obj = scf.RHF(mol)
    rhf_obj.init_guess = "1e"
    rhf_obj.get_hcore  = lambda *args: h1e
    rhf_obj.get_ovlp   = lambda *args: numpy.eye(nsite)
    rhf_obj._eri       = ao2mo.restore(8, h2e, nsite)

    rhf_obj.verbose = 0
    rhf_obj.conv_tol = 1e-12
    rhf_obj.kernel()
    assert rhf_obj.converged

    res = coeff_rhf_to_ghf(
        rhf_obj.mo_coeff,
        rhf_obj.mo_energy,
        rhf_obj.mo_occ
    )
    coeff_rhf    = res[0]
    mo_occ_rhf   = res[1]
    orb_spin_rhf = res[2]
    ene_rhf      = rhf_obj.energy_elec()[0]

    dm0 = numpy.zeros((2, nsite, nsite))
    dm0[0, 0, 0] = 1.0
    dm0[0, 2, 2] = 1.0
    dm0[1, 1, 1] = 1.0
    dm0[1, 3, 3] = 1.0

    uhf_obj = scf.UHF(mol)
    uhf_obj.get_hcore  = lambda *args: h1e
    uhf_obj.get_ovlp   = lambda *args: numpy.eye(nsite)
    uhf_obj._eri       = ao2mo.restore(8, h2e, nsite)
    uhf_obj.verbose  = 0
    uhf_obj.conv_tol = 1e-8
    uhf_obj.kernel(dm0)
    assert uhf_obj.converged

    res = coeff_uhf_to_ghf(
        uhf_obj.mo_coeff,
        uhf_obj.mo_energy,
        uhf_obj.mo_occ
    )
    coeff_uhf    = res[0]
    mo_occ_uhf   = res[1]
    orb_spin_uhf = res[2]
    ene_uhf      = uhf_obj.energy_elec()[0]

    coeff_uhf_ = numpy.zeros_like(coeff_uhf)
    for i in range(coeff_uhf.shape[1]):
        mask = numpy.argmax(numpy.abs(coeff_uhf[:, i]))
        coeff_uhf_[mask, i] = 1.0
    
    coeff_uhf = coeff_uhf_

    norb_alph = uhf_obj.mo_coeff[0].shape[1]
    norb_beta = uhf_obj.mo_coeff[1].shape[1]
    norb = norb_alph + norb_beta

    ovlp_ao = numpy.eye(nsite * 2)
    ghf_obj = scf.GHF(mol)
    ghf_obj.get_hcore  = lambda *args: scipy.linalg.block_diag(h1e, h1e)
    ghf_obj.get_ovlp   = lambda *args: ovlp_ao
    ghf_obj._eri       = h2e
    ghf_obj.verbose  = 0
    ghf_obj.conv_tol = 1e-8
    
    dm_rhf = ghf_obj.make_rdm1(coeff_rhf, mo_occ_rhf)
    dm_uhf = ghf_obj.make_rdm1(coeff_uhf, mo_occ_uhf)

    # dump_rec(stdout, coeff_uhf)
    # coeff_uhf_   = 
    # coeff_uhf    = numpy.array(coeff_uhf_)
    # dump_rec(stdout, coeff_uhf)

    ghf_obj.kernel(dm_uhf)
    assert ghf_obj.converged

    ene_ghf = ghf_obj.energy_elec()[0]

    gmp2 = mp.MP2(ghf_obj)
    gmp2.verbose = 0
    gmp2.kernel()
    ene_gmp2 = gmp2.e_corr + ene_ghf

    from pyscf.fci import cistring, addons
    norb_alph, norb_beta   = uhf_obj.mo_coeff[0].shape[1], uhf_obj.mo_coeff[1].shape[1]
    nelec_alph, nelec_beta = uhf_obj.nelec[0], uhf_obj.nelec[1]
    ndet      = cistring.num_strings(norb_alph + norb_beta, nelec_alph + nelec_beta)
    occ_list  = cistring._gen_occslst(range(norb), nelec_alph + nelec_beta)
    orb_label = ["0a", "1a", "2a", "3a", "0b", "1b"]
    orb_label  = ["%s%s" % (i, "a") for i in range(norb_alph)]
    orb_label += ["%s%s" % (i, "b") for i in range(norb_beta)]

    nocc = nelec_alph + nelec_beta
    norb = norb_alph + norb_beta
    nvir = norb - nocc

    vfci_uhf_list = []
    vfci_mp2_list = []

    from pyscf.ci.cisd import tn_addrs_signs
    t2addr, t2sign = tn_addrs_signs(norb, nocc, 2)

    coeff_rhf = ovlp_ao

    for alpha in numpy.linspace(0.0, numpy.pi, 20):
    # alpha = 1.0
        for beta in numpy.linspace(0.0, numpy.pi, 20):
            coeff_uhf_beta = rotate_coeff_ghf(coeff_uhf, alpha=alpha, beta=beta)
            rdm1_uhf_beta  = ghf_obj.make_rdm1(coeff_uhf_beta, mo_occ_uhf)

            csc  = reduce(numpy.dot, (coeff_uhf_beta.conj().T, ovlp_ao, coeff_uhf_beta))
            is_orth = numpy.linalg.norm(csc - numpy.eye(norb)) < 1e-8
            assert is_orth

            # ene_mp2, t2  = gmp_obj.kernel(mo_coeff=coeff_uhf_beta)
            # ene_mp2     += ene_uhf
            # assert abs(ene_mp2 - ene_gmp2) < 1e-8

            # err = ghf_obj.get_grad(coeff_uhf_beta, mo_occ_uhf)
            # err = numpy.linalg.norm(err)
            # assert err < 1e-4

            u = coeff_uhf_beta.conj().T @ coeff_rhf
            coeff_uhf_beta_ = coeff_rhf @ u.conj().T
            assert numpy.linalg.norm(coeff_uhf_beta - coeff_uhf_beta_) < 1e-8

            vfci_uhf_beta = numpy.zeros((ndet, 1))
            vfci_uhf_beta[0, 0] = 1.0
            vfci_uhf_list.append(addons.transform_ci_for_orbital_rotation(vfci_uhf_beta, norb_alph + norb_beta, (nelec_alph + nelec_beta, 0), u))

            c = vfci_uhf_list[-1]
            print("\nbeta = %12.4e" % beta)
            for i, ci in enumerate(c):
                occ_i = occ_list[i]
                occ_i_str = "".join([orb_label[j] for j in occ_i])

                if numpy.abs(ci[0]) > 1e-8:
                    print("  %s: %12.6f + %12.6f i" % (occ_i_str, ci[0].real, ci[0].imag))

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
        "RHF": ene_rhf, 
        "UHF": ene_uhf, 
        "GHF": ene_ghf,
        "MP2": ene_gmp2,
    }

    vfci_uhf_list = numpy.array(vfci_uhf_list)
    
    v_dot_v = numpy.einsum("Iij,Jij->IJ", vfci_uhf_list.conj(), vfci_uhf_list)
    ee, c    = scipy.linalg.eigh(v_dot_v)
    assert numpy.linalg.norm(v_dot_v - c @ numpy.diag(ee) @ c.conj().T) < 1e-8
    cc = numpy.einsum("Iij,IX->Xij", vfci_uhf_list, c.conj())
    
    for ie, (e, c) in enumerate(zip(ee, cc)):
        if abs(e**2) < 1e-20:
            continue

        print("\nie = %d, e = %12.4e" % (ie, e))
        for i, ci in enumerate(c):
            occ_i = occ_list[i]
            occ_i_str = "".join([orb_label[j] for j in occ_i])

            if numpy.abs(ci[0]) > 1e-8:
                print("  %s: %12.6f + %12.6f i" % (occ_i_str, ci[0].real, ci[0].imag))

    mask = numpy.abs(ee) > 1e-8
    print("\n\n")
    print("Number of states: ", numpy.sum(mask))


if __name__ == "__main__":
    dirpath = f"/Users/yangjunjie/work/bs-uhf/data/hub/"
    os.makedirs(dirpath, exist_ok=True)

    solve_hub_bs_uhf((2, 2), 4, u=20.0, is_pbc=False)

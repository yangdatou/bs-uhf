from functools import reduce
from itertools import combinations

import os, sys
from sys import stdout
import numpy, scipy
from scipy.linalg import sqrtm

import pyscf
from pyscf import gto, scf, lo, fci
from pyscf import ao2mo, mp, mcscf
from pyscf.tools.dump_mat   import dump_rec
from pyscf.ci.ucisd import amplitudes_to_cisdvec
from pyscf.fci.direct_spin1 import absorb_h1e
from pyscf.fci.direct_spin1 import contract_2e

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

def get_ump2_t1(mp2_obj, eris, ignore_t1=True, is_delta_inv=True):
    fock_a, fock_b = eris.fock
    nocc_a, nocc_b = eris.nocc
    norb = fock_a.shape[0]
    nvir_a = norb - nocc_a
    nvir_b = norb - nocc_b

    foo_a = fock_a[:nocc_a, :nocc_a]
    fvv_a = fock_a[nocc_a:, nocc_a:]
    fov_a = fock_a[:nocc_a, nocc_a:]

    foo_b = fock_b[:nocc_b, :nocc_b]
    fvv_b = fock_b[nocc_b:, nocc_b:]
    fov_b = fock_b[:nocc_b, nocc_b:]

    ene_t1 = 0.0
    t1_a   = numpy.zeros((nocc_a, nvir_a))
    t1_b   = numpy.zeros((nocc_b, nvir_b))

    if not ignore_t1:
        if is_delta_inv:
            d_a     = numpy.einsum('ab,ij->aibj', fvv_a, numpy.eye(nocc_a))
            d_a    -= numpy.einsum('ab,ij->aibj', numpy.eye(nvir_a), foo_a)
            d_a     = d_a.reshape(nvir_a * nocc_a, nvir_a * nocc_a)
            d_a_inv = numpy.linalg.inv(d_a).reshape(nvir_a, nocc_a, nvir_a, nocc_a)

            d_b     = numpy.einsum('ab,ij->aibj', fvv_b, numpy.eye(nocc_b))
            d_b    -= numpy.einsum('ab,ij->aibj', numpy.eye(nvir_b), foo_b)
            d_b     = d_b.reshape(nvir_b * nocc_b, nvir_b * nocc_b)
            d_b_inv = numpy.linalg.inv(d_b).reshape(nvir_b, nocc_b, nvir_b, nocc_b)

            t1_a = - numpy.einsum('ia,bjai->jb', fov_a, d_a_inv)
            t1_b = - numpy.einsum('ia,bjai->jb', fov_b, d_b_inv)

        else:
            d_a = numpy.diag(fvv_a)[:, None] - numpy.diag(foo_a)[None, :]
            d_b = numpy.diag(fvv_b)[:, None] - numpy.diag(foo_b)[None, :]

            t1_a = - fov_a.conj() / d_a.T
            t1_b = - fov_b.conj() / d_b.T

        ene_t1 += numpy.einsum('ia,ia->', fov_a, t1_a)
        ene_t1 += numpy.einsum('ia,ia->', fov_b, t1_b)

    return ene_t1, (t1_a, t1_b)

def get_ump2_t2(mp2_obj, eris, use_iterative_kernel=False):
    from pyscf.mp.ump2 import kernel
    from pyscf.mp.mp2  import _iterative_kernel

    if use_iterative_kernel:
        is_converged, ene_t2, t2 = _iterative_kernel(mp2_obj, eris=eris, verbose=5)
        assert is_converged
    else:
        ene_t2, t2 = kernel(mp2_obj, eris=eris, with_t2=True)

    return ene_t2, t2

def analyze_mo(mos, mol_obj, tol=0.1):
    nao, nmo = mos.shape
    natm = mol_obj.natm
    
    ovlp_ao = mol_obj.intor("int1e_ovlp")
    w_ao    = sqrtm(ovlp_ao)
    assert w_ao.shape == (nao, nao)

    w_ao_mo  = numpy.einsum("mn,mp->np", w_ao, mos)
    w2_ao_mo = numpy.einsum("np,np->np", w_ao_mo, w_ao_mo)
    assert w2_ao_mo.shape == (nao, nmo)

    ao_label = mol_obj.ao_labels()
    mo_label = []

    for p in range(nmo):
        w2_ao_p = w2_ao_mo[:, p]
        w2_ao_p_argsort = numpy.argsort(w2_ao_p)[::-1]
    
        assert w2_ao_p[w2_ao_p_argsort[0]] > tol

        mo_label.append(ao_label[w2_ao_p_argsort[0]])
        print("MO %5d: %s, weight: %6.4f" % (p, ao_label[w2_ao_p_argsort[0]], w2_ao_p[w2_ao_p_argsort[0]]))

    return mo_label

def bs_uhf_n2_uno(x=1.0, spin=0, basis="ccpvdz"):
    mol = gto.Mole()
    mol.verbose = 0
    # mol.output = './tmp/n2-bs-uhf-%.2f.out' % x
    mol.atom = f"""
        N 0.00000 0.00000 0.00000
        N 0.00000 0.00000 {x:12.8f}
    """ 
    mol.basis = basis
    mol.spin = spin
    mol.build()

    ao_labels = mol.ao_labels()
    nelec      = mol.nelec
    nelec_alph = mol.nelec[0]
    nelec_beta = mol.nelec[1]

    ao_idx = list(range(mol.nao_nr()))
    nao = len(ao_idx)

    core_idx   = []
    core_idx  += list(mol.search_ao_label("N 1s"))
    core_idx  += list(mol.search_ao_label("N 2s"))
    norb_core  = len(core_idx)
    
    bs_idx     = []
    bs_idx    += list(mol.search_ao_label("N 2p"))

    
    bs_ao_idx_alph_list = list(combinations(bs_idx, nelec_alph - norb_core))

    ene_uhf_list = []

    bs_ao_idx_alph = list(mol.search_ao_label("0 N 2p"))
    bs_ao_idx_beta = list(mol.search_ao_label("1 N 2p"))

    print("bs_ao_idx_alph = ", bs_ao_idx_alph)
    print("bs_ao_idx_beta = ", bs_ao_idx_beta)

    dm0 = numpy.zeros((2, nao, nao))
    dm0[0, core_idx, core_idx] = 1.0
    dm0[1, core_idx, core_idx] = 1.0
    dm0[0, bs_ao_idx_alph, bs_ao_idx_alph] = 1.0
    dm0[1, bs_ao_idx_beta, bs_ao_idx_beta] = 1.0

    mf = scf.UHF(mol)
    mf.verbose = 4
    mf.kernel(dm0=dm0)

    coeff_alph = mf.mo_coeff[0] # [:, :nelec_alph]
    coeff_beta = mf.mo_coeff[1] # [:, :nelec_beta]
    coeff_alph_occ = coeff_alph[:, :nelec_alph]
    coeff_beta_occ  = coeff_beta[:, :nelec_beta]

    ovlp_ab = reduce(numpy.dot, (coeff_alph.T, mf.get_ovlp(), coeff_beta))
    print("ovlp_ab = ")
    dump_rec(stdout, ovlp_ab, label=mol.ao_labels(), ncol=7)

    ovlp_ab_occ = reduce(numpy.dot, (coeff_alph_occ.T, mf.get_ovlp(), coeff_beta_occ))
    print("ovlp_ab_occ = ")
    dump_rec(stdout, ovlp_ab_occ, label=mol.ao_labels(), ncol=7)

    dm_uhf = mf.make_rdm1()
    dm     = dm_uhf[0] + dm_uhf[1]
    dm = reduce(numpy.dot, (coeff_alph.T, mf.get_ovlp(), dm, mf.get_ovlp(), coeff_alph))
    print("dm = ")
    dump_rec(stdout, dm, label=mol.ao_labels(), ncol=7)

    u, s, vh = scipy.linalg.svd(dm_tot)
    print("s = ", ", ".join(["%6.4f" % x for x in s]))

 

def solve_n2_rohf(x=1.0, spin=0, basis="ccpvdz"):
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = './tmp/n2-bs-uhf-%.2f.out' % x
    mol.atom = f"""
        N 0.00000 0.00000 0.00000
        N 0.00000 0.00000 {x:12.8f}
    """ 
    mol.basis = basis
    mol.spin = spin
    mol.build()

    ao_labels = mol.ao_labels()
    nelec      = mol.nelec
    nelec_alph = mol.nelec[0]
    nelec_beta = mol.nelec[1]

    ao_idx = list(range(mol.nao_nr()))

    core_idx = []
    core_idx += list(mol.search_ao_label("N 1s"))
    core_idx += list(mol.search_ao_label("N 2s"))
    
    bs_idx  = []
    bs_idx += list(mol.search_ao_label("N 2p"))

    rhf_obj = scf.RHF(mol)
    rhf_obj.kernel(dm0=None)

    # fci_obj = fci.FCI(mol, rhf_obj.mo_coeff)
    # fci_obj.kernel()
    # print("FCI energy: %16.8f" % (fci_obj.e_tot - mol.energy_nuc()))
    # assert 1 == 2

    coeff_rhf = rhf_obj.mo_coeff
    coeff_iao = localize_mo(coeff_rhf, mol_obj=mol, ovlp_ao=mol.intor("int1e_ovlp"), method="iao")
    nao, nlo  = coeff_iao.shape
    norb      = nlo

    h1e  = reduce(numpy.dot, (coeff_iao.conj().T, rhf_obj.get_hcore(), coeff_iao))
    h2e  = ao2mo.kernel(rhf_obj._eri, coeff_iao)
    ham  = absorb_h1e(h1e, h2e, norb, (nelec_alph, nelec_beta), .5)

    ao_label  = mol.ao_labels()
    ovlp_ao   = rhf_obj.get_ovlp()
    w_ao      = sqrtm(ovlp_ao)
    w_ao_iao  = numpy.einsum("mn,mp->np", w_ao, coeff_iao)
    w2_ao_iao = numpy.einsum("np,np->np", w_ao_iao, w_ao_iao)

    for p in range(nlo):
        print("LO %d %s, weight: %6.4f" % (p, ao_label[p], w2_ao_iao[p, p]))
        assert abs(numpy.sum(w2_ao_iao[:, p]) - 1.0) < 1e-6
        assert w2_ao_iao[p, p] > 0.5

    uhf_obj    = scf.UHF(mol)

    nelec_core = len(core_idx)
    bs_ao_idx_alph_list = list(combinations(bs_idx, nelec_alph - nelec_core))
    bs_ao_idx_beta_list = list(combinations(bs_idx, nelec_beta - nelec_core))

    ene_uhf_list = []

    for bs_ao_idx_alph in bs_ao_idx_alph_list:
        for bs_ao_idx_beta in bs_ao_idx_beta_list:
            bs_ao_idx_alph = list(numpy.sort(bs_ao_idx_alph))
            bs_ao_idx_beta = list(numpy.sort(bs_ao_idx_beta))

            # assert bs_ao_idx_alph != [1, 3, 4, 6, 7]
            # assert bs_ao_idx_beta != [1, 3, 4, 6, 7]

            vir_ao_idx_alph = list(set(ao_idx) - set(core_idx) - set(bs_ao_idx_alph))
            vir_ao_idx_beta = list(set(ao_idx) - set(core_idx) - set(bs_ao_idx_beta))

            coeff_iao_core     = coeff_iao[:, core_idx]
            coeff_iao_bs_alph  = coeff_iao[:, bs_ao_idx_alph]
            coeff_iao_bs_beta  = coeff_iao[:, bs_ao_idx_beta]
            coeff_iao_vir_alph = coeff_iao[:, vir_ao_idx_alph]
            coeff_iao_vir_beta = coeff_iao[:, vir_ao_idx_beta]

            assert coeff_iao_core.shape[1]     == nelec_core
            assert coeff_iao_bs_alph.shape[1]  == nelec_alph - nelec_core
            assert coeff_iao_bs_beta.shape[1]  == nelec_beta - nelec_core
            assert coeff_iao_vir_alph.shape[1] == nlo - nelec_alph
            assert coeff_iao_vir_beta.shape[1] == nlo - nelec_beta

            coeff_iao_occ_alph = numpy.hstack((coeff_iao_core, coeff_iao_bs_alph))
            coeff_iao_occ_beta = numpy.hstack((coeff_iao_core, coeff_iao_bs_beta))
            dm_alph = numpy.einsum("pi,qi->pq", coeff_iao_occ_alph, coeff_iao_occ_alph)
            dm_beta = numpy.einsum("pi,qi->pq", coeff_iao_occ_beta, coeff_iao_occ_beta)

            nocc_alph = coeff_iao_occ_alph.shape[1]
            nocc_beta = coeff_iao_occ_beta.shape[1]
            nvir_alph = coeff_iao_vir_alph.shape[1]
            nvir_beta = coeff_iao_vir_beta.shape[1]

            assert nocc_alph == nelec_alph
            assert nocc_beta == nelec_beta

            coeff_iao_alph = numpy.hstack((coeff_iao_occ_alph, coeff_iao_vir_alph))
            coeff_iao_beta = numpy.hstack((coeff_iao_occ_beta, coeff_iao_vir_beta))

            mo_occ_alph = [1] * nocc_alph + [0] * nvir_alph
            mo_occ_beta = [1] * nocc_beta + [0] * nvir_beta

            coeff_iao_uhf = (coeff_iao_alph, coeff_iao_beta)
            mo_occ_uhf    = numpy.array((mo_occ_alph, mo_occ_beta))
            dm_uhf        = (dm_alph, dm_beta)

            from bs import proj_uhf_to_fci_vec, proj_ucisd_to_fci_vec
            vfci_uhf = proj_uhf_to_fci_vec(coeff_iao, coeff_iao_uhf, nelec, ovlp_ao)
            v2       = numpy.einsum("ab,ab->", vfci_uhf, vfci_uhf)
            assert abs(v2 - 1.0) < 1e-6

            ene_uhf     = uhf_obj.energy_elec(dm=dm_uhf, h1e=None, vhf=None)[0]
            hv          = contract_2e(ham, vfci_uhf, norb, nelec)
            ene_uhf_ref = numpy.einsum("ab,ab->", hv, vfci_uhf)
            assert abs(ene_uhf - ene_uhf_ref) < 1e-6

            mp2_obj  = mp.MP2(uhf_obj, mo_coeff=coeff_iao_uhf, mo_occ=mo_occ_uhf)
            mp2_obj.verbose = 0
            eris     = mp2_obj.ao2mo()

            fvv_a = eris.fock[0][nocc_alph:, nocc_alph:]
            fvv_b = eris.fock[1][nocc_beta:, nocc_beta:]
            foo_a = eris.fock[0][:nocc_alph, :nocc_alph]
            foo_b = eris.fock[1][:nocc_beta, :nocc_beta]
            d_a = numpy.diag(fvv_a)[:, None] - numpy.diag(foo_a)[None, :]
            d_b = numpy.diag(fvv_b)[:, None] - numpy.diag(foo_b)[None, :]

            # print("\nmo_ene occ alph:" + " ".join(["%10.5f" % x for x in numpy.diag(foo_a)]))
            # print("mo_ene occ beta:" + " ".join(["%10.5f" % x for x in numpy.diag(foo_b)]))
            # print("mo_ene vir alph:" + " ".join(["%10.5f" % x for x in numpy.diag(fvv_a)]))
            # print("mo_ene vir beta:" + " ".join(["%10.5f" % x for x in numpy.diag(fvv_b)]))

            if numpy.min(d_a) < 0.0:
                # print("Warning: fvv_a is not positive definite")
                continue

            if numpy.min(d_b) < 0.0:
                # print("Warning: fvv_b is not positive definite")
                continue

            print("\n")
            for bs_ao_idx in bs_ao_idx_alph:
                print("alpha occupied: ao_idx = %d, ao_label = %s" % (bs_ao_idx, ao_labels[bs_ao_idx]))

            for bs_ao_idx in bs_ao_idx_beta:
                print("beta  occupied: ao_idx = %d, ao_label = %s" % (bs_ao_idx, ao_labels[bs_ao_idx]))

            t0 = 1.0
            ene_t0 = ene_uhf

            ene_t1_1, t1_1 = get_ump2_t1(mp2_obj, eris, ignore_t1=True,  is_delta_inv=False)
            ene_t1_2, t1_2 = get_ump2_t1(mp2_obj, eris, ignore_t1=False, is_delta_inv=True)
            ene_t1_3, t1_3 = get_ump2_t1(mp2_obj, eris, ignore_t1=False, is_delta_inv=False)

            ene_t2_1, t2_1 = get_ump2_t2(mp2_obj, eris, use_iterative_kernel=True)
            ene_t2_2, t2_2 = get_ump2_t2(mp2_obj, eris, use_iterative_kernel=False)

            assert abs(ene_t2_1 - ene_t2_2) < 1e-6
            t2     = t2_1
            ene_t2 = ene_t2_1
            
            ene_uhf_list.append(ene_uhf)

            print("UHF energy             = %16.8f" % (ene_uhf))
            for idx_t1, (ene_t1, t1) in enumerate([(ene_t1_1, t1_1), (ene_t1_2, t1_2), (ene_t1_3, t1_3)]):
                ene_ump2         = ene_t0 + ene_t2
                ene_ump2_with_t1 = ene_t0 + ene_t1 + ene_t2
                
                vec_ump2  = amplitudes_to_cisdvec(t0, t1, t2)
                vfci_ump2 = proj_ucisd_to_fci_vec(coeff_iao, coeff_iao_uhf, vec_ump2, nelec, ovlp_ao)

                ene_ump2_ref = numpy.einsum("ab,ab->", hv, vfci_ump2)

                print("idx_t1 = %d" % (idx_t1))
                print("UMP2 energy             = %16.8f" % ene_ump2)
                print("UMP2 energy with t1     = %16.8f" % ene_ump2_with_t1)
                print("UMP2 energy from fcivec = %16.8f" % ene_ump2_ref)

                assert abs(ene_ump2_with_t1 - ene_ump2_ref) < 1e-6

            # assert bs_ao_idx_alph != [1, 3, 4, 6, 7] # or bs_ao_idx_beta != [1, 2, 6, 8, 9]

    print(len(ene_uhf_list))
    print(numpy.min(ene_uhf_list))

if __name__ == "__main__":
    for x in numpy.arange(0.4, 3.0, 0.1):
        x = 4.0
        bs_uhf_n2_uno(x=x, spin=0, basis="sto3g")
        assert 1 == 2
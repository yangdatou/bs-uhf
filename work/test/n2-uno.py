from functools import reduce
from itertools import combinations, groupby

import os, sys
from sys import stdout
import numpy, scipy
from scipy.linalg import sqrtm

import pyscf
from pyscf import gto, scf, lo, fci
from pyscf import ci, ao2mo, mp, mcscf
from pyscf.tools.dump_mat   import dump_rec
from pyscf.ci.ucisd import amplitudes_to_cisdvec
from pyscf.fci.direct_spin1 import absorb_h1e
from pyscf.fci.direct_spin1 import contract_2e

def bs_uhf_n2_uno(x=1.0, spin=0, basis="ccpvdz"):
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = f"""
        N 0.00000 0.00000 0.00000
        N 0.00000 0.00000 {x:12.8f}
    """ 

    mol.basis = basis
    mol.spin = spin
    mol.build()

    coeff_ao_lo = lo.orth_ao(mol, 'meta_lowdin')
    nao, nlo = coeff_ao_lo.shape

    ao_labels = mol.ao_labels()
    nelec      = mol.nelec
    nelec_alph = mol.nelec[0]
    nelec_beta = mol.nelec[1]

    ao_idx = list(range(mol.nao_nr()))
    nao = len(ao_idx)

    cor_ao_idx  = []
    cor_ao_idx += list(mol.search_ao_label("N 1s"))
    cor_ao_idx += list(mol.search_ao_label("N 2s"))

    act_ao_idx  = []
    act_ao_idx += list(mol.search_ao_label("N 2p"))
    
    ext_ao_idx  = list(set(ao_idx) - set(cor_ao_idx) - set(act_ao_idx))

    nao_cor = len(cor_ao_idx)
    nao_act = len(act_ao_idx)
    nao_ext = len(ext_ao_idx)

    # print("nao     = %d" % nao)
    # print("nao_cor = %d" % nao_cor)
    # print("nao_act = %d" % nao_act)
    # print("nao_ext = %d" % nao_ext)

    ene_uhf_list    = []
    dm_uhf_ao_list  = []
    coeff_uhf_list  = []

    act_idx_alph_list = list(combinations(act_ao_idx, nelec_alph - nao_cor))
    for act_idx_alph in act_idx_alph_list:
        act_idx_beta = list(set(act_ao_idx) - set(act_idx_alph))

        dm0 = numpy.zeros((2, nao, nao))
        dm0[0][cor_ao_idx, cor_ao_idx] = 1.0
        dm0[1][cor_ao_idx, cor_ao_idx] = 1.0
        dm0[0][act_idx_alph, act_idx_alph] = 1.0
        dm0[1][act_idx_beta, act_idx_beta] = 1.0

        dm0_ao = numpy.einsum("mp,nq,spq->smn", coeff_ao_lo, coeff_ao_lo, dm0)
        
        uhf_obj = scf.UHF(mol)
        uhf_obj.verbose = 0
        uhf_obj.max_cycle = 100
        uhf_obj.conv_tol = 1e-12
        uhf_obj.conv_tol_grad = 1e-12
        uhf_obj.kernel(dm0_ao)

        fock_ao = uhf_obj.get_fock(dm=dm0_ao)
        print()
        print([ao_labels[i] for i in act_idx_alph])
        print([ao_labels[i] for i in act_idx_beta])
        print(numpy.diag(fock_ao[0]))
        print(numpy.diag(fock_ao[1]))
        fock_lo = numpy.einsum("mp,nq,smn->spq", coeff_ao_lo, coeff_ao_lo, fock_ao)
        print(numpy.diag(fock_lo[0]))
        print(numpy.diag(fock_lo[1]))
        fd_a = numpy.diag(fock_lo[0])
        fo_a = fd_a[list(cor_ao_idx)+list(act_idx_alph)]
        fv_a = fd_a[list(act_idx_beta)+list(ext_ao_idx)]
        print(fv_a - fo_a[:,None])
        print(numpy.min(fv_a - fo_a[:,None]))

        fo_fv_a = fv_a - fo_a[:,None]

    ovlp_ao = mol.intor("int1e_ovlp")
    ene_uhf_list = numpy.array(ene_uhf_list)
    tmp = [f"{ene:12.8f}" for ene in numpy.sort(ene_uhf_list-numpy.min(ene_uhf_list))]
    print(", ".join([f"{ii:s}: {len(list(ee)):2d}" for ii, ee in groupby(tmp)]))

    coeff_uhf = coeff_uhf_list[numpy.argmin(ene_uhf_list)]
    coeff_uhf_alph_occ = coeff_uhf[0][:, :nelec_alph]
    coeff_uhf_beta_occ = coeff_uhf[1][:, :nelec_beta]

    dm_uhf_ao = dm_uhf_ao_list[numpy.argmin(ene_uhf_list)]
    dm_uhf_lo = [reduce(numpy.dot, (coeff_ao_lo.T, ovlp_ao, dm_uhf_ao[s], ovlp_ao, coeff_ao_lo)) for s in range(2)]
    tmp = scipy.linalg.eigh(dm_uhf_lo[0] + dm_uhf_lo[1])
    occ_uno   = tmp[0][::-1]
    coeff_uno = numpy.dot(coeff_ao_lo, tmp[1])[:, ::-1]
    csc = reduce(numpy.dot, (coeff_uno.T, ovlp_ao, coeff_uno))
    assert numpy.linalg.norm(csc - numpy.eye(nlo)) < 1e-8
    
    mask_cor = numpy.abs(occ_uno - 2.0) < 1e-2
    coeff_uno_cor = coeff_uno[:, mask_cor]
    dump_rec(stdout, coeff_uno_cor)
    
    mask_act = numpy.abs(occ_uno - 1.0) < 1e-2
    coeff_uno_act = coeff_uno[:, mask_act]
    norb_act = coeff_uno_act.shape[1]
    dump_rec(stdout, coeff_uno_act)
    assert 1 == 2

    mask_ext = numpy.abs(occ_uno - 0.0) < 1e-2
    coeff_uno_ext = coeff_uno[:, mask_ext]

    ene_bs_list = []
    dm_bs_list  = []

    act_idx_alph_list =list(combinations(range(norb_act), norb_act // 2))
    for act_idx_alph in act_idx_alph_list:
        act_idx_beta = list(set(range(norb_act)) - set(act_idx_alph))

        coeff_alph_occ = numpy.hstack((coeff_uno_cor, coeff_uno_act[:, act_idx_alph]))
        coeff_beta_occ = numpy.hstack((coeff_uno_cor, coeff_uno_act[:, act_idx_beta]))
        coeff_alph_vir = coeff_uno_ext
        coeff_beta_vir = coeff_uno_ext

        csc_alph = reduce(numpy.dot, (coeff_alph_occ.T, ovlp_ao, coeff_uhf_alph_occ))
        csc_beta = reduce(numpy.dot, (coeff_beta_occ.T, ovlp_ao, coeff_uhf_beta_occ))
        dump_rec(stdout, csc_alph)
        dump_rec(stdout, csc_beta)

        dm_bs_alph = numpy.dot(coeff_alph_occ, coeff_alph_occ.T)
        dm_bs_beta = numpy.dot(coeff_beta_occ, coeff_beta_occ.T)
        dm_bs = numpy.array([dm_bs_alph, dm_bs_beta])
        fock_bs_ao = uhf_obj.get_fock(dm=dm_bs) 

        coeff_alph = numpy.hstack((coeff_alph_occ, coeff_alph_vir))
        coeff_beta = numpy.hstack((coeff_beta_occ, coeff_beta_vir))
        coeff_bs   = [coeff_alph, coeff_beta]

        ovlp_bs_mo = numpy.einsum("smp,snq,mn->spq", coeff_bs, coeff_bs, ovlp_ao)
        fock_bs_mo = numpy.einsum("smp,snq,smn->spq", coeff_bs, coeff_bs, fock_bs_ao)
        assert numpy.linalg.norm(ovlp_bs_mo[0] - numpy.eye(ovlp_bs_mo[0].shape[0])) < 1e-8
        assert numpy.linalg.norm(ovlp_bs_mo[1] - numpy.eye(ovlp_bs_mo[1].shape[0])) < 1e-8

        ene_bs = uhf_obj.energy_elec(dm=dm_bs)[0]
        ene_bs_list.append(ene_bs)

    ene_bs_list = numpy.array(ene_bs_list)
    tmp = [f"{ene:12.8f}" for ene in numpy.sort(ene_bs_list-numpy.min(ene_uhf_list))]
    print(", ".join([f"{ii:s}: {len(list(ee)):2d}" for ii, ee in groupby(tmp)]))

if __name__ == "__main__":
    # bs_uhf_n2_uno(x=1.0, spin=0, basis="321g")
    # bs_uhf_n2_uno(x=2.0, spin=0, basis="321g")
    # bs_uhf_n2_uno(x=3.0, spin=0, basis="321g")
    bs_uhf_n2_uno(x=1.0, spin=0, basis="sto3g")

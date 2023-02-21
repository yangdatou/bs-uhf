from functools import reduce

import os, sys
from sys import stdout
import numpy, scipy
from scipy.linalg import sqrtm

import pyscf
from pyscf import gto, scf, lo
from pyscf.tools.dump_mat import dump_rec

def localize_mo(mo, mol_obj=None, ovlp_ao=None, method="iao"):
    if method == "iao":
        c = lo.iao.iao(mol_obj, mo)
        c = lo.vec_lowdin(c, ovlp_ao)
        mo = reduce(numpy.dot, (c.T, ovlp_ao, mo))
        print(mo.shape)
        print(c.shape)

    if method == "boys":
        c = lo.Boys(mol_obj, mo).kernel()
        mo = c
    else:
        raise NotImplementedError
    
    return mo

def analyze_weight(mos, mol_obj, w_ao, label_list, tol=1e-3):
    nao, nmo = mos.shape
    assert w_ao.shape == (nao, nao)

    w_mo  = numpy.einsum("mn,mp->np", w_ao, mos)
    w2_mo = numpy.einsum("np,np->np", w_mo, w_mo)
    assert w2_mo.shape == (nao, nmo)

    idx_list = []
    for label in label_list:
        idx_list.append(mol_obj.search_ao_label(label))

    mo_label = []

    label_str = ", ".join(["%8s" % x for x in label_list])
    # print("LO weight", label_str)
    for i in range(nmo):
        assert abs(1.0 - sum(w2_mo[:, i])) < 1e-8
        lo_w2 = [sum(w2_mo[idx, i]) for idx in idx_list]
        lo_w2_str = ", ".join(["%8.4f" % x for x in lo_w2])
        # print("MO %5d: %s" % (i, lo_w2_str))
        
        mo_idx = numpy.argmax(lo_w2)
        mo_label.append((i, label_list[mo_idx], lo_w2[mo_idx]))

    return mo_label

def analyze_weight2(mos, mol_obj, w_ao, label_list, tol=1e-3):
    nao, nmo = mos.shape
    assert w_ao.shape == (nao, nao)

    w_mo  = numpy.einsum("mn,mp->np", w_ao, mos)
    w2_mo = numpy.einsum("np,np->np", w_mo, w_mo)
    assert w2_mo.shape == (nao, nmo)

    mo_list  = []

    for label in label_list:
        ao_idx = mol_obj.search_ao_label(label)
        w2_ao  = numpy.einsum("np->p", w2_mo[ao_idx, :])
        mo_idx = numpy.argsort(w2_ao)[::-1]

    return mo_label

def analyze_bs_mo(mos, mol_obj, w_ao, tol=1e-3):
    nao, nmo = mos.shape
    assert w_ao.shape == (nao, nao)

    w_mo  = numpy.einsum("mn,mp->np", w_ao, mos)
    w2_mo = numpy.einsum("np,np->np", w_mo, w_mo)
    assert w2_mo.shape == (nao, nmo)

    for ia, tmp  in enumerate(mol_obj.aoslice_by_atom):
        print("Atom %d" % ia)
        print(tmp)


def solve_n2_rohf(x=1.0, spin=0, basis="ccpvdz"):
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = 'n2-rohf.out'
    mol.atom = f"""
        N 0.00000 0.00000 0.00000
        N 0.00000 0.00000 {x:12.8f}
    """ 
    mol.basis = basis
    mol.spin = spin
    mol.build()

    ao_idx = list(range(mol.nao_nr()))
    alph_occ_idx = []
    alph_occ_idx += list(mol.search_ao_label("N 1s"))
    alph_occ_idx += list(mol.search_ao_label("N 2s"))
    alph_occ_idx += list(mol.search_ao_label("0 N 2p"))
    alph_vir_idx  = list(set(ao_idx) - set(alph_occ_idx))

    beta_occ_idx = []
    beta_occ_idx += list(mol.search_ao_label("N 1s"))
    beta_occ_idx += list(mol.search_ao_label("N 2s"))
    beta_occ_idx += list(mol.search_ao_label("1 N 2p"))
    beta_vir_idx  = list(set(ao_idx) - set(beta_occ_idx))
    nelec_alph = mol.nelec[0]
    nelec_beta = mol.nelec[1]

    assert len(alph_occ_idx) == nelec_alph
    assert len(beta_occ_idx) == nelec_beta

    dm0 = numpy.zeros((2, mol.nao_nr(), mol.nao_nr()))
    dm0[0,alph_occ_idx,alph_occ_idx] = 1.0
    dm0[1,beta_occ_idx,beta_occ_idx] = 1.0

    rhf_obj = scf.RHF(mol)
    rhf_obj.kernel(dm0=None)

    uhf_obj = scf.UHF(mol)
    fock_ao_uhf = uhf_obj.get_fock(dm=dm0)
    ovlp_ao     = uhf_obj.get_ovlp()

    mo_energy, mo_coeff = uhf_obj.eig(fock_ao_uhf, ovlp_ao)
    mo_occ = uhf_obj.get_occ(mo_energy, mo_coeff)

    coeff_alph_occ = mo_coeff[0][:, mo_occ[0] > 0]
    coeff_alph_vir = mo_coeff[0][:, mo_occ[0] == 0]
    coeff_beta_occ = mo_coeff[1][:, mo_occ[1] > 0]
    coeff_beta_vir = mo_coeff[1][:, mo_occ[1] == 0]

    label_list = ["0 N", "1 N"]

    w_ao = sqrtm(ovlp_ao)    
    mo_label = analyze_weight(mo_coeff[0], mol, w_ao, label_list, tol=1e-3)
    # ene_list = [mo_energy[0][lo_dict[x]] if lo_dict.get(x) else None for x in label_list]
    
    n_2s_2pz_ene_list = []
    n0_2px_ene_list   = []
    n0_2py_ene_list   = []
    n1_2px_ene_list   = []
    n1_2py_ene_list   = []

    for i, label, weight in mo_label:
        if "N 2s" in label:
            n_2s_2pz_ene_list.append(mo_energy[0][i])
        elif "0 N 2px" in label:
            n0_2px_ene_list.append(mo_energy[0][i])
        elif "0 N 2py" in label:
            n0_2py_ene_list.append(mo_energy[0][i])
        elif "1 N 2px" in label:
            n1_2px_ene_list.append(mo_energy[0][i])
        elif "1 N 2py" in label:
            n1_2py_ene_list.append(mo_energy[0][i])

    assert len(n0_2px_ene_list) == 1
    assert len(n0_2py_ene_list) == 1
    assert len(n1_2px_ene_list) == 1
    assert len(n1_2py_ene_list) == 1

    # if len(n_2s_2pz_ene_list) == 2:
    #     print(f"{x:12.8}, {n_2s_2pz_ene_list[0]:12.8f}, {n_2s_2pz_ene_list[1]:12.8f}, {n0_2px_ene_list[0]:12.8f}, {n0_2py_ene_list[0]:12.8f}, {n1_2px_ene_list[0]:12.8f}, {n1_2py_ene_list[0]:12.8f}")
    # elif len(n_2s_2pz_ene_list) == 1:
    #     print(f"{x:12.8}, {n_2s_2pz_ene_list[0]:12.8f}, {n_2s_2pz_ene_list[0]:12.8f}, {n0_2px_ene_list[0]:12.8f}, {n0_2py_ene_list[0]:12.8f}, {n1_2px_ene_list[0]:12.8f}, {n1_2py_ene_list[0]:12.8f}")


    # if len(n2s_ene_list) == 2:
    #     print(f"{x:12.8}, {n2s_ene_list[0]:12.8f}, {n2s_ene_list[1]:12.6f}, {n0_2p_ene_list[0]:12.8f}, {n1_2p_ene_list[0]:12.8f}")
    # else:
    #     print(len(n0_2pz_ene_list))
    #     print(len(n1_2pz_ene_list))
    #     print(f"{x:12.8}, {n2s_ene_list[0]:12.8f}, {'None':12s}, {n0_2p_ene_list[0]:12.8f}, {n1_2p_ene_list[0]:12.8f}")
    # # print(n2s_ene_list)
    

    # assert 1 == 2
    print("\nMO label, weight, energy")
    for i, label, weight in mo_label:
        print(f"MO {i:5d}: {label:8s} {weight:8.4f} {mo_energy[0][i]:8.4f}")

    # label_str = ", ".join(["%8s" % x for x in label_list])
    # print("\nLO weight", label_str)
    # print("MO energy", ", ".join(["%8.4f" % x if x else "%8s"%"None" for x in ene_list]))

    # print("\ncoeff_alph_vir")
    # analyze_weight(coeff_alph_vir, mol, w_ao, label_list, tol=1e-3)

    # print("\ncoeff_beta_occ")
    # analyze_weight(coeff_beta_occ, mol, w_ao, label_list, tol=1e-3)

    # print("\ncoeff_beta_vir")
    # analyze_weight(coeff_beta_vir, mol, w_ao, label_list, tol=1e-3)

    # coeff_alph_occ_lo = localize_mo(coeff_alph_occ, mol_obj=mol, ovlp_ao=ovlp_ao, method="boys")
    # coeff_alph_vir_lo = localize_mo(coeff_alph_vir, mol_obj=mol, ovlp_ao=ovlp_ao, method="boys")
    # coeff_beta_occ_lo = localize_mo(coeff_beta_occ, mol_obj=mol, ovlp_ao=ovlp_ao, method="boys")
    # coeff_beta_vir_lo = localize_mo(coeff_beta_vir, mol_obj=mol, ovlp_ao=ovlp_ao, method="boys")

    # foo = numpy.einsum("mp,nq,mn->pq", coeff_alph_occ_lo, coeff_alph_occ_lo, fock_ao_uhf[0])
    # fvv = numpy.einsum("mp,nq,mn->pq", coeff_alph_vir_lo, coeff_alph_vir_lo, fock_ao_uhf[0])
    # print("foo = \n", foo)
    # print("fvv = \n", fvv)
    
    # assert 1 == 2


if __name__ == "__main__":
    for x in numpy.arange(0.4, 3.0, 0.1):
        solve_n2_rohf(x=x, spin=0, basis="sto3g")
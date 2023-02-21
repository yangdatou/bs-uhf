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

def analyze_bs_mo(mos, mol_obj, tol=1e-3):
    nao, nmo = mos.shape
    
    ovlp_ao = mol_obj.intor("int1e_ovlp")
    w_ao    = sqrtm(ovlp_ao)
    assert w_ao.shape == (nao, nao)

    w_ao_mo  = numpy.einsum("mn,mp->np", w_ao, mos)
    w2_ao_mo = numpy.einsum("np,np->np", w_ao_mo, w_ao_mo)
    assert w2_ao_mo.shape == (nao, nmo)

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

    analyze_bs_mo(mo_coeff[0], mol)
    analyze_bs_mo(mo_coeff[1], mol)

    


if __name__ == "__main__":
    for x in numpy.arange(0.4, 3.0, 0.1):
        solve_n2_rohf(x=x, spin=0, basis="sto3g")
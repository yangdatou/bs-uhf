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

    elif method == "boys":
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

def pick_bs_orbital(mos, mol_obj, bs_ao_idx=None, tol_bs=0.1, tol_w=0.1, max_bs_orb=6):
    """
    Pick the broken symmetry MOs from the given MOs.
        - calculate the weight of AO on each MO
        - obatin the AO idx of the given label
        - calculate the weight of the given AO of MOs on each atom
        - pick the MOs with broken symmetry character

    Args:
        mos (numpy.ndarray): MOs
        mol_obj (pyscf.gto.Mole): molecule object
        bs_ao_idx (list): list of AO idx
        tol (float): tolerance for the broken symmetry character
        max_bs_orb (int): maximum number of broken symmetry MOs

    Returns:
        mo_list (list): list of broken symmetry MOs
    """
    nao, nmo = mos.shape
    natm = mol_obj.natm
    
    ovlp_ao = mol_obj.intor("int1e_ovlp")
    w_ao    = sqrtm(ovlp_ao)
    assert w_ao.shape == (nao, nao)

    w_ao_mo  = numpy.einsum("mn,mp->np", w_ao, mos)
    w2_ao_mo = numpy.einsum("np,np->np", w_ao_mo, w_ao_mo)
    assert w2_ao_mo.shape == (nao, nmo)

    w2_mo_atm = []

    for iatm, tmp  in enumerate(mol_obj.aoslice_by_atom()):
        iatm_ao_slice = list(range(tmp[2], tmp[3]))

        ao_idx = []
        for mu in iatm_ao_slice:
            if mu in bs_ao_idx:
                ao_idx.append(mu)

        print("Atom %5d, AO idx: %s" % (iatm, ", ".join([str(x) for x in ao_idx])))
        
        tmp = numpy.einsum("np->p", w2_ao_mo[ao_idx, :])
        w2_mo_atm.append(tmp)

    w2_mo_atm = numpy.array(w2_mo_atm)
    assert w2_mo_atm.shape == (natm, nmo)

    bs_list = []

    for p in range(nmo):
        
        w2_mo_atm_p = w2_mo_atm[:, p]
        w2_mo_atm_p_max = numpy.max(w2_mo_atm_p)
        w2_mo_atm_p_min = numpy.min(w2_mo_atm_p)
        print("MO %5d, w2_mo_atm = %s" % (p, ", ".join(["%8.4f" % x for x in w2_mo_atm_p])))

        if w2_mo_atm_p_max > tol_w and w2_mo_atm_p_max - w2_mo_atm_p_min > tol_bs:
            bs_list.append(p)

        if len(bs_list) >= max_bs_orb:
            break

    return bs_list

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

    ao_idx = list(range(mol.nao_nr()))

    alph_core_idx = []
    alph_core_idx += list(mol.search_ao_label("N 1s"))
    alph_core_idx += list(mol.search_ao_label("N 2s"))

    alph_bs_idx  = []
    alph_bs_idx += list(mol.search_ao_label("0 N 2p"))

    alph_occ_idx = alph_core_idx + alph_bs_idx
    alph_vir_idx = list(set(ao_idx) - set(alph_occ_idx))

    beta_core_idx = []
    beta_core_idx += list(mol.search_ao_label("N 1s"))
    beta_core_idx += list(mol.search_ao_label("N 2s"))

    beta_bs_idx  = []
    beta_bs_idx += list(mol.search_ao_label("1 N 2p"))

    beta_occ_idx = beta_core_idx + beta_bs_idx
    beta_vir_idx = list(set(ao_idx) - set(beta_occ_idx))

    bs_idx = alph_bs_idx + beta_bs_idx

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
    uhf_obj.kernel(dm0=dm0)
    # fock_ao_uhf = uhf_obj.get_fock(dm=dm0)
    # ovlp_ao     = uhf_obj.get_ovlp()

    # mo_energy, mo_coeff = uhf_obj.eig(fock_ao_uhf, ovlp_ao)
    # mo_occ = uhf_obj.get_occ(mo_energy, mo_coeff)

    # print("\n")
    # print("x = %.2f" % x)
    # mo_is_uhf_alph = pick_bs_orbital(mo_coeff[0], mol, bs_ao_idx=bs_idx, tol_bs=0.1, tol_w=0.1, max_bs_orb=10)
    # mo_is_uhf_beta = pick_bs_orbital(mo_coeff[1], mol, bs_ao_idx=bs_idx, tol_bs=0.1, tol_w=0.1, max_bs_orb=10)

    # print("MO is UHF alpha: %s" % ", ".join([str(x) for x in mo_is_uhf_alph]))
    # print("MO is UHF beta:  %s" % ", ".join([str(x) for x in mo_is_uhf_beta]))

    # bs_occ_alph = []
    # bs_occ_beta = []

    # bs_vir_alph = []
    # bs_vir_beta = []

    # nmo = mo_coeff[0].shape[1]
    # nao = mo_coeff[0].shape[0]
    # for p in range(nmo):
    #     if p in mo_is_uhf_alph and p < nelec_alph:
    #         bs_occ_alph.append(p)
    #     elif p in mo_is_uhf_alph and p >= nelec_alph:
    #         bs_vir_alph.append(p)

    #     if p in mo_is_uhf_beta and p < nelec_beta:
    #         bs_occ_beta.append(p)
    #     elif p in mo_is_uhf_beta and p >= nelec_beta:
    #         bs_vir_beta.append(p)

    # coeff_bs_occ_alph = mo_coeff[0][:, bs_occ_alph]
    # coeff_bs_occ_beta = mo_coeff[1][:, bs_occ_beta]
    # coeff_bs_vir_alph = mo_coeff[0][:, bs_vir_alph]
    # coeff_bs_vir_beta = mo_coeff[1][:, bs_vir_beta]

    # print("coeff_bs_occ_alph = ", coeff_bs_occ_alph.shape)
    # print("coeff_bs_occ_beta = ", coeff_bs_occ_beta.shape)
    # print("coeff_bs_vir_alph = ", coeff_bs_vir_alph.shape)
    # print("coeff_bs_vir_beta = ", coeff_bs_vir_beta.shape)

    from pyscf.mcscf import avas
    avas_obj = avas.AVAS(uhf_obj, ["0 N 2p", "1 N 2p"])
    norb_act, nelec_act, coeff_ao_mo = avas_obj.kernel()
    print("norb_act = ", norb_act)
    print("nelec_act = ", nelec_act)
    print("coeff_ao_mo = ", coeff_ao_mo.shape)
    dump_rec(stdout, coeff_ao_mo, mol.ao_labels())

    assert 1 == 2

if __name__ == "__main__":
    for x in numpy.arange(0.4, 3.0, 0.1):
        solve_n2_rohf(x=x, spin=0, basis="sto3g")
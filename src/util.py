from itertools import combinations
from functools import reduce
import os, sys, numpy, scipy
from sys import stdout

from pyscf import gto, scf, lo
from pyscf import fci, mp, ao2mo, ci
from pyscf.lib import chkfile
from pyscf.fci.spin_op import spin_square
from pyscf.fci.direct_spin1 import absorb_h1e, contract_2e

from pyscf.tools.dump_mat import dump_rec
from pyscf.tools import cubegen

from bs import get_coeff_uhf, get_uhf_vfci
from bs import get_ump2_vfci, get_ucisd_vfci
from bs import solve_uhf_noci
from bs import solve_ump2_noci
from bs import solve_ucisd_noci

def get_mol(r, basis="sto-3g", m="h2"):
    if m == "h2":
        atoms  = ""
        atoms += "H 0.0000 0.0000 % 12.8f\n" % ( r / 2.0)
        atoms += "H 0.0000 0.0000 % 12.8f\n" % (-r / 2.0)

    elif m == "h4-line":
        atoms  = ""
        atoms += "H 0.0000 0.0000 % 12.8f\n" % ( 3.0 * r / 2.0)
        atoms += "H 0.0000 0.0000 % 12.8f\n" % (       r / 2.0)
        atoms += "H 0.0000 0.0000 % 12.8f\n" % (     - r / 2.0)
        atoms += "H 0.0000 0.0000 % 12.8f\n" % (-3.0 * r / 2.0)

    elif m == "h4-square":
        atoms = ""
        atoms += "H % 12.8f % 12.8f % 12.8f\n" % ( r / 2.0,  r / 2.0, 0.0)
        atoms += "H % 12.8f % 12.8f % 12.8f\n" % ( r / 2.0, -r / 2.0, 0.0)
        atoms += "H % 12.8f % 12.8f % 12.8f\n" % (-r / 2.0,  r / 2.0, 0.0)
        atoms += "H % 12.8f % 12.8f % 12.8f\n" % (-r / 2.0, -r / 2.0, 0.0)

    elif m == "hub":
        raise NotImplementedError

    elif m == "n2":
        atoms = ""
        atoms += "N 0.0000 0.0000 % 12.8f\n" % ( r / 2.0)
        atoms += "N 0.0000 0.0000 % 12.8f\n" % (-r / 2.0)

    elif m == "h6":
        raise NotImplementedError

    else:
        raise NotImplementedError

    mol = gto.Mole()
    mol.atom    = atoms
    mol.basis   = basis
    mol.verbose = 0
    mol.build()

    coeff_ao_lo = lo.orth_ao(mol, 'meta_lowdin')
    nao, nlo = coeff_ao_lo.shape

    for p in range(nlo):
        max_ao_idx = numpy.argmax(numpy.abs(coeff_ao_lo[:, p]))
        assert max_ao_idx == p

    return mol, coeff_ao_lo

def solve_rhf(mol, dm0=None):
    rhf_obj = scf.RHF(mol)
    rhf_obj.verbose   = 0
    rhf_obj.max_cycle = 500
    rhf_obj.conv_tol  = 1e-12
    rhf_obj.kernel(dm0)
    assert rhf_obj.converged
    return rhf_obj

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

def solve_uhf(mol, dm0=None):
    uhf_obj = scf.UHF(mol)
    uhf_obj.verbose   = 0
    uhf_obj.max_cycle = 500
    uhf_obj.conv_tol  = 1e-12
    uhf_obj.kernel(dm0)
    assert uhf_obj.converged
    return uhf_obj

def get_dm_bs(nao, core_ao_idx, alph_ao_idx, beta_ao_idx):
    dm0 = numpy.zeros((2, nao, nao))

    for i in core_ao_idx:
        dm0[0, i, i] = 1.0
        dm0[1, i, i] = 1.0

    for i in alph_ao_idx:
        dm0[0, i, i] = 1.0

    for i in beta_ao_idx:
        dm0[1, i, i] = 1.0

    return dm0

def get_bs_uhf_ao_label(mol=None, m=None):
    core_ao_idx = []
    alph_ao_idx = []
    beta_ao_idx = []

    if m == "h2":
        alph_ao_idx.append(mol.search_ao_label("0 H 1s")[0])
        beta_ao_idx.append(mol.search_ao_label("1 H 1s")[0])

    elif m == "h4-line":
        alph_ao_idx.append(mol.search_ao_label("0 H 1s")[0])
        alph_ao_idx.append(mol.search_ao_label("3 H 1s")[0])
        beta_ao_idx.append(mol.search_ao_label("1 H 1s")[0])
        beta_ao_idx.append(mol.search_ao_label("2 H 1s")[0])

    elif m == "h4-square":
        alph_ao_idx.append(mol.search_ao_label("0 H 1s")[0])
        alph_ao_idx.append(mol.search_ao_label("3 H 1s")[0])
        beta_ao_idx.append(mol.search_ao_label("1 H 1s")[0])
        beta_ao_idx.append(mol.search_ao_label("2 H 1s")[0])

    elif m == "n2":
        core_ao_idx.append(mol.search_ao_label("0 N 1s")[0])
        core_ao_idx.append(mol.search_ao_label("1 N 1s")[0])
        core_ao_idx.append(mol.search_ao_label("0 N 2s")[0])
        core_ao_idx.append(mol.search_ao_label("1 N 2s")[0])

        alph_ao_idx.append(mol.search_ao_label("0 N 2px")[0])
        alph_ao_idx.append(mol.search_ao_label("0 N 2py")[0])
        alph_ao_idx.append(mol.search_ao_label("0 N 2pz")[0])
        # alph_ao_idx.append(mol.search_ao_label("0 N 2s")[0])
        beta_ao_idx.append(mol.search_ao_label("1 N 2px")[0])
        beta_ao_idx.append(mol.search_ao_label("1 N 2py")[0])
        beta_ao_idx.append(mol.search_ao_label("1 N 2pz")[0])
        # beta_ao_idx.append(mol.search_ao_label("1 N 2s")[0])

    elif m == "hub":
        raise NotImplementedError

    elif m == "h6":
        raise NotImplementedError

    else:
        raise NotImplementedError

    return core_ao_idx, alph_ao_idx, beta_ao_idx

def solve_bs_noci(r, basis="sto-3g", m="h2", is_scf=False, tmp_dir=None):
    res = get_mol(r, m=m, basis=basis)
    mol = res[0]
    coeff_ao_lo = res[1]
    nao, nlo = coeff_ao_lo.shape
    ene_nuc  = mol.energy_nuc()

    nelec_alph, nelec_beta = mol.nelec
    nelec     = nelec_alph + nelec_beta
    
    res = get_bs_uhf_ao_label(mol=mol, m=m)
    core_ao_idx = res[0]
    alph_ao_idx = res[1]
    beta_ao_idx = res[2]
    bs_ao_idx   = alph_ao_idx + beta_ao_idx

    dm0 = get_dm_bs(nao, core_ao_idx, alph_ao_idx, beta_ao_idx)
    uhf_obj   = solve_uhf(mol, dm0=dm0)
    coeff_uhf = uhf_obj.mo_coeff
    dm_uhf    = uhf_obj.make_rdm1()
    ene_uhf   = uhf_obj.energy_elec()[0]
    assert uhf_obj.converged

    coeff_ao_lo = coeff_uhf[0]

    nelec_alph_core = len(core_ao_idx)
    nelec_beta_core = len(core_ao_idx)
    nelec_alph_bs   = 3 # len(bs_ao_idx) // 2
    nelec_beta_bs   = 3 # len(bs_ao_idx) - nelec_alph_bs
    nelec_bs        = nelec_alph_bs + nelec_beta_bs
    assert nelec_alph_core + nelec_alph_bs == nelec_alph
    assert nelec_beta_core + nelec_beta_bs == nelec_beta

    rhf_obj   = solve_rhf(mol, dm0=dm_uhf[0] + dm_uhf[1])
    coeff_rhf = rhf_obj.mo_coeff
    dm_rhf    = rhf_obj.make_rdm1()
    ene_rhf   = rhf_obj.energy_elec()[0]
    assert rhf_obj.converged

    ovlp_ao = rhf_obj.get_ovlp()
    norb = coeff_ao_lo.shape[1]
    h1e  = reduce(numpy.dot, (coeff_ao_lo.conj().T, rhf_obj.get_hcore(), coeff_ao_lo))
    h2e  = ao2mo.kernel(rhf_obj._eri, coeff_ao_lo)
    ham  = absorb_h1e(h1e, h2e, norb, (nelec_alph, nelec_beta), .5)

    def s2_from_fcivec(fcivec):
        return spin_square(fcivec, norb, (nelec_alph, nelec_beta), mo_coeff=coeff_ao_lo, ovlp=ovlp_ao)[0]

    mp2_obj   = mp.RMP2(rhf_obj)
    ene_rmp2  = mp2_obj.kernel()[0] + ene_rhf

    mp2_obj   = mp.UMP2(uhf_obj)
    ene_ump2  = mp2_obj.kernel()[0] + ene_uhf

    cisd_obj  = ci.CISD(rhf_obj)
    cisd_obj.max_cycle = 50000
    cisd_obj.max_space = 1000
    ene_rcisd = cisd_obj.kernel()[0] + ene_rhf

    cisd_obj  = ci.CISD(uhf_obj)
    cisd_obj.max_cycle = 50000
    cisd_obj.max_space = 1000
    ene_ucisd = cisd_obj.kernel()[0] + ene_uhf

    ene_fci   = fci.FCI(rhf_obj).kernel()[0] - ene_nuc

    data_dict = {
        "r"        : r,
        "ene_nuc"  : ene_nuc,
        "ene_rhf"  : ene_rhf,
        "ene_uhf"  : ene_uhf,
        "ene_rmp2" : ene_rmp2,
        "ene_ump2" : ene_ump2,
        "ene_rcisd": ene_rcisd,
        "ene_ucisd": ene_ucisd,
        "ene_fci"  : ene_fci,
        "s2_rhf"   : rhf_obj.spin_square()[0],
        "s2_uhf"   : uhf_obj.spin_square()[0],
    }

    ene_bs_uhf_list    = []
    ene_bs_ump2_list   = []
    ene_bs_ucisd_list  = []

    v_bs_uhf_list      = []
    v_bs_ump2_list     = []
    v_bs_ucisd_list    = []

    hv_bs_uhf_list     = []
    hv_bs_ump2_list    = []
    hv_bs_ucisd_list   = []

    alph_ao_idx_comb = combinations(bs_ao_idx, nelec_alph_bs)

    for idxa, alph_ao_idx in enumerate(alph_ao_idx_comb):
        beta_ao_idx = list(set(bs_ao_idx) - set(alph_ao_idx))
        alph_occ_idx = list(core_ao_idx) + list(alph_ao_idx)
        beta_occ_idx = list(core_ao_idx) + list(beta_ao_idx)
        assert len(alph_occ_idx) == nelec_alph
        assert len(beta_occ_idx) == nelec_beta

        alph_vir_idx = list(set(range(nao)) - set(alph_occ_idx))
        beta_vir_idx = list(set(range(nao)) - set(beta_occ_idx))

        alph_idx = alph_occ_idx + alph_vir_idx
        beta_idx = beta_occ_idx + beta_vir_idx

        coeff_bs_alph = numpy.array(coeff_ao_lo[:, alph_idx])
        coeff_bs_beta = numpy.array(coeff_ao_lo[:, beta_idx])
        coeff_bs = (coeff_bs_alph, coeff_bs_beta)

        mo_occ_alph = [0] * nelec_alph + [1] * (norb - nelec_alph)
        mo_occ_beta = [0] * nelec_beta + [1] * (norb - nelec_beta)
        mo_occ_uhf  = numpy.asarray((mo_occ_alph, mo_occ_beta))

        ene_bs_uhf, vfci_bs_uhf      = get_uhf_vfci(coeff_ao_lo,  coeff_bs, mo_occ_uhf=mo_occ_uhf, ovlp_ao=ovlp_ao, uhf_obj=uhf_obj)
        ene_bs_ump2, vfci_bs_ump2    = get_ump2_vfci(coeff_ao_lo, coeff_bs, mo_occ_uhf=mo_occ_uhf, ovlp_ao=ovlp_ao, uhf_obj=uhf_obj)
        # ene_bs_ucisd, vfci_bs_ucisd  = get_ucisd_vfci(coeff_rhf, coeff_bs_uhf, uhf_obj=uhf_obj)

        ene_bs_uhf_list.append(ene_bs_uhf)
        ene_bs_ump2_list.append(ene_bs_ump2)
        # ene_bs_ucisd_list.append(ene_bs_ucisd)

        v_bs_uhf_list.append(vfci_bs_uhf)
        v_bs_ump2_list.append(vfci_bs_ump2)
        # v_bs_ucisd_list.append(vfci_bs_ucisd)

        hv_bs_uhf_list.append(contract_2e(ham, vfci_bs_uhf, norb, (nelec_alph, nelec_beta)))
        hv_bs_ump2_list.append(contract_2e(ham, vfci_bs_ump2, norb, (nelec_alph, nelec_beta)))
        # hv_bs_ucisd_list.append(contract_2e(ham, vfci_bs_ucisd, norb, (nelec_alph, nelec_beta)))

        idxb = idxa
        data_dict["ene_bs_uhf_%s_%s" % (idxa, idxb)]  = ene_bs_uhf
        data_dict["ene_bs_ump2_%s_%s" % (idxa, idxb)] = ene_bs_ump2
        # data_dict["ene_bs_ucisd_%s" % idx] = ene_bs_ucisd

        data_dict["s2_bs_uhf_%s_%s" % (idxa, idxb)]   = s2_from_fcivec(vfci_bs_uhf)
        data_dict["s2_bs_ump2_%s_%s" % (idxa, idxb)]  = s2_from_fcivec(vfci_bs_ump2)
        # data_dict["s2_bs_ucisd_%s" % idx]  = s2_from_fcivec(vfci_bs_ucisd)

    ene_noci_uhf, vfci_noci_uhf        = solve_uhf_noci(v_bs_uhf_list,  hv_bs_uhf_list, ene_bs_uhf_list, tol=1e-8)
    # ene_noci_ump2_1, vfci_noci_ump2_1  = solve_ump2_noci(v_bs_ump2_list, hv_bs_ump2_list, v_bs_uhf_list=v_bs_uhf_list, ene_ump2_list=ene_bs_ump2_list, tol=1e-8, method=1, ref=ene_fci)
    # ene_noci_ump2_2, vfci_noci_ump2_2  = solve_ump2_noci(v_bs_ump2_list, hv_bs_ump2_list, v_bs_uhf_list=v_bs_uhf_list, ene_ump2_list=ene_bs_ump2_list, tol=1e-2, method=2, ref=ene_fci)
    assert 1 == 2
    # ene_noci_ucisd_1 = solve_ucisd_noci(v_bs_ucisd_list, hv_bs_ucisd_list, v_bs_uhf_list=v_bs_uhf_list, ene_ucisd_list=ene_bs_ucisd_list, tol=1e-8, method=1, ref=ene_fci)
    # print("Solve UCISD NOCI with method 2")
    # ene_noci_ucisd_2 = solve_ucisd_noci(v_bs_ucisd_list, hv_bs_ucisd_list, v_bs_uhf_list=v_bs_uhf_list, ene_ucisd_list=ene_bs_ucisd_list, tol=1e-6, method=2, ref=ene_fci)

    data_dict["ene_noci_uhf"]     = ene_noci_uhf
    data_dict["ene_noci_ump2_1"]  = ene_noci_ump2_1
    data_dict["ene_noci_ump2_2"]  = ene_noci_ump2_2
    # data_dict["ene_noci_ucisd_1"] = ene_noci_ucisd_1
    # data_dict["ene_noci_ucisd_2"] = ene_noci_ucisd_2
    
    data_dict["s2_noci_uhf"]     = s2_from_fcivec(vfci_noci_uhf)
    data_dict["s2_noci_ump2_1"]  = s2_from_fcivec(vfci_noci_ump2_1)
    data_dict["s2_noci_ump2_2"]  = s2_from_fcivec(vfci_noci_ump2_2)

    print("%6.4f, %12.6f, %12.6f, %12.6f, %12.6f" % (r, ene_fci, ene_noci_uhf, ene_noci_ump2_1, ene_noci_ump2_2))

    return data_dict

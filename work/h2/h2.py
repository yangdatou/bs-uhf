from itertools import combinations
from functools import reduce
import os, sys, numpy, scipy

from pyscf import gto, scf, fci
from pyscf import mp, ao2mo, ci
from pyscf.lib import chkfile
from pyscf.fci.direct_spin1 import absorb_h1e, contract_2e

from pyscf.tools.dump_mat import dump_rec

from bs import get_coeff_uhf, get_uhf_vfci
from bs import get_ump2_vfci, get_ucisd_vfci
from bs import solve_uhf_noci
from bs import solve_ump2_noci
from bs import solve_ucisd_noci

def get_h2(r, basis="sto-3g"):
    atoms  = ""
    atoms += "H 0.0000 0.0000 % 12.8f\n" % ( r / 2.0)
    atoms += "H 0.0000 0.0000 % 12.8f\n" % (-r / 2.0)

    mol = gto.Mole()
    mol.atom    = atoms
    mol.basis   = basis
    mol.verbose = 0
    mol.build()

    return mol

def get_rhf(mol, dm0=None):
    rhf_obj = scf.RHF(mol)
    rhf_obj.verbose = 0
    rhf_obj.conv_tol = 1e-12
    rhf_obj.kernel(dm0)
    return rhf_obj

def get_bs_uhf(mol):
    nao = mol.nao
    alph_ao_label = ["0 H 1s"]
    beta_ao_label = ["1 H 1s"]
    alph_ao_idx = [mol.search_ao_label(label)[0] for label in alph_ao_label]
    beta_ao_idx = [mol.search_ao_label(label)[0] for label in beta_ao_label]

    alph_ao_idx = numpy.ix_(alph_ao_idx, alph_ao_idx)
    beta_ao_idx = numpy.ix_(beta_ao_idx, beta_ao_idx)
    dm0 = numpy.zeros((2, nao, nao))
    dm0[0, alph_ao_idx] = 1.0
    dm0[1, beta_ao_idx] = 1.0

    uhf_obj = scf.UHF(mol)
    uhf_obj.verbose = 0
    uhf_obj.conv_tol = 1e-12
    uhf_obj.kernel(dm0)

    return uhf_obj

def solve_h2_bs_uhf(r, basis="sto-3g", f=None):
    mol = get_h2(r, basis=basis)
    nelec_alph, nelec_beta = mol.nelec
    nelec     = nelec_alph + nelec_beta
    ene_nuc   = mol.energy_nuc()

    occ_ao_idx = []
    bs_ao_idx  = []
    bs_ao_idx.append(mol.search_ao_label("0 H 1s")[0])
    bs_ao_idx.append(mol.search_ao_label("1 H 1s")[0])
    nelec_alph_occ = len(occ_ao_idx)
    nelec_beta_occ = len(occ_ao_idx)
    nelec_alph_bs  = len(bs_ao_idx) // 2
    nelec_beta_bs  = len(bs_ao_idx) - nelec_alph_bs
    nelec_bs       = nelec_alph_bs + nelec_beta_bs
    assert nelec_alph_occ + nelec_alph_bs == nelec_alph
    assert nelec_beta_occ + nelec_beta_bs == nelec_beta

    uhf_obj   = get_bs_uhf(mol)
    coeff_uhf = uhf_obj.mo_coeff
    dm_uhf    = uhf_obj.make_rdm1()
    ene_uhf   = uhf_obj.energy_elec()[0]
    assert uhf_obj.converged

    rhf_obj  = get_rhf(mol, dm0=dm_uhf[0] + dm_uhf[1])
    coeff_rhf = rhf_obj.mo_coeff
    dm_rhf    = rhf_obj.make_rdm1()
    ene_rhf   = rhf_obj.energy_elec()[0]
    assert rhf_obj.converged

    norb = coeff_rhf.shape[1]
    h1e  = reduce(numpy.dot, (coeff_rhf.conj().T, rhf_obj.get_hcore(), coeff_rhf))
    h2e  = ao2mo.kernel(rhf_obj._eri, coeff_rhf)
    ham  = absorb_h1e(h1e, h2e, norb, (nelec_alph, nelec_beta), .5)

    ene_rmp2  = mp.RMP2(rhf_obj).kernel()[0] + ene_rhf
    ene_ump2  = mp.UMP2(uhf_obj).kernel()[0] + ene_uhf
    ene_rcisd = ci.CISD(rhf_obj).kernel()[0] + ene_rhf
    ene_ucisd = ci.CISD(uhf_obj).kernel()[0] + ene_uhf
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

    alph_idx_comb = combinations(bs_ao_idx, nelec_alph_bs)
    for idx, alph_idx in enumerate(alph_idx_comb):
        dm0 = numpy.zeros((2, mol.nao, mol.nao))
        alph_idx = list(alph_idx)
        beta_idx = list(set(bs_ao_idx) - set(alph_idx))

        for p in occ_ao_idx:
            dm0[0, p, p] = 1.0
            dm0[1, p, p] = 1.0

        for p in alph_idx:
            dm0[0, p, p] = 1.0

        for p in beta_idx:
            dm0[1, p, p] = 1.0

        ene_bs_uhf_ref, coeff_bs_uhf = get_coeff_uhf(uhf_obj, dm0, is_scf=False)
        ene_bs_uhf, vfci_bs_uhf      = get_uhf_vfci(coeff_rhf, coeff_bs_uhf, uhf_obj=uhf_obj)
        ene_bs_ump2, vfci_bs_ump2    = get_ump2_vfci(coeff_rhf, coeff_bs_uhf, uhf_obj=uhf_obj)
        ene_bs_ucisd, vfci_bs_ucisd  = get_ucisd_vfci(coeff_rhf, coeff_bs_uhf, uhf_obj=uhf_obj)

        assert abs(ene_bs_uhf - ene_bs_uhf_ref) < 1e-8

        ene_bs_uhf_list.append(ene_bs_uhf)
        ene_bs_ump2_list.append(ene_bs_ump2)
        ene_bs_ucisd_list.append(ene_bs_ucisd)

        v_bs_uhf_list.append(vfci_bs_uhf)
        v_bs_ump2_list.append(vfci_bs_ump2)
        v_bs_ucisd_list.append(vfci_bs_ucisd)

        hv_bs_uhf_list.append(contract_2e(ham, vfci_bs_uhf, norb, (nelec_alph, nelec_beta)))
        hv_bs_ump2_list.append(contract_2e(ham, vfci_bs_ump2, norb, (nelec_alph, nelec_beta)))
        hv_bs_ucisd_list.append(contract_2e(ham, vfci_bs_ucisd, norb, (nelec_alph, nelec_beta)))

        data_dict["ene_bs_uhf_%s" % idx]   = ene_bs_uhf
        data_dict["ene_bs_ump2_%s" % idx]  = ene_bs_ump2
        data_dict["ene_bs_ucisd_%s" % idx] = ene_bs_ucisd

    ene_noci_uhf     = solve_uhf_noci(v_bs_uhf_list,  hv_bs_uhf_list, ene_bs_uhf_list, tol=1e-8)
    ene_noci_ump2_1  = solve_ump2_noci(v_bs_ump2_list, hv_bs_ump2_list, v_bs_uhf_list=v_bs_uhf_list, ene_ump2_list=ene_bs_ump2_list, tol=1e-8, method=1)
    ene_noci_ump2_2  = solve_ump2_noci(v_bs_ump2_list, hv_bs_ump2_list, v_bs_uhf_list=v_bs_uhf_list, ene_ump2_list=ene_bs_ump2_list, tol=1e-8, method=2)
    ene_noci_ucisd_1 = solve_ucisd_noci(v_bs_ucisd_list, hv_bs_ucisd_list, v_bs_uhf_list=v_bs_uhf_list, ene_ucisd_list=ene_bs_ucisd_list, tol=1e-8)
    ene_noci_ucisd_2 = solve_ucisd_noci(v_bs_ucisd_list, hv_bs_ucisd_list, v_bs_uhf_list=v_bs_uhf_list, ene_ucisd_list=ene_bs_ucisd_list, tol=1e-8, method=2)

    data_dict["ene_noci_uhf"]     = ene_noci_uhf
    data_dict["ene_noci_ump2_1"]  = ene_noci_ump2_1
    data_dict["ene_noci_ump2_2"]  = ene_noci_ump2_2
    data_dict["ene_noci_ucisd_1"] = ene_noci_ucisd_1
    data_dict["ene_noci_ucisd_2"] = ene_noci_ucisd_2

    return data_dict

if __name__ == "__main__":
    basis = "sto-3g"
    dir_path = f"/Users/yangjunjie/work/bs-uhf/data/h2/{basis}"
    h5_path  = f"/Users/yangjunjie/work/bs-uhf/data/h2/{basis}/data.h5"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    if os.path.exists(h5_path):
        os.remove(h5_path)

    for x in numpy.linspace(0.4, 2.0, 40):
        data_dict = solve_h2_bs_uhf(x, basis=basis)
        chkfile.save(h5_path, "%12.8f" % x, data_dict)

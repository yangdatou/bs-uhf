import sys, os, numpy, scipy
from functools import reduce
from scipy import linalg

from itertools import combinations

from pyscf import gto, scf, mp, ci
from pyscf import lo, fci, ao2mo
from pyscf.lib import chkfile
from pyscf.tools import dump_mat
print_mat = lambda x, t: (print(t), dump_mat.dump_rec(sys.stdout, x))

from pyscf.fci.spin_op import spin_square
from pyscf.fci.direct_spin1 import absorb_h1e
from pyscf.fci.direct_spin1 import contract_2e

from bs import get_uhf_vfci
from bs import get_ump2_vfci
from bs import get_ucisd_vfci

from bs import solve_uhf_noci
from bs import solve_ump2_noci
from bs import solve_ucisd_noci

dm_rhf_pre = None

def make_h4(x, basis="sto-3g"):
    '''Make a tetrahedral H4 molecule
    '''

    atom  = "H % 12.8f % 12.8f % 12.8f\n" % ( x, -x,  x)
    atom += "H % 12.8f % 12.8f % 12.8f\n" % ( x,  x, -x)
    atom += "H % 12.8f % 12.8f % 12.8f\n" % (-x,  x,  x)
    atom += "H % 12.8f % 12.8f % 12.8f"   % (-x, -x, -x)

    h4 = gto.Mole()
    h4.atom = atom
    h4.basis = basis
    h4.build()
    coeff_ao_lo = lo.orth_ao(h4, 'meta_lowdin')

    return h4, coeff_ao_lo

def make_dm0(h4):
    nao = h4.nao_nr()
    nelec_alph, nelec_beta = h4.nelec

    # 1s orbital
    h_1s_idx = h4.search_ao_label("H 1s")

    dm0_list = []
    for alph_ao_idx in combinations(h_1s_idx, nelec_alph):
        beta_ao_idx = list(set(h_1s_idx) - set(alph_ao_idx))

        dm0 = numpy.zeros((2, nao, nao))
        dm0[0][alph_ao_idx, alph_ao_idx] = 1.0
        dm0[1][beta_ao_idx, beta_ao_idx] = 1.0
        dm0_list.append(dm0)

    return dm0_list

def solve_h4_bs_noci(x, basis="sto3g"):
    res = make_h4(x, basis)
    h4  = res[0]
    h4.verbose = 0

    uhf_obj  = scf.UHF(h4)
    hcore_ao = uhf_obj.get_hcore()
    ovlp_ao  = uhf_obj.get_ovlp()

    coeff_ao_lo = res[1]
    norb = coeff_ao_lo.shape[1]
    nelec_alph, nelec_beta = h4.nelec
    ndeta  = fci.cistring.num_strings(norb, nelec_alph)
    ndetb  = fci.cistring.num_strings(norb, nelec_beta)

    h1e    = reduce(numpy.dot, (coeff_ao_lo.conj().T, hcore_ao, coeff_ao_lo))
    h2e    = ao2mo.kernel(h4, coeff_ao_lo)
    ham    = absorb_h1e(h1e, h2e, norb, (nelec_alph, nelec_beta), .5)
    ene_fci, v_fci = fci.direct_spin1.kernel(h1e, h2e, norb, (nelec_alph, nelec_beta))

    get_s2 = lambda v: spin_square(v, norb, (nelec_alph, nelec_beta), mo_coeff=coeff_ao_lo, ovlp=ovlp_ao)[0]
    get_hv = lambda v: contract_2e(ham, v, norb, (nelec_alph, nelec_beta))

    method_list = ["uhf", "ump2", "ucisd"]
    func_dict   = {
        "vfci-uhf"  : get_uhf_vfci,
        "vfci-ump2" : get_ump2_vfci,
        "vfci-ucisd": get_ucisd_vfci,
    
        "noci-uhf"  : None,
        "noci-ump2" : solve_ump2_noci,
        "noci-ucisd": solve_ucisd_noci,
    }

    data_dict   = {
        "x" : x, 
        "ene-nuc"  : h4.energy_nuc(),
        "ene-fci"  : ene_fci,
        "v-fci"    : v_fci,
        "s2-fci"   : get_s2(v_fci),
    }

    for m in method_list:
        data_dict[f"ene-{m}-bs"]  = []
        data_dict[f"s2-{m}-bs"]   = []
        data_dict[f"v-{m}-bs"]    = []
        data_dict[f"hv-{m}-bs"]   = []

    dm0_list   = make_dm0(h4)
    dm_bs_list = []
    nbs = len(dm0_list)

    for ibs, dm0 in enumerate(dm0_list):
        uhf_obj.kernel(dm0=dm0)
        dm_uhf = uhf_obj.make_rdm1()
        dm_bs_list.append(dm_uhf)

        coeff_uhf  = uhf_obj.mo_coeff
        ca_o = coeff_uhf[0] # [:, :nelec_alph]
        cb_o = coeff_uhf[1] # [:, :nelec_beta]
        s_ab = reduce(numpy.dot, (ca_o.conj().T, ovlp_ao, cb_o))
        print_mat(s_ab, f"Overlap between alpha and beta orbitals (basis {ibs})")

        mo_occ_uhf = uhf_obj.mo_occ
        args = (coeff_ao_lo, coeff_uhf, mo_occ_uhf, ovlp_ao, uhf_obj)

        for m in method_list:
            ene_bs, v_bs = func_dict[f"vfci-{m}"](*args)
            hv_bs = get_hv(v_bs)
            s2_bs = get_s2(v_bs)

            data_dict[f"ene-{m}-bs"].append(ene_bs)
            data_dict[f"s2-{m}-bs"].append(s2_bs)
            data_dict[f"v-{m}-bs"].append(v_bs)
            data_dict[f"hv-{m}-bs"].append(hv_bs)
    
    rhf_obj = scf.RHF(h4)
    ene_rhf_list = []
    dm_rhf_list  = []

    global dm_rhf_pre

    dm0_list = [dm_rhf_pre] + [dm[0] + dm[1] for dm in dm_bs_list]
    for ibs, dm0 in enumerate(dm0_list):
        rhf_obj.verbose   = 0
        rhf_obj.conv_tol  = 1e-10
        rhf_obj.max_cycle = 500

        if dm0 is not None and dm0.shape == ovlp_ao.shape:
            rhf_obj.kernel(dm0=dm0)
        else:
            rhf_obj.kernel()

        ene_rhf_list.append(rhf_obj.energy_elec()[0])
        dm_rhf_list.append(rhf_obj.make_rdm1())
    
    rhf_min_idx = numpy.argmin(ene_rhf_list)
    rhf_obj.verbose   = 4
    rhf_obj.conv_tol  = 1e-12
    rhf_obj.max_cycle = 2000
    rhf_obj.kernel(dm_rhf_list[rhf_min_idx])
    ene_rhf = rhf_obj.energy_elec()[0]
    assert rhf_obj.converged

    dm_rhf_pre = rhf_obj.make_rdm1()

    rmp2_obj = mp.RMP2(rhf_obj)
    rmp2_obj.kernel()
    ene_rmp2 = rmp2_obj.e_corr + ene_rhf

    rcisd_obj = ci.CISD(rhf_obj)
    rcisd_obj.kernel()
    ene_rcisd = rcisd_obj.e_corr + ene_rhf
    assert rcisd_obj.converged

    data_dict["ene-rhf"]   = ene_rhf
    data_dict["ene-rmp2"]  = ene_rmp2
    data_dict["ene-rcisd"] = ene_rcisd

    v_bs_uhf   = numpy.array(data_dict["v-uhf-bs"])
    hv_bs_uhf  = numpy.array(data_dict["hv-uhf-bs"])
    ene_bs_uhf = numpy.array(data_dict["ene-uhf-bs"])
    s2_bs_uhf  = numpy.array(data_dict["s2-uhf-bs"])
    data_dict["v-uhf-bs"]   = 0 # v_bs_uhf
    data_dict["hv-uhf-bs"]  = 0 # hv_bs_uhf
    data_dict["ene-uhf-bs"] = ene_bs_uhf
    data_dict["s2-uhf-bs"]  = s2_bs_uhf
    args       = (v_bs_uhf,  hv_bs_uhf, ene_bs_uhf)
    
    assert v_bs_uhf.shape   == (nbs, ndeta, ndetb)
    assert hv_bs_uhf.shape  == (nbs, ndeta, ndetb)
    assert ene_bs_uhf.shape == (nbs,)

    tmp = solve_uhf_noci(*args, tol=1e-8)
    ene_noci_uhf  = tmp[0]
    v_noci_uhf    = tmp[1]
    data_dict["v-noci-uhf"]   = 0 # v_noci_uhf
    data_dict["ene-noci-uhf"] = ene_noci_uhf
    data_dict["s2-noci-uhf"]  = get_s2(v_noci_uhf)

    for m in method_list:
        if m == "uhf":
            continue

        v_bs   = numpy.array(data_dict[f"v-{m}-bs"])
        hv_bs  = numpy.array(data_dict[f"hv-{m}-bs"])
        ene_bs = numpy.array(data_dict[f"ene-{m}-bs"])
        s2_bs  = numpy.array(data_dict[f"s2-{m}-bs"])
        data_dict[f"v-{m}-bs"]    = 0 # v_bs
        data_dict[f"hv-{m}-bs"]   = 0 # hv_bs
        data_dict[f"ene-{m}-bs"]  = ene_bs
        data_dict[f"s2-{m}-bs"]   = s2_bs
        args = (v_bs, hv_bs, v_bs_uhf, ene_bs)

        assert v_bs.shape   == (nbs, ndeta, ndetb)
        assert hv_bs.shape  == (nbs, ndeta, ndetb)
        assert ene_bs.shape == (nbs,)

        tmp = func_dict[f"noci-{m}"](*args, tol=1e-4, method=1, ref=None)
        ene_noci_1 = tmp[0]
        v_noci_1   = tmp[1]
        data_dict[f"v-noci-{m}-1"]     = 0 # v_noci_1
        data_dict[f"ene-noci-{m}-1"]   = ene_noci_1
        data_dict[f"s2-noci-{m}-1"]    = get_s2(v_noci_1)

        tmp = func_dict[f"noci-{m}"](*args, tol=1e-4, method=2, ref=None)
        ene_noci_2 = tmp[0]
        v_noci_2   = tmp[1]
        data_dict[f"v-noci-{m}-2"]     = 0 # v_noci_2
        data_dict[f"ene-noci-{m}-2"]   = ene_noci_2
        data_dict[f"s2-noci-{m}-2"]    = get_s2(v_noci_2)

    print(f"x = {x: 6.4f}, {ene_fci: 16.12f}")

    return data_dict

if __name__ == "__main__":
    m        = "h4-td"
    basis    = "sto-3g"
    dir_path = f"../data/"
    tmp_dir  = dir_path + f"/{m}-{basis}/"
    h5_path  = os.path.join(dir_path, f"{m}-{basis}.h5")

    # if not os.path.exists(tmp_dir):
    #     os.makedirs(tmp_dir, exist_ok=True)

    # if os.path.exists(h5_path):
    #     os.remove(h5_path)

    print("\n")
    print("#" * 20)
    print("m        = %s" % m)
    print("basis    = %s" % basis)
    print("dir_path = %s" % dir_path)
    print("tmp_dir  = %s" % tmp_dir)
    print("h5_path  = %s" % h5_path)

    for x in numpy.linspace(0.5, 1.5, 41):
        data_dict = solve_h4_bs_noci(x, basis=basis)
        assert 1 == 2
        # chkfile.save(h5_path, "%.8f" % x, data_dict)

    # basis    = "cc-pvdz"
    # tmp_dir  = dir_path + f"/{m}-{basis}/"
    # h5_path  = os.path.join(dir_path, f"{m}-{basis}.h5")

    # if not os.path.exists(tmp_dir):
    #     os.makedirs(tmp_dir, exist_ok=True)

    # if os.path.exists(h5_path):
    #     os.remove(h5_path)

    # print("\n")
    # print("#" * 20)
    # print("m        = %s" % m)
    # print("basis    = %s" % basis)
    # print("dir_path = %s" % dir_path)
    # print("tmp_dir  = %s" % tmp_dir)
    # print("h5_path  = %s" % h5_path)

    # for x in numpy.linspace(0.4, 1.5, 41):
    #     data_dict = solve_h4_bs_noci(x, basis=basis)
    #     chkfile.save(h5_path, "%.8f" % x, data_dict)

from functools import reduce
import os, sys, numpy, scipy
from numpy import linalg

from pyscf import gto, scf, fci, mp, ao2mo, ci
from pyscf.tools.dump_mat import dump_rec
from pyscf.tools import cubegen
from pyscf.fci import cistring, spin_op, addons
from pyscf.ci  import ucisd
from pyscf.lib import chkfile
from pyscf.lib.logger import perf_counter, process_clock

def proj_ump2_to_fci_vec(coeff_rhf, coeff_uhf, 
                         t0=None, t1=None, t2=None, 
                         nelec=None, ovlp_ao=None):

    nao, norb = coeff_rhf.shape
    coeff_uhf = numpy.asarray(coeff_uhf)
    nelec_alph, nelec_beta = nelec
    
    assert coeff_rhf.shape == (nao, norb)
    assert coeff_uhf.shape == (2, nao, norb)
    assert ovlp_ao.shape   == (nao, nao)

    nocc_a = nelec_alph
    nocc_b = nelec_beta
    nvir_a = norb - nocc_a
    nvir_b = norb - nocc_b

    if t0 is None:
        t0 = 1.0

    if t1 is None:
        t1_a = numpy.zeros((nocc_a, nvir_a))
        t1_b = numpy.zeros((nocc_b, nvir_b))

    ovlp_rhf_uhf = numpy.einsum("smp,nq,mn->spq", coeff_uhf, coeff_rhf, ovlp_ao)
    vec_cisd = ucisd.amplitudes_to_cisdvec(t0, (t1_a, t1_b), t2)
    vec_fci  = ucisd.to_fcivec(vec_cisd, norb, nelec)
    return addons.transform_ci_for_orbital_rotation(vec_fci, norb, nelec, ovlp_rhf_uhf)

def proj_uhf_to_fci_vec(coeff_rhf, coeff_uhf, nelec=None, ovlp_ao=None):
    nao, norb = coeff_rhf.shape
    coeff_uhf = numpy.asarray(coeff_uhf)
    nelec_alph, nelec_beta = nelec
    
    assert coeff_rhf.shape == (nao, norb)
    assert coeff_uhf.shape == (2, nao, norb)
    assert ovlp_ao.shape   == (nao, nao)

    ovlp_rhf_uhf = numpy.einsum("smp,nq,mn->spq", coeff_uhf, coeff_rhf, ovlp_ao)

    na = cistring.num_strings(norb, nelec_alph)
    nb = cistring.num_strings(norb, nelec_beta)
    vec_fci = numpy.zeros((na, nb))
    vec_fci[0, 0] = 1.0

    return addons.transform_ci_for_orbital_rotation(vec_fci, norb, nelec, ovlp_rhf_uhf)

def solve_n2_bs_uhf(r, basis="sto-3g", f=None,
                    alph_ao_labels=None, 
                    beta_ao_labels=None):
    mol = gto.Mole()
    mol.atom = f"""
    N1 0.000 0.000 {( r/2.0): 12.8f}
    N2 0.000 0.000 {(-r/2.0): 12.8f}
    """
    mol.basis = basis
    mol.build()

    nelec = mol.nelec

    rhf_obj = scf.RHF(mol)
    rhf_obj.verbose = 0
    rhf_obj.kernel(dm0=None)
    assert rhf_obj.converged

    ovlp_ao   = rhf_obj.get_ovlp()
    coeff_rhf = rhf_obj.mo_coeff
    nao, norb = coeff_rhf.shape
    dm_rhf = rhf_obj.make_rdm1()

    fci_obj = fci.FCI(mol, mo=rhf_obj.mo_coeff, singlet=False)
    fci_obj.verbose = 0
    ene_fci = fci_obj.kernel()[0]

    h1e = reduce(numpy.dot, (coeff_rhf.conj().T, rhf_obj.get_hcore(), coeff_rhf))
    h2e = ao2mo.kernel(rhf_obj._eri, coeff_rhf)
    from pyscf.fci.direct_spin1 import absorb_h1e, contract_2e
    ham = absorb_h1e(h1e, h2e, norb, nelec, 0.5)

    data_dict = {
        "r": r, "ene_nuc": mol.energy_nuc(),
        "RHF": rhf_obj.energy_elec()[0],
        "FCI": ene_fci - mol.energy_nuc(),
    }

    uhf_obj = scf.UHF(mol)
    uhf_obj.verbose = 0

    alph_ao_label_list = [[], ["N1 2pz"], ["N1 2px", "N1 2py"]]
    beta_ao_label_list = [[], ["N2 2pz"], ["N2 2px", "N2 2py"]]

    vhf_list   = []
    vmp2_list  = []
    hvmp2_list = []

    for i, (alph_ao_label, beta_ao_label) in enumerate(zip(alph_ao_label_list, beta_ao_label_list)):
        dms_bs = numpy.zeros((2, mol.nao, mol.nao))
        alph_ao_idx = mol.search_ao_label(alph_ao_label)
        beta_ao_idx = mol.search_ao_label(beta_ao_label)

        dms_bs[0] = dm_rhf / 2.0
        dms_bs[1] = dm_rhf / 2.0

        dms_bs[0, alph_ao_idx, alph_ao_idx] = 1.0
        dms_bs[1, beta_ao_idx, beta_ao_idx] = 1.0
        dms_bs[0, beta_ao_idx, beta_ao_idx] = 0.0
        dms_bs[1, alph_ao_idx, alph_ao_idx] = 0.0

        uhf_obj.kernel(dm0=dms_bs)
        assert uhf_obj.converged

        s2      = uhf_obj.spin_square()[0]
        ene_uhf = uhf_obj.energy_elec()[0]

        mp2_obj = mp.MP2(uhf_obj)
        mp2_obj.verbose = 0

        data_dict[f"BS-UHF-{i}"]    = ene_uhf
        data_dict[f"BS-UHF-{i}-S2"] = s2

        coeff_uhf_alph = uhf_obj.mo_coeff[0]
        coeff_uhf_beta = uhf_obj.mo_coeff[1]

        coeff_uhf_ab = (coeff_uhf_alph, coeff_uhf_beta)
        coeff_uhf_ba = (coeff_uhf_beta, coeff_uhf_alph)

        ene_mp2_corr_ab, t2_uhf_ab = mp2_obj.kernel(mo_coeff=coeff_uhf_ab)
        ene_mp2_corr_ba, t2_uhf_ba = mp2_obj.kernel(mo_coeff=coeff_uhf_ba)

        assert abs(ene_mp2_corr_ab - ene_mp2_corr_ba) < 1e-8
        ene_mp2 = ene_mp2_corr_ab + ene_uhf
        data_dict[f"BS-UMP2-{i}"] = ene_mp2

        vfci_uhf_ab = proj_uhf_to_fci_vec(coeff_rhf,  coeff_uhf_ab, nelec=nelec, ovlp_ao=ovlp_ao)
        vfci_uhf_ba = proj_uhf_to_fci_vec(coeff_rhf,  coeff_uhf_ba, nelec=nelec, ovlp_ao=ovlp_ao)

        vfci_ump2_ab = proj_ump2_to_fci_vec(coeff_rhf, coeff_uhf_ab, nelec=nelec, t2=t2_uhf_ab, ovlp_ao=ovlp_ao)
        vfci_ump2_ba = proj_ump2_to_fci_vec(coeff_rhf, coeff_uhf_ba, nelec=nelec, t2=t2_uhf_ba, ovlp_ao=ovlp_ao)
    
        vhf   = [vfci_uhf_ab,  vfci_uhf_ba]
        vmp2  = [vfci_ump2_ab, vfci_ump2_ba]
        hvmp2 = [contract_2e(ham, v, norb, nelec) for v in vmp2]

        vhf_list   += vhf
        vmp2_list  += vmp2
        hvmp2_list += hvmp2

        for iv, v in enumerate([vhf, vmp2]):
            v_dot_v  = numpy.einsum("Iij,Jij->IJ", v, vmp2)
            v_dot_hv = numpy.einsum("Iij,Jij->IJ", v, hvmp2)

            if abs(numpy.linalg.det(v_dot_v)) > 1e-8:
                ene_mp2_noci = scipy.linalg.eig(v_dot_hv, v_dot_v)[0]
                ene_mp2_noci = ene_mp2_noci.real.min()

            else:
                ene_mp2_noci = v_dot_hv[0, 0] / v_dot_v[0, 0]
            data_dict[f"BS-UMP2-{i}-NOCI-{iv}"] = ene_mp2_noci

    for iv, v in enumerate([vhf_list, vmp2_list]):
        v_dot_v  = numpy.einsum("Iij,Jij->IJ", v, vmp2_list)
        v_dot_hv = numpy.einsum("Iij,Jij->IJ", v, hvmp2_list)

        # print("v_dot_v = %6.4e " % abs(numpy.linalg.det(v_dot_v)))
        # print("v_dot_v = \n", v_dot_v)
        # if abs(numpy.linalg.det(v_dot_v)) > 1e-8:
        ene_mp2_noci = scipy.linalg.eig(v_dot_hv, v_dot_v)[0]
        ene_mp2_noci = ene_mp2_noci.real
        num_uhf_sol  = v_dot_v.shape[0]
        ene_mp2_noci = [v_dot_hv[i, i] / v_dot_v[i, i] for i in range(num_uhf_sol)]
        ene_mp2_noci = numpy.array(ene_mp2_noci)

        for j, ene in enumerate(ene_mp2_noci):
            data_dict[f"BS-UMP2-NOCI-{iv}-{j}"] = ene

    chkfile.save(f"./data/n2-bs-uhf-{basis}.h5", f"{r}", data_dict)



if __name__ == "__main__":
    basis = "sto-3g"
    with open(f"./data/n2-bs-uhf-{basis}.csv", "w") as f:
        for x in numpy.linspace(0.8, 2.4, 40):
            solve_n2_bs_uhf(x, basis=basis, f = f)
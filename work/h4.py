import itertools
from functools import reduce

import os, sys, numpy, scipy
from sys import stdout
from numpy import linalg

from pyscf import gto, scf, fci, mp, ao2mo, ci
from pyscf.fci import cistring, spin_op, addons
from pyscf.fci.direct_spin1 import absorb_h1e, contract_2e
from pyscf.tools.dump_mat import dump_rec
from pyscf.fci import cistring, spin_op
from pyscf.lib import chkfile
from pyscf.ci  import ucisd

from spin_utils import coeff_rhf_to_ghf, coeff_uhf_to_ghf
from spin_utils import rotate_coeff_ghf, get_spin_avg

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

def proj_ump2_to_fci_vec(coeff_rhf, coeff_uhf, t0=None, t1=None, t2=None, 
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
    else:
        t1_a, t1_b = t1

    ovlp_rhf_uhf = numpy.einsum("smp,nq,mn->spq", coeff_uhf, coeff_rhf, ovlp_ao)
    vec_cisd = ucisd.amplitudes_to_cisdvec(t0, (t1_a, t1_b), t2)
    vec_fci  = ucisd.to_fcivec(vec_cisd, norb, nelec)
    return addons.transform_ci_for_orbital_rotation(vec_fci, norb, nelec, ovlp_rhf_uhf)

def solve_h4_bs_uhf(r, basis="sto-3g", tol=1e-8):
    atoms  = ""
    atoms += "H 0.0000 0.0000 % 12.8f\n" % ( 3.0 * r / 2.0)
    atoms += "H 0.0000 0.0000 % 12.8f\n" % (-3.0 * r / 2.0)
    atoms += "H 0.0000 0.0000 % 12.8f\n" % ( r / 2.0)
    atoms += "H 0.0000 0.0000 % 12.8f\n" % (-r / 2.0)

    mol = gto.Mole()
    mol.atom  = atoms
    mol.basis = basis
    mol.build()

    h1_1s_idx = mol.search_ao_label("0 H 1s")
    h2_1s_idx = mol.search_ao_label("1 H 1s")
    h3_1s_idx = mol.search_ao_label("2 H 1s")
    h4_1s_idx = mol.search_ao_label("3 H 1s")

    rhf_obj = scf.RHF(mol)
    rhf_obj.verbose = 0
    rhf_obj.conv_tol = 1e-12
    rhf_obj.kernel()
    assert rhf_obj.converged

    ene_rhf   = rhf_obj.energy_elec()[0]
    coeff_rhf = rhf_obj.mo_coeff
    ovlp_ao   = rhf_obj.get_ovlp()

    mp2_obj = mp.MP2(rhf_obj)
    mp2_obj.verbose = 0
    mp2_obj.kernel()
    ene_mp2 = mp2_obj.e_tot - mol.energy_nuc()

    fci_obj = fci.FCI(mol, mo=rhf_obj.mo_coeff, singlet=False)
    fci_obj.verbose = 0
    ene_fci = fci_obj.kernel()[0] - mol.energy_nuc()

    vfci_uhf_list   = []
    vfci_ump2_list  = []
    vfci_ucisd_list = []
    hvfci_uhf_list  = []
    hvfci_ump2_list = []
    hvfci_ucisd_list= []
    ene_uhf_list    = []
    ene_ump2_list   = []
    ene_ucisd_list  = []

    norb = coeff_rhf.shape[1]
    nelec_alph, nelec_beta = mol.nelec[0], mol.nelec[1]
    nelec = (nelec_alph, nelec_beta)
    spin_list  = [0 for i in range(nelec_alph)]
    spin_list += [1 for i in range(nelec_beta)]

    from pyscf.fci.direct_spin1 import absorb_h1e, contract_2e
    h1e = reduce(numpy.dot, (coeff_rhf.conj().T, rhf_obj.get_hcore(), coeff_rhf))
    h2e = ao2mo.kernel(rhf_obj._eri, coeff_rhf)
    ham = absorb_h1e(h1e, h2e, norb, nelec, 0.5)

    s1 = 0
    s2 = 1
    s3 = 1
    s4 = 0
    dm0 = numpy.zeros((2, mol.nao, mol.nao))
    dm0[s1, h1_1s_idx, h1_1s_idx] = 1.0
    dm0[s2, h2_1s_idx, h2_1s_idx] = 1.0
    dm0[s3, h3_1s_idx, h3_1s_idx] = 1.0
    dm0[s4, h4_1s_idx, h4_1s_idx] = 1.0

    uhf_obj = scf.UHF(mol)
    uhf_obj.verbose = 0
    uhf_obj.max_cycle = 1200
    uhf_obj.conv_tol = 1e-8
    uhf_obj.diis_start_cycle = 20
    uhf_obj.diis_space = 10
    uhf_obj.kernel(dm0)
    assert uhf_obj.converged
    ene_uhf = uhf_obj.energy_elec()[0]

    ump2_obj = mp.UMP2(uhf_obj)
    ump2_obj.verbose = 0
    ump2_obj.kernel()
    ene_ump2 = ump2_obj.e_tot - mol.energy_nuc()    

    data_dict = {
        "RHF" : ene_rhf, "FCI" : ene_fci,
        "r": r, "ene_nuc": mol.energy_nuc(),
        "MP2": ene_mp2,
        "UHF": ene_uhf, "UMP2": ene_ump2,
    }

    for iidx, alph_idx in enumerate(itertools.combinations(range(nelec_alph + nelec_beta), nelec_alph)):
        alph_idx = list(alph_idx)
        ss_idx   = [0 if i not in alph_idx else 1 for i in range(nelec_alph + nelec_beta)]
        s1, s2, s3, s4 = ss_idx

        dm0 = numpy.zeros((2, mol.nao, mol.nao))
        dm0[s1, h1_1s_idx, h1_1s_idx] = 1.0
        dm0[s2, h2_1s_idx, h2_1s_idx] = 1.0
        dm0[s3, h3_1s_idx, h3_1s_idx] = 1.0
        dm0[s4, h4_1s_idx, h4_1s_idx] = 1.0

        fock = uhf_obj.get_fock(dm=dm0)
        mo_energy, mo_coeff = uhf_obj.eig(fock, ovlp_ao)
        uhf_obj.mo_energy = mo_energy
        uhf_obj.mo_coeff  = mo_coeff
        mo_occ = uhf_obj.get_occ(mo_energy=mo_energy, mo_coeff=mo_coeff)
        uhf_obj.mo_occ = mo_occ
        dm = uhf_obj.make_rdm1(mo_coeff, mo_occ)

        coeff_uhf = uhf_obj.mo_coeff
        ene_uhf   = uhf_obj.energy_elec()[0]

        ucisd_obj = ci.UCISD(uhf_obj)
        ucisd_obj.verbose = 0
        ene_ump2,  ci_ump2  = ucisd_obj.get_init_guess()
        ene_ucisd, ci_ucisd = ucisd_obj.kernel()
        ene_ump2  += ene_uhf
        ene_ucisd += ene_uhf

        vfci_uhf  = proj_uhf_to_fci_vec(coeff_rhf, coeff_uhf, nelec, ovlp_ao)
        t0, t1, t2 = ucisd_obj.cisdvec_to_amplitudes(ci_ump2)
        vfci_ump2 = proj_ump2_to_fci_vec(
            coeff_rhf, coeff_uhf, nelec=nelec, 
            ovlp_ao=ovlp_ao, t0=t0, t1=t1, t2=t2
        )
        t0, t1, t2 = ucisd_obj.cisdvec_to_amplitudes(ci_ucisd)
        vfci_ucisd = proj_ump2_to_fci_vec(
            coeff_rhf, coeff_uhf, nelec=nelec, 
            ovlp_ao=ovlp_ao, t0=t0, t1=t1, t2=t2
        )

        hvfci_uhf   = contract_2e(ham, vfci_uhf,  norb, nelec)
        hvfci_ump2  = contract_2e(ham, vfci_ump2, norb, nelec)
        hvfci_ucisd = contract_2e(ham, vfci_ucisd, norb, nelec)

        data_dict["UHF-%d" % iidx]  = ene_uhf
        data_dict["UMP2-%d" % iidx] = ene_ump2
        data_dict["UCISD-%d" % iidx] = ene_ucisd

        vfci_uhf_list.append(vfci_uhf)
        vfci_ump2_list.append(vfci_ump2)
        vfci_ucisd_list.append(vfci_ucisd)

        hvfci_uhf_list.append(hvfci_uhf)
        hvfci_ump2_list.append(hvfci_ump2)
        hvfci_ucisd_list.append(hvfci_ucisd)

        ene_uhf_list.append(ene_uhf)
        ene_ump2_list.append(ene_ump2)
        ene_ucisd_list.append(ene_ucisd)

    v, hv    = vfci_uhf_list, hvfci_uhf_list 
    v_dot_v  = numpy.einsum("Iij,Jij->IJ", v, v)
    v_dot_hv = numpy.einsum("Iij,Jij->IJ", v, hv)

    # eigvals, eigvecs = scipy.linalg.eigh(v_dot_v)
    # mask = numpy.abs(eigvals) > 1e-8

    # tmp  = "r = %10.6f " % r
    # tmp += "eigvals ="
    # tmp += "".join(["%12.4e, " % i for i in eigvals])[:-2]
    # print(tmp)

    # eigvals = eigvals[mask]
    # eigvecs = eigvecs[:, mask]

    # v_dot_v_  = numpy.einsum("IJ,Ii,Jj->ij", v_dot_v, eigvecs, eigvecs)
    # v_dot_hv_ = numpy.einsum("IJ,Ii,Jj->ij", v_dot_hv, eigvecs, eigvecs)
    det1 = numpy.linalg.det(v_dot_v)
    if numpy.abs(det1) > tol:
        ene_noci  = scipy.linalg.eigh(v_dot_hv, v_dot_v)[0]
        data_dict["UHF-NOCI"] = numpy.min(ene_noci)
    else:
        data_dict["UHF-NOCI"] = numpy.nan

    v_dot_v   = numpy.einsum("Iij,Jij->IJ", vfci_ump2_list, vfci_ump2_list)
    v_dot_hv  = numpy.einsum("Iij,Jij->IJ", vfci_ump2_list, hvfci_ump2_list)
    # v_dot_v_  = numpy.einsum("IJ,Ii,Jj->ij", v_dot_v, eigvecs, eigvecs)
    # v_dot_hv_ = numpy.einsum("IJ,Ii,Jj->ij", v_dot_hv, eigvecs, eigvecs)
    # ene_noci  = scipy.linalg.eig(v_dot_hv_, v_dot_v_)[0]
    # assert numpy.linalg.norm(ene_noci.imag) < 1e-8
    # ene_noci  = ene_noci.real
    # data_dict["UMP2-NOCI-1"] = numpy.min(ene_noci)
    det2 = numpy.linalg.det(v_dot_v)
    if numpy.abs(det2) > tol:
        ene_noci  = scipy.linalg.eigh(v_dot_hv, v_dot_v)[0]
        # assert numpy.linalg.norm(ene_noci.imag) < 1e-8
        # ene_noci  = ene_noci.real
        data_dict["UMP2-NOCI-1"] = numpy.min(ene_noci)
    else:
        data_dict["UMP2-NOCI-1"] = numpy.nan

    # if (data_dict["UMP2-NOCI-1"] - data_dict["UHF-NOCI"]) > 0.1:
    print("r = %10.6f, ene_uhf_noci = %12.8f, ene_ump2_noci = %12.8f, det = % 6.4e % 6.4e" % (r, data_dict["UHF-NOCI"], data_dict["UMP2-NOCI-1"], det1, det2))
        # breakpoint()

    # v, hv     = vfci_ump2_list, hvfci_ump2_list
    # v_dot_v   = numpy.einsum("Iij,Jij->IJ", v, vfci_ump2_list)
    # v_dot_hv  = numpy.einsum("Iij,Jij->IJ", v, hvfci_ump2_list)
    # v_dot_v_  = numpy.einsum("IJ,Ii,Jj->ij", v_dot_v, eigvecs, eigvecs)
    # v_dot_hv_ = numpy.einsum("IJ,Ii,Jj->ij", v_dot_hv, eigvecs, eigvecs)
    # ene_noci  = scipy.linalg.eig(v_dot_hv_, v_dot_v_)[0]
    # assert numpy.linalg.norm(ene_noci.imag) < 1e-8
    # ene_noci  = ene_noci.real
    # data_dict["UMP2-NOCI-2"] = numpy.min(ene_noci)

    os.makedirs(f"/Users/yangjunjie/work/bs-uhf/data/h4", exist_ok=True)
    chkfile.save(f"/Users/yangjunjie/work/bs-uhf/data/h4/bs-uhf-{basis}.h5", f"{r}", data_dict)

if __name__ == "__main__":
    basis = "cc-pvdz"
    for x in numpy.linspace(0.4, 3.2, 41):
        solve_h4_bs_uhf(x, basis=basis, tol=1e-6)
                
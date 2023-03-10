import itertools
from functools import reduce

import os, sys, numpy, scipy
from sys import stdout
from numpy import linalg

from pyscf import gto, scf, fci
from pyscf import mp, ao2mo, ci, lib
from pyscf.ci  import ucisd
from pyscf.ci.ucisd import _cp
from pyscf.fci import cistring, addons

from pyscf.tools.dump_mat import dump_rec

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

def proj_ucisd_to_fci_vec(coeff_rhf, coeff_uhf, vec_ucisd=None,
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

    ovlp_rhf_uhf = numpy.einsum("smp,nq,mn->spq", coeff_uhf, coeff_rhf, ovlp_ao)
    vec_fci      = ucisd.to_fcivec(vec_ucisd, norb, nelec)
    return addons.transform_ci_for_orbital_rotation(vec_fci, norb, nelec, ovlp_rhf_uhf)

def ucisd_guess(ucisd_obj, eris=None):
    if eris is None: eris = ucisd_obj.ao2mo()

    nocca, noccb = ucisd_obj.nocc
    mo_ea, mo_eb = eris.mo_energy
    eia_a = mo_ea[:nocca,None] - mo_ea[None,nocca:]
    eia_b = mo_eb[:noccb,None] - mo_eb[None,noccb:]

    norb_a  = ucisd_obj.nmo[0]
    norb_b  = ucisd_obj.nmo[1]
    assert norb_a == norb_b
    norb    = norb_a

    nvira   = norb - nocca
    nvirb   = norb - noccb
    fvv_a   = eris.focka[nocca:,nocca:]
    fvv_b   = eris.fockb[noccb:,noccb:]
    foo_a   = eris.focka[:nocca,:nocca]
    foo_b   = eris.fockb[:noccb,:noccb]

    # This shall be inv(f_ab - f_ij)
    t1a = eris.focka[:nocca,nocca:].conj() / eia_a
    t1b = eris.fockb[:noccb,noccb:].conj() / eia_b

    mp2_obj = mp.MP2(ucisd_obj._scf)
    ucisd_obj._scf.converged = False
    mp2_obj.kernel()
    emp2 = mp2_obj.e_corr
    t2aa, t2ab, t2bb = mp2_obj.t2

    ci_guess = ucisd.amplitudes_to_cisdvec(1.0, (t1a, t1b), (t2aa, t2ab, t2bb))

    return emp2, ci_guess

def get_coeff_uhf(uhf_obj=None, dm0=None, is_scf=False):
    fock_ao = uhf_obj.get_fock(dm=dm0)
    mo_energy, mo_coeff = uhf_obj.eig(fock_ao, uhf_obj.get_ovlp())

    uhf_obj.mo_energy = mo_energy
    uhf_obj.mo_coeff  = mo_coeff
    uhf_obj.mo_occ    = uhf_obj.get_occ(mo_energy=mo_energy, mo_coeff=mo_coeff)
    dm_uhf = uhf_obj.make_rdm1(mo_coeff, uhf_obj.mo_occ)

    if is_scf:
        uhf_obj.kernel(dm0=dm_uhf)
        dm_uhf = uhf_obj.make_rdm1(uhf_obj.mo_coeff, uhf_obj.mo_occ)
        fock_ao = uhf_obj.get_fock(dm=dm_uhf)
        mo_energy, mo_coeff = uhf_obj.eig(fock_ao, uhf_obj.get_ovlp())
        uhf_obj.mo_energy = mo_energy
        uhf_obj.mo_coeff  = mo_coeff
        uhf_obj.mo_occ    = uhf_obj.get_occ(mo_energy=mo_energy, mo_coeff=mo_coeff)

    ene_uhf = uhf_obj.energy_elec(dm_uhf, h1e=None, vhf=None)[0]
    return ene_uhf, uhf_obj.mo_coeff

def get_uhf_vfci(coeff_rhf=None, coeff_uhf=None, mo_occ_uhf=None, ovlp_ao=None, uhf_obj=None):
    nao, norb   = coeff_rhf.shape
    nelec       = uhf_obj.nelec
    nelec_alph  = nelec[0]
    nelec_beta  = nelec[1]

    # uhf_obj.reset()
    uhf_obj.mo_occ   = mo_occ_uhf
    uhf_obj.mo_coeff = coeff_uhf
    dm_uhf  = uhf_obj.make_rdm1(coeff_uhf, mo_occ_uhf)
    ene_uhf = uhf_obj.energy_elec(dm_uhf, h1e=None, vhf=None)[0]
    
    vfci = proj_uhf_to_fci_vec(coeff_rhf, coeff_uhf, nelec, ovlp_ao)
    return ene_uhf, vfci

def get_ump2_vfci(coeff_rhf=None, coeff_uhf=None, mo_occ_uhf=None, ovlp_ao=None, uhf_obj=None):
    nao, norb   = coeff_rhf.shape
    nelec       = uhf_obj.nelec
    nelec_alph  = nelec[0]
    nelec_beta  = nelec[1]
    
    uhf_obj.reset()
    uhf_obj.mo_occ   = mo_occ_uhf
    uhf_obj.mo_coeff = coeff_uhf
    dm_uhf  = uhf_obj.make_rdm1(coeff_uhf, mo_occ_uhf)
    ene_uhf = uhf_obj.energy_elec(dm_uhf, h1e=None, vhf=None)[0]

    ucisd_obj = ci.UCISD(uhf_obj)
    ene_ump2_corr, vec_ucisd_ump2 = ucisd_guess(ucisd_obj, eris=None)
    vfci = proj_ucisd_to_fci_vec(coeff_rhf, coeff_uhf, vec_ucisd_ump2, nelec, ovlp_ao)
    return ene_ump2_corr + ene_uhf, vfci

def get_ucisd_vfci(coeff_rhf=None, coeff_uhf=None, mo_occ_uhf=None, ovlp_ao=None, uhf_obj=None):
    nelec   = uhf_obj.nelec
    ovlp_ao = uhf_obj.get_ovlp()

    assert uhf_obj.mo_occ is not None
    assert uhf_obj.mo_energy is not None
    dm_uhf  = uhf_obj.make_rdm1(coeff_uhf, uhf_obj.mo_occ)
    ene_uhf = uhf_obj.energy_elec(dm_uhf, h1e=None, vhf=None)[0]

    ucisd_obj = ci.UCISD(uhf_obj)
    ucisd_obj.verbose = 4
    ucisd_obj.max_cycle = 50000
    ucisd_obj.max_space = 1000
    ene_ump2_corr, vec_ucisd_ump2 = ucisd_obj.get_init_guess()
    ene_ucisd_corr, vec_ucisd = ucisd_obj.kernel(vec_ucisd_ump2)
    assert ucisd_obj.converged
    
    vfci = proj_ucisd_to_fci_vec(coeff_rhf, coeff_uhf, vec_ucisd, nelec, ovlp_ao)
    return ene_ucisd_corr + ene_uhf, vfci

def solve_uhf_noci(v_bs_uhf_list, hv_bs_uhf_list, ene_bs_uhf_list, tol=1e-8):
    ene_bs_uhf_list = numpy.asarray(ene_bs_uhf_list)
    v_bs_uhf_list   = numpy.asarray(v_bs_uhf_list)
    hv_bs_uhf_list  = numpy.asarray(hv_bs_uhf_list)

    nstate    = ene_bs_uhf_list.size
    ndet_alph = v_bs_uhf_list.shape[1]
    ndet_beta = v_bs_uhf_list.shape[2]
    assert ene_bs_uhf_list.shape == (nstate,)
    assert v_bs_uhf_list.shape   == (nstate, ndet_alph, ndet_beta)
    assert hv_bs_uhf_list.shape  == (nstate, ndet_alph, ndet_beta)

    v_dot_v  = numpy.einsum('Iab,Jab->IJ', v_bs_uhf_list, v_bs_uhf_list)
    v_dot_hv = numpy.einsum('Iab,Jab->IJ', v_bs_uhf_list, hv_bs_uhf_list)

    diag_err = numpy.linalg.norm(numpy.diag(v_dot_v) - 1.0)
    if not diag_err < tol:
        print("Warning: diagonal elements of v_uhf_dot_v_ump2 is not 1.0")
        print(f"diag_err = {diag_err : 12.8e}")

    ene_err = numpy.diag(v_dot_hv) - ene_bs_uhf_list
    ene_err = numpy.linalg.norm(ene_err)
    if not ene_err < tol:
        print("Warning: diagonal elements of v_uhf_dot_hv_ump2 is not ene_ump2_list")
        print(f"ene_err = {ene_err : 12.8e}")

        print("ene_ump2_list = ", ene_bs_uhf_list)
        print("diag(v_uhf_dot_hv_ump2) = ", numpy.diag(v_dot_hv))

    ene_noci, vfci_noci = solve_variational_noci(v_bs_uhf_list, hv_bs_uhf_list, tol=tol)
    return ene_noci, vfci_noci

def truncate_generalized_eigen_problem(h, s, tol=1e-8):
    u, e, vh = scipy.linalg.svd(s)
    mask = numpy.abs(e) > tol
    n0 = s.shape[0]

    u   = u[:,mask]
    e   = e[mask]
    vh  = vh[mask,:]
    
    n1 = vh.shape[0]
    trunc_err = numpy.linalg.norm(s - reduce(numpy.dot, (u, numpy.diag(e), vh)))
    
    if not trunc_err < tol:
        print("Warning: truncation error is large")
        print("tol = %8.4e, trunc_err = %8.4e: %d -> %d" % (tol, trunc_err, n0, n1))

    if n0 != n1:
        print("tol = %8.4e, trunc_err = %8.4e: %d -> %d" % (tol, trunc_err, n0, n1))

    heff = reduce(numpy.dot, (u.T, h, vh.T))
    seff = reduce(numpy.dot, (u.T, s, vh.T))

    return heff, seff, vh

def solve_variational_noci(v1, hv1, v2=None, tol=1e-8, ref=None):
    v1_dot_v1  = numpy.einsum('Iab,Jab->IJ', v1, v1)
    v1_dot_hv1 = numpy.einsum('Iab,Jab->IJ', v1, hv1)

    res  = truncate_generalized_eigen_problem(v1_dot_hv1, v1_dot_v1, tol=tol)
    heff = res[0]
    seff = res[1]
    vh   = res[2]

    is_symmetric = numpy.allclose(heff, heff.T, atol=tol)
    if not is_symmetric:
        print("Warning: heff is not symmetric, please check")
        print("heff = ")
        dump_rec(stdout, heff)

    ene_noci, vec_noci = scipy.linalg.eigh(heff, seff)
    gs_idx = numpy.argmin(ene_noci)
    ene_noci = ene_noci[gs_idx]

    if not numpy.abs(ene_noci.imag) < tol:
        print("Warning: imaginary part of noci energy is large")
        print(f"ene_noci = {ene_noci.real : 20.12f} + {ene_noci.imag : 20.12f}i")

    vfci_noci = numpy.einsum('Iab,JI,JA->Aab', v1, vh, vec_noci, optimize=True)
    vfci_noci = vfci_noci[gs_idx]
    return ene_noci, vfci_noci

def solve_projection_noci(v1, hv1, v2=None, tol=1e-8, ref=None):
    v2_dot_v1  = numpy.einsum('Iab,Jab->IJ', v2,  v1)
    v2_dot_hv1 = numpy.einsum('Iab,Jab->IJ', v2, hv1)

    res  = truncate_generalized_eigen_problem(v2_dot_hv1, v2_dot_v1, tol=tol)
    heff = res[0]
    seff = res[1]
    vh   = res[2]

    ene_noci, vec_noci = scipy.linalg.eig(heff, seff)

    gs_idx = numpy.argmin(ene_noci.real)

    if ref is not None:
        ref_idx = numpy.argmin(numpy.abs(ene_noci.real - ref))

        if gs_idx != ref_idx:
            if numpy.abs(ene_noci[gs_idx] - ene_noci[ref_idx]) > 1.0:
                print("Warning: noci reference energy is used instead of the minimum energy")
                print(f"ene_min = {ene_noci.real[gs_idx] : 20.12f}")
                print(f"ene_ref = {ene_noci.real[ref_idx] : 20.12f}")
                gs_idx = ref_idx

    ene_noci = ene_noci[gs_idx]

    if not numpy.abs(ene_noci.imag) < tol:
        print("Warning: imaginary part of noci energy is large")
        print(f"ene_noci = {ene_noci.real : 12.8f}{ene_noci.imag :+12.8f}i")

    vfci_noci = numpy.einsum('Iab,JI,JA->Aab', v1, vh, vec_noci, optimize=True)
    vfci_noci = vfci_noci[gs_idx]

    if not numpy.linalg.norm(vfci_noci.imag) < tol:
        print("Warning: imaginary part of noci coefficients is large: %8.4e" % numpy.linalg.norm(vfci_noci.imag))

    return ene_noci.real, vfci_noci.real

def solve_ump2_noci(v_bs_list, hv_bs_list, v_bs_uhf_list=None, ene_ump2_list=None, tol=1e-8, ref=None, method=1):
    ene_ump2_list = numpy.asarray(ene_ump2_list)
    v_bs_list     = numpy.asarray(v_bs_list)
    hv_bs_list    = numpy.asarray(hv_bs_list)
    v_bs_uhf_list = numpy.asarray(v_bs_uhf_list)

    nstate    = ene_ump2_list.size
    ndet_alph = v_bs_list.shape[1]
    ndet_beta = v_bs_list.shape[2]
    assert ene_ump2_list.shape == (nstate,)
    assert v_bs_list.shape     == (nstate, ndet_alph, ndet_beta)
    assert hv_bs_list.shape    == (nstate, ndet_alph, ndet_beta)
    assert v_bs_uhf_list.shape == (nstate, ndet_alph, ndet_beta)

    v_uhf_dot_hv_ump2 = numpy.einsum('Iab,Jab->IJ', v_bs_uhf_list, hv_bs_list)
    v_uhf_dot_v_ump2  = numpy.einsum('Iab,Jab->IJ', v_bs_uhf_list, v_bs_list)

    diag_err = numpy.linalg.norm(numpy.diag(v_uhf_dot_v_ump2) - 1.0)
    if not diag_err < tol:
        print("Warning: diagonal elements of v_uhf_dot_v_ump2 is not 1.0")
        print(f"diag_err = {diag_err : 12.8e}")

    ene_err = numpy.diag(v_uhf_dot_hv_ump2) - ene_ump2_list
    ene_err = numpy.linalg.norm(ene_err)
    if not ene_err < tol:
        print("Warning: diagonal elements of v_uhf_dot_hv_ump2 is not ene_ump2_list")
        print(f"ene_err = {ene_err : 12.8e}")

    if method == 1:
        v1  = v_bs_list
        hv1 = hv_bs_list
        v2  = v_bs_uhf_list
        ene_noci, vfci_noci = solve_variational_noci(v1, hv1, v2=v2, tol=tol, ref=ref)

    elif method == 2:
        v1  = v_bs_list
        hv1 = hv_bs_list
        v2  = v_bs_uhf_list
        ene_noci, vfci_noci = solve_projection_noci(v1, hv1, v2=v2, tol=tol, ref=ref)

    return ene_noci, vfci_noci

def solve_ucisd_noci(v_bs_list, hv_bs_list, v_bs_uhf_list=None, ene_ucisd_list=None, tol=1e-8, ref=None, method=1):
    ene_ucisd_list = numpy.asarray(ene_ucisd_list)
    v_bs_list     = numpy.asarray(v_bs_list)
    hv_bs_list    = numpy.asarray(hv_bs_list)
    v_bs_uhf_list = numpy.asarray(v_bs_uhf_list)

    nstate    = ene_ucisd_list.size
    ndet_alph = v_bs_list.shape[1]
    ndet_beta = v_bs_list.shape[2]
    assert ene_ucisd_list.shape == (nstate,)
    assert v_bs_list.shape      == (nstate, ndet_alph, ndet_beta)
    assert hv_bs_list.shape     == (nstate, ndet_alph, ndet_beta)
    assert v_bs_uhf_list.shape  == (nstate, ndet_alph, ndet_beta)

    v_dot_v  = numpy.einsum('Iab,Jab->IJ', v_bs_list, v_bs_list)
    v_dot_hv = numpy.einsum('Iab,Jab->IJ', v_bs_list, hv_bs_list)

    diag_err = numpy.diag(v_dot_v) - 1.0
    if not numpy.linalg.norm(diag_err) < tol:
        print("Warning: diagonal elements of v_dot_v is not 1.0")
        print(f"diag_err = {diag_err : 12.8e}")

    ene_err = numpy.diag(v_dot_hv) - ene_ucisd_list
    ene_err = numpy.linalg.norm(ene_err)
    if not ene_err < tol:
        print("Warning: diagonal elements of v_dot_hv is not ene_ucisd_list")
        print(f"ene_err = {ene_err : 12.8e}")

    if method == 1:
        v1  = v_bs_list
        hv1 = hv_bs_list
        v2  = v_bs_uhf_list
        ene_noci, vfci_noci = solve_variational_noci(v1, hv1, v2=v2, tol=tol, ref=ref)

    elif method == 2:
        v1  = v_bs_list
        hv1 = hv_bs_list
        v2  = v_bs_uhf_list
        ene_noci, vfci_noci = solve_projection_noci(v1, hv1, v2=v2, tol=tol, ref=ref)

    return ene_noci, vfci_noci
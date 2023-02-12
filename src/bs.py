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
    t1a = eris.focka[:nocca,nocca:].conj() / eia_a
    t1b = eris.fockb[:noccb,noccb:].conj() / eia_b

    eris_ovov = _cp(eris.ovov)
    eris_ovOV = _cp(eris.ovOV)
    eris_OVOV = _cp(eris.OVOV)
    t2aa = eris_ovov.transpose(0,2,1,3) - eris_ovov.transpose(0,2,3,1)
    t2bb = eris_OVOV.transpose(0,2,1,3) - eris_OVOV.transpose(0,2,3,1)
    t2ab = eris_ovOV.transpose(0,2,1,3).copy()
    t2aa = t2aa.conj()
    t2ab = t2ab.conj()
    t2bb = t2bb.conj()
    t2aa /= lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
    t2ab /= lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
    t2bb /= lib.direct_sum('ia+jb->ijab', eia_b, eia_b)

    emp2  = numpy.einsum('iajb,ijab', eris_ovov, t2aa) * .25
    emp2 -= numpy.einsum('jaib,ijab', eris_ovov, t2aa) * .25
    emp2 += numpy.einsum('iajb,ijab', eris_OVOV, t2bb) * .25
    emp2 -= numpy.einsum('jaib,ijab', eris_OVOV, t2bb) * .25
    emp2 += numpy.einsum('iajb,ijab', eris_ovOV, t2ab)
    emp2 += numpy.einsum('ia,ia', t1a, eris.focka[:nocca,nocca:])
    emp2 += numpy.einsum('ia,ia', t1b, eris.fockb[:noccb,noccb:])
    ucisd_obj.emp2 = emp2.real

    ci_guess = ucisd.amplitudes_to_cisdvec(1, (t1a,t1b), (t2aa,t2ab,t2bb))

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

def get_uhf_vfci(coeff_rhf=None, coeff_uhf=None, uhf_obj=None):
    nelec   = uhf_obj.nelec
    ovlp_ao = uhf_obj.get_ovlp()
    
    vfci = proj_uhf_to_fci_vec(coeff_rhf, coeff_uhf, nelec, ovlp_ao)
    assert uhf_obj.mo_occ is not None
    assert uhf_obj.mo_energy is not None
    dm_uhf = uhf_obj.make_rdm1(coeff_uhf, uhf_obj.mo_occ)
    ene_uhf = uhf_obj.energy_elec(dm_uhf, h1e=None, vhf=None)[0]
    return ene_uhf, vfci

def get_ump2_vfci(coeff_rhf=None, coeff_uhf=None, uhf_obj=None):
    nelec   = uhf_obj.nelec
    ovlp_ao = uhf_obj.get_ovlp()

    assert uhf_obj.mo_occ is not None
    assert uhf_obj.mo_energy is not None
    dm_uhf = uhf_obj.make_rdm1(coeff_uhf, uhf_obj.mo_occ)
    ene_uhf = uhf_obj.energy_elec(dm_uhf, h1e=None, vhf=None)[0]

    ucisd_obj = ci.UCISD(uhf_obj)
    ene_ump2_corr, vec_ucisd_ump2 = ucisd_guess(ucisd_obj, eris=None)
    vfci = proj_ucisd_to_fci_vec(coeff_rhf, coeff_uhf, vec_ucisd_ump2, nelec, ovlp_ao)
    return ene_ump2_corr + ene_uhf, vfci

def get_ucisd_vfci(coeff_rhf=None, coeff_uhf=None, uhf_obj=None):
    nelec   = uhf_obj.nelec
    ovlp_ao = uhf_obj.get_ovlp()

    assert uhf_obj.mo_occ is not None
    assert uhf_obj.mo_energy is not None
    dm_uhf  = uhf_obj.make_rdm1(coeff_uhf, uhf_obj.mo_occ)
    ene_uhf = uhf_obj.energy_elec(dm_uhf, h1e=None, vhf=None)[0]

    ucisd_obj = ci.UCISD(uhf_obj)
    ucisd_obj.max_cycle = 5000
    ucisd_obj.max_space = 500
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

    ene_err = numpy.diag(v_dot_hv) / numpy.diag(v_dot_v) - ene_bs_uhf_list
    assert numpy.linalg.norm(ene_err) < 1e-10

    eigvals, eigvecs = scipy.linalg.eigh(v_dot_hv, v_dot_v)
    mask = numpy.abs(eigvals) > tol

    eigvals = eigvals[mask]
    eigvecs = eigvecs[:,mask]

    heff = reduce(numpy.dot, (eigvecs.T, v_dot_hv, eigvecs))
    seff = reduce(numpy.dot, (eigvecs.T, v_dot_v, eigvecs))

    ene_noci, vec_noci = scipy.linalg.eigh(heff, seff)
    return ene_noci[0]

def solve_ump2_noci(v_bs_list, hv_bs_list, v_bs_uhf_list=None, ene_ump2_list=None, tol=1e-8, method=1):
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
    assert diag_err < 1e-10

    ene_err = numpy.diag(v_uhf_dot_hv_ump2) / numpy.diag(v_uhf_dot_v_ump2) - ene_ump2_list
    assert numpy.linalg.norm(ene_err) < 1e-10

    if method == 1:
        v_dot_v  = numpy.einsum('Iab,Jab->IJ', v_bs_list, v_bs_list)
        v_dot_hv = numpy.einsum('Iab,Jab->IJ', v_bs_list, hv_bs_list)

        eigvals, eigvecs = scipy.linalg.eigh(v_dot_hv, v_dot_v)
        mask = numpy.abs(eigvals) > tol

        eigvals = eigvals[mask]
        eigvecs = eigvecs[:,mask]

        heff = reduce(numpy.dot, (eigvecs.T, v_dot_hv, eigvecs))
        seff = reduce(numpy.dot, (eigvecs.T, v_dot_v, eigvecs))

        ene_noci, vec_noci = scipy.linalg.eigh(heff, seff)
        ene_noci = numpy.min(ene_noci)

    elif method == 2:
        v_dot_v_diag = numpy.einsum('Iab,Iab->I', v_bs_uhf_list, v_bs_list)
        diag_sign    = numpy.sign(v_dot_v_diag)

        v_dot_v  = numpy.einsum('Iab,Jab,J->IJ', v_bs_uhf_list, v_bs_list, diag_sign)
        v_dot_hv = numpy.einsum('Iab,Jab,J->IJ', v_bs_uhf_list, hv_bs_list, diag_sign)

        eigvals, eigvecs = scipy.linalg.eig(v_dot_v)
        mask = numpy.abs(eigvals) > tol

        eigvals = eigvals[mask]
        eigvecs = eigvecs[:,mask]

        heff = reduce(numpy.dot, (eigvecs.T.conj(), v_dot_hv, eigvecs))
        seff = reduce(numpy.dot, (eigvecs.T.conj(), v_dot_v, eigvecs))
        assert numpy.linalg.norm(numpy.diag(seff) - eigvals) < 1e-10

        ene_noci, vec_noci = scipy.linalg.eig(heff, seff)
        ene_noci_argsort   = numpy.argsort(ene_noci.real)
        ene_noci           = ene_noci[ene_noci_argsort[0]]

        assert numpy.abs(ene_noci.imag) < 1e-10
        ene_noci = ene_noci.real

    return ene_noci

def solve_ucisd_noci(v_bs_list, hv_bs_list, v_bs_uhf_list=None, ene_ucisd_list=None, tol=1e-8, method=1):
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
    assert numpy.linalg.norm(diag_err) < 1e-10

    ene_err = numpy.diag(v_dot_hv) - ene_ucisd_list
    assert numpy.linalg.norm(ene_err) < 1e-10

    if method == 1:
        eigvals, eigvecs = scipy.linalg.eigh(v_dot_v)
        mask = numpy.abs(eigvals) > tol

        eigvals = eigvals[mask]
        eigvecs = eigvecs[:,mask]

        heff = reduce(numpy.dot, (eigvecs.T, v_dot_hv, eigvecs))
        seff = reduce(numpy.dot, (eigvecs.T, v_dot_v, eigvecs))

        ene_noci, vec_noci = scipy.linalg.eigh(heff, seff)
        ene_noci = numpy.min(ene_noci)

    elif method == 2:
        v_dot_v_diag = numpy.einsum('Iab,Iab->I', v_bs_uhf_list, v_bs_list)
        diag_sign    = numpy.sign(v_dot_v_diag)

        v_dot_v  = numpy.einsum('Iab,Jab,J->IJ', v_bs_uhf_list, v_bs_list, diag_sign)
        v_dot_hv = numpy.einsum('Iab,Jab,J->IJ', v_bs_uhf_list, hv_bs_list, diag_sign)

        eigvals, eigvecs = scipy.linalg.eig(v_dot_v)
        mask = numpy.abs(eigvals) > tol

        eigvals = eigvals[mask]
        eigvecs = eigvecs[:,mask]

        heff = reduce(numpy.dot, (eigvecs.T.conj(), v_dot_hv, eigvecs))
        seff = reduce(numpy.dot, (eigvecs.T.conj(), v_dot_v, eigvecs))
        assert numpy.linalg.norm(numpy.diag(seff) - eigvals) < 1e-10

        ene_noci, vec_noci = scipy.linalg.eig(heff, seff)
        ene_noci_argsort   = numpy.argsort(ene_noci.real)
        ene_noci           = ene_noci[ene_noci_argsort[0]]
        
        assert numpy.abs(ene_noci.imag) < 1e-10
        ene_noci = ene_noci.real

    return ene_noci
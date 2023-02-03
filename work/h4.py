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

    ovlp_rhf_uhf = numpy.einsum("smp,nq,mn->spq", coeff_uhf, coeff_rhf, ovlp_ao)
    vec_cisd = ucisd.amplitudes_to_cisdvec(t0, (t1_a, t1_b), t2)
    vec_fci  = ucisd.to_fcivec(vec_cisd, norb, nelec)
    return addons.transform_ci_for_orbital_rotation(vec_fci, norb, nelec, ovlp_rhf_uhf)

def solve_h4_bs_uhf(r, basis="sto-3g"):
    atoms  = ""
    atoms += "H 0.0000 0.0000 % 12.8f\n" % ( 3.0 * r / 2.0)
    atoms += "H 0.0000 0.0000 % 12.8f\n" % (-3.0 * r / 2.0)
    atoms += "H 0.0000 0.0000 % 12.8f\n" % ( r / 2.0)
    atoms += "H 0.0000 0.0000 % 12.8f\n" % (-r / 2.0)

    mol = gto.Mole()
    mol.atom  = atoms
    mol.basis = basis
    mol.build()

    rhf_obj = scf.RHF(mol)
    rhf_obj.verbose = 0
    rhf_obj.conv_tol = 1e-12
    rhf_obj.kernel()
    assert rhf_obj.converged

    ene_rhf   = rhf_obj.energy_elec()[0]
    coeff_rhf = rhf_obj.mo_coeff
    ovlp_ao   = rhf_obj.get_ovlp()

    fci_obj = fci.FCI(mol, mo=rhf_obj.mo_coeff, singlet=False)
    fci_obj.verbose = 0
    ene_fci = fci_obj.kernel()[0] - mol.energy_nuc()

    vfci_uhf_list   = []
    hvfci_uhf_list  = []
    hvfci_ump2_list = []
    ene_uhf_list    = []
    ene_ump2_list   = []

    norb = coeff_rhf.shape[1]
    nelec_alph, nelec_beta = mol.nelec[0], mol.nelec[1]
    nelec = (nelec_alph, nelec_beta)
    spin_list  = [0 for i in range(nelec_alph)]
    spin_list += [1 for i in range(nelec_beta)]

    from pyscf.fci.direct_spin1 import absorb_h1e, contract_2e
    h1e = reduce(numpy.dot, (coeff_rhf.conj().T, rhf_obj.get_hcore(), coeff_rhf))
    h2e = ao2mo.kernel(rhf_obj._eri, coeff_rhf)
    ham = absorb_h1e(h1e, h2e, norb, nelec, 0.5)

    data_dict = {
        "RHF" : ene_rhf, "FCI" : ene_fci,
        "r": r, "ene_nuc": mol.energy_nuc(),
    }

    uhf_obj = scf.UHF(mol)
    uhf_obj.verbose = 0
    uhf_obj.max_cycle = 1200
    uhf_obj.conv_tol = 1e-8
    uhf_obj.diis_start_cycle = 20
    uhf_obj.diis_space = 10

    for iidx, alph_idx in enumerate(itertools.combinations(range(nelec_alph + nelec_beta), nelec_alph)):
        alph_idx = list(alph_idx)
        ss_idx   = [0 if i not in alph_idx else 1 for i in range(nelec_alph + nelec_beta)]
        s1, s2, s3, s4 = ss_idx
        dm0 = numpy.zeros((2, mol.nao, mol.nao))
        dm0[s1, 0, 0] = 1.0
        dm0[s2, 1, 1] = 1.0
        dm0[s3, 2, 2] = 1.0
        dm0[s4, 3, 3] = 1.0

        uhf_obj.kernel(dm0=dm0)
        assert uhf_obj.converged

        coeff_uhf = uhf_obj.mo_coeff
        ene_uhf   = uhf_obj.energy_elec()[0]

        ump2_obj = mp.UMP2(uhf_obj)
        ump2_obj.verbose = 0
        ene_ump2, t2_ump2 = ump2_obj.kernel(mo_coeff=coeff_uhf)
        ene_ump2 += ene_uhf

        vfci_uhf  = proj_uhf_to_fci_vec(coeff_rhf, coeff_uhf, nelec, ovlp_ao)
        vfci_ump2 = proj_ump2_to_fci_vec(coeff_rhf, coeff_uhf, nelec=nelec, ovlp_ao=ovlp_ao, t2=t2_ump2)

        hvfci_uhf  = contract_2e(ham, vfci_uhf,  norb, nelec)
        hvfci_ump2 = contract_2e(ham, vfci_ump2, norb, nelec)

        data_dict["UHF-%d" % iidx]  = ene_uhf
        data_dict["UMP2-%d" % iidx] = ene_ump2

        vfci_uhf_list.append(vfci_uhf)
        hvfci_uhf_list.append(hvfci_uhf)
        ene_uhf_list.append(ene_uhf)
        ene_ump2_list.append(ene_ump2)

    ene_list   = [ene_uhf_list, ene_ump2_list]
    vfci_list  = [vfci_uhf_list, vfci_ump2_list]
    hvfci_list = [hvfci_uhf_list, hvfci_ump2_list]

    for iv, v in enumerate([vfci_uhf_list, vfci_ump2])

    vfci_uhf_list = numpy.asarray(vfci_uhf_list)
    v_dot_v  = numpy.einsum("Iij,Jij->IJ", vfci_uhf_list, vfci_uhf_list)
    v_dot_hv = numpy.einsum("Iij,Jij->IJ", vfci_uhf_list, hvfci_uhf_list)
    
    h_diag  = numpy.diag(v_dot_hv)
    ene_uhf = numpy.asarray(ene_uhf_list)
    assert numpy.linalg.norm(h_diag - ene_uhf) < 1e-8

    eigval, eigvec = scipy.linalg.eigh(v_dot_v)
    mask = numpy.abs(eigval) > 1e-12
    h    = reduce(numpy.dot, (eigvec[:, mask].conj().T, v_dot_hv, eigvec[:, mask]))
    s    = reduce(numpy.dot, (eigvec[:, mask].conj().T, v_dot_v,  eigvec[:, mask]))
    ene_noci, vec_noci = scipy.linalg.eigh(h ,s)

    tmp  = "r = %10.6f, " % r
    tmp += "".join(["%12.4e, " % i for i in eigval])[:-1]
    print(tmp)

    data_dict["NOCI"] = ene_noci[0]
    chkfile.save(f"/Users/yangjunjie/work/bs-uhf/data/h4/bs-uhf-{basis}.h5", f"{r}", data_dict)

if __name__ == "__main__":
    basis = "sto3g"
    for x in numpy.linspace(0.4, 3.2, 41):
        solve_h4_bs_uhf(x, basis=basis)
                
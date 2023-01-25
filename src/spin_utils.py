import functools
import numpy, scipy

import pyscf
from pyscf import fci, lib
from pyscf.fci import cistring

def coeff_rhf_to_ghf(coeff_rhf, mo_energy_rhf = None):
    coeff_alph = coeff_rhf
    coeff_beta = coeff_rhf

    nao, norb_alph = coeff_alph.shape
    norb_beta = coeff_beta.shape[1]

    assert coeff_alph.shape == (nao, norb_alph)
    assert coeff_beta.shape == (nao, norb_beta)

    coeff_ghf = [numpy.hstack((coeff_alph, numpy.zeros((nao, norb_beta)))), 
                 numpy.hstack((numpy.zeros((nao, norb_alph)), coeff_beta))]
    coeff_ghf = numpy.vstack(coeff_ghf).reshape(nao * 2, norb_alph + norb_beta)

    if mo_energy_rhf is not None:
        mo_energy_alph = mo_energy_rhf
        mo_energy_beta = mo_energy_rhf
        assert mo_energy_alph.shape == (norb_alph,)
        assert mo_energy_beta.shape == (norb_beta,)

        mo_energy_ghf = numpy.hstack((mo_energy_alph, mo_energy_beta))
        mo_argsort    = numpy.argsort(mo_energy_ghf)
        coeff_ghf     = coeff_ghf[:, mo_argsort]

    return coeff_ghf

def coeff_uhf_to_ghf(coeff_uhf, mo_energy_uhf = None, mo_occ_uhf = None):
    coeff_alph, coeff_beta = coeff_uhf
    mo_energy_alph, mo_energy_beta = mo_energy_uhf

    nao, norb_alph = coeff_alph.shape
    norb_beta = coeff_beta.shape[1]

    assert coeff_alph.shape == (nao, norb_alph)
    assert coeff_beta.shape == (nao, norb_beta)

    coeff_ghf  = [numpy.hstack((coeff_alph, numpy.zeros((nao, norb_beta)))), 
                  numpy.hstack((numpy.zeros((nao, norb_alph)), coeff_beta))]
    coeff_ghf  = numpy.vstack(coeff_ghf).reshape(nao * 2, norb_alph + norb_beta)
    mo_occ_ghf = numpy.hstack((mo_occ_uhf[0], mo_occ_uhf[1]))

    if mo_energy_uhf is not None:
        mo_energy_alph, mo_energy_beta = mo_energy_uhf

        assert mo_energy_alph.shape == (norb_alph,)
        assert mo_energy_beta.shape == (norb_beta,)

        mo_energy_ghf = numpy.hstack((mo_energy_alph, mo_energy_beta))
        mo_argsort    = numpy.argsort(mo_energy_ghf)
        coeff_ghf     = coeff_ghf[:, mo_argsort]
        mo_occ_ghf    = mo_occ_ghf[mo_argsort]

    return coeff_ghf, mo_occ_ghf

def coeff_ghf_to_uhf(coeff_ghf, ovlp_ao=None):
    assert coeff_ghf.shape[0] % 2 == 0
    nao = coeff_ghf.shape[0] // 2
    norb = coeff_ghf.shape[1]

    coeff_uhf_alph = []
    coeff_uhf_beta = []

    for p in range(norb):
        coeff_ghf_alph_p = coeff_ghf[:nao, p]
        coeff_ghf_beta_p = coeff_ghf[nao:, p]

        is_alph = False
        is_beta = False

        if ovlp_ao is not None:
            w_alph = functools.reduce(numpy.dot, (coeff_ghf_alph_p.T, ovlp_ao, coeff_ghf_alph_p))
            w_beta = functools.reduce(numpy.dot, (coeff_ghf_beta_p.T, ovlp_ao, coeff_ghf_beta_p))

            if w_alph > 0.9:
                is_alph = True
            
            if w_beta > 0.9:
                is_beta = True

        else:
            w_alph = numpy.linalg.norm(coeff_ghf_alph_p)
            w_beta = numpy.linalg.norm(coeff_ghf_beta_p)

            if w_alph > w_beta:
                is_alph = True
            
            else:
                is_beta = True

        if is_alph and not is_beta:
            coeff_uhf_alph.append(coeff_ghf_alph_p)
        
        elif is_beta and not is_alph:
            coeff_uhf_beta.append(coeff_ghf_beta_p)
        
        else:
            print("Warning: GHF orbital %d is not a UHF orbital." % p)
            raise RuntimeError

    coeff_uhf_alph = numpy.array(coeff_uhf_alph)
    coeff_uhf_beta = numpy.array(coeff_uhf_beta)

    norb_alph = coeff_uhf_alph.shape[1]
    norb_beta = coeff_uhf_beta.shape[1]

    assert coeff_uhf_alph.shape == (nao, norb_alph)
    assert coeff_uhf_beta.shape == (nao, norb_beta)
    return (coeff_uhf_alph, coeff_uhf_beta)

def coeff_ghf_to_gso(coeff_ghf):
    assert coeff_ghf.shape[0] % 2 == 0
    nao = coeff_ghf.shape[0] // 2
    norb = coeff_ghf.shape[1]

    coeff_gso = [[coeff_ghf[:nao, :], coeff_ghf[nao:, :]]]
    coeff_gso = numpy.array(coeff_gso).reshape((2, nao, norb))
    return coeff_gso

def coeff_gso_to_ghf(coeff_gso):
    assert coeff_gso.shape[0] == 2
    nao = coeff_gso.shape[1]
    norb = coeff_gso.shape[2]

    coeff_ghf = [[coeff_gso[0, :, :], coeff_gso[1, :, :]]]
    coeff_ghf = numpy.array(coeff_ghf).reshape((2*nao, norb))
    return coeff_ghf

def rotate_coeff_gso(coeff_gso, alpha=0.0, beta=0.0, gamma=0.0):
    pauli_matrix_y = 0.5 * numpy.array([[0.0, -1.0j], [1.0j, 0.0]])
    pauli_matrix_z = 0.5 * numpy.array([[1.0,   0.0], [0.0, -1.0]])

    rot_matrix_1 = scipy.linalg.expm(-1.0j * alpha * pauli_matrix_z)
    rot_matrix_2 = scipy.linalg.expm(-1.0j * beta  * pauli_matrix_y)
    rot_matrix_3 = scipy.linalg.expm(-1.0j * gamma * pauli_matrix_z)

    rot_matrix = functools.reduce(numpy.dot, (rot_matrix_1, rot_matrix_2, rot_matrix_3))
    return numpy.einsum("ab,bnp->anp", rot_matrix, coeff_gso)

def rotate_coeff_ghf(coeff_ghf, alpha=0.0, beta=0.0, gamma=0.0):
    coeff_gso = coeff_ghf_to_gso(coeff_ghf)
    coeff_gso = rotate_coeff_gso(coeff_gso, alpha=alpha, beta=beta, gamma=gamma)
    coeff_ghf = coeff_gso_to_ghf(coeff_gso)
    return coeff_ghf

def rdm1_ghf_to_gso(rdm1_ghf):
    assert rdm1_ghf.shape[0] % 2 == 0
    nao = rdm1_ghf.shape[0] // 2
    assert rdm1_ghf.shape == (2*nao, 2*nao)

    rdm1_aa = rdm1_ghf[:nao, :nao]
    rdm1_ab = rdm1_ghf[:nao, nao:]
    rdm1_ba = rdm1_ghf[nao:, :nao]
    rdm1_bb = rdm1_ghf[nao:, nao:]

    rdm1_gso = [[rdm1_aa, rdm1_ab], [rdm1_ba, rdm1_bb]]
    rdm1_gso = numpy.array(rdm1_gso).reshape((2, 2, nao, nao))
    return rdm1_gso

def rdm1_gso_to_ghf(rdm1_gso):
    assert rdm1_gso.shape[0] == 2
    assert rdm1_gso.shape[1] == 2
    nao = rdm1_gso.shape[2]
    assert rdm1_gso.shape == (2, 2, nao, nao)

    rdm1_aa = rdm1_gso[0, 0, :, :]
    rdm1_ab = rdm1_gso[0, 1, :, :]
    rdm1_ba = rdm1_gso[1, 0, :, :]
    rdm1_bb = rdm1_gso[1, 1, :, :]

    rdm1_ghf = [[rdm1_aa, rdm1_ab], [rdm1_ba, rdm1_bb]]
    rdm1_ghf = numpy.array(rdm1_ghf).reshape((2*nao, 2*nao))
    return rdm1_ghf

def get_spin_avg(rdm1_gso=None, rdm1_ghf=None, ovlp_ao=None, ao_idx=None):
    assert ovlp_ao  is not None
    assert rdm1_gso is not None or rdm1_ghf is not None

    if rdm1_gso is None:
        rdm1_gso = rdm1_ghf_to_gso(rdm1_ghf)

    assert rdm1_gso.shape[0] == 2
    assert rdm1_gso.shape[1] == 2
    nao = rdm1_gso.shape[2]
    assert rdm1_gso.shape == (2, 2, nao, nao)

    pauli_matrix_x = 0.5 * numpy.array([[0.0,  1.0],  [1.0,  0.0]])
    pauli_matrix_y = 0.5 * numpy.array([[0.0, -1.0j], [1.0j, 0.0]])
    pauli_matrix_z = 0.5 * numpy.array([[1.0,  0.0],  [0.0, -1.0]])
    pauli_matrix   = [pauli_matrix_x, pauli_matrix_y, pauli_matrix_z]

    if ao_idx is None:
        ao_idx = numpy.arange(nao)

    rdm1_gso_block = rdm1_gso[:, :, ao_idx, :][:, :, :, ao_idx]
    ovlp_ao_block  = ovlp_ao[ao_idx, :][:, ao_idx]

    return numpy.einsum("xab,abmn,mn->x", pauli_matrix, rdm1_gso_block, ovlp_ao_block)
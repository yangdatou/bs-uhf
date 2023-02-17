import os, sys
from sys import stdout
import numpy, scipy
from scipy.linalg import sqrtm

import pyscf
from pyscf import gto, scf, lo
from pyscf.tools.dump_mat import dump_rec

def solve_n2_rohf(spin=0, basis="sto3g"):
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = 'n2-rohf.out'
    mol.atom = """
        N 0.00000 0.00000 0.00000
        N 0.00000 0.00000 3.00000
    """ 
    mol.basis = basis
    mol.spin = spin
    mol.build()

    uhf_obj = scf.UHF(mol)

    ovlp_ao = uhf_obj.get_ovlp()
    coeff_ao_lo = lo.orth_ao(mol, 'meta_lowdin')
    err = numpy.einsum("mi,nj,mn->ij", coeff_ao_lo, coeff_ao_lo, ovlp_ao) - numpy.eye(mol.nao_nr())
    assert numpy.linalg.norm(err) < 1e-10

    print("\nspin = %d" % spin)
    dump_rec(stdout, coeff_ao_lo, mol.ao_labels())

    ao_idx = list(range(mol.nao_nr()))

    alph_occ_idx = []
    alph_occ_idx += list(mol.search_ao_label("N 1s"))
    alph_occ_idx += list(mol.search_ao_label("N 2s"))
    alph_occ_idx += list(mol.search_ao_label("0 N 2p"))
    alph_vir_idx  = list(set(ao_idx) - set(alph_occ_idx))
    print("alph_occ_idx = ", alph_occ_idx)
    print("alph_vir_idx = ", alph_vir_idx)

    beta_occ_idx = []
    beta_occ_idx += list(mol.search_ao_label("N 1s"))
    beta_occ_idx += list(mol.search_ao_label("N 2s"))
    beta_occ_idx += list(mol.search_ao_label("1 N 2p"))
    beta_vir_idx  = list(set(ao_idx) - set(beta_occ_idx))
    nelec_alph = mol.nelec[0]
    nelec_beta = mol.nelec[1]

    assert len(alph_occ_idx) == nelec_alph
    assert len(beta_occ_idx) == nelec_beta

    coeff_uhf_alph = coeff_ao_lo[:, alph_occ_idx + alph_vir_idx]
    coeff_uhf_beta = coeff_ao_lo[:, beta_occ_idx + beta_vir_idx]
    coeff_uhf_alph_occ = coeff_uhf_alph[:, :nelec_alph]
    coeff_uhf_beta_occ = coeff_uhf_beta[:, :nelec_beta]

    dm_alph = numpy.dot(coeff_uhf_alph_occ, coeff_uhf_alph_occ.T)
    dm_beta = numpy.dot(coeff_uhf_beta_occ, coeff_uhf_beta_occ.T)
    dm = (dm_alph, dm_beta)
    fock = uhf_obj.get_fock(dm=dm)

    fock_alph_mo = numpy.dot(coeff_uhf_alph.T, numpy.dot(fock[0], coeff_uhf_alph))
    fock_alph_vo = fock_alph_mo[nelec_alph:, :nelec_alph]
    fock_beta_mo = numpy.dot(coeff_uhf_beta.T, numpy.dot(fock[1], coeff_uhf_beta))
    fock_beta_vo = fock_beta_mo[nelec_beta:, :nelec_beta]
    dump_rec(stdout, fock_alph_vo)
    dump_rec(stdout, fock_beta_vo)
    assert 1== 2

if __name__ == "__main__":
    solve_n2_rohf(spin=0, basis="sto3g")
    solve_n2_rohf(spin=2, basis="sto3g")
    solve_n2_rohf(spin=4, basis="sto3g")
    solve_n2_rohf(spin=6, basis="sto3g")
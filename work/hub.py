from functools import reduce
import os, sys, numpy, scipy
from sys import stdout
from numpy import linalg

from pyscf import gto, scf, fci, mp, ao2mo, ci
from pyscf.fci.direct_spin1 import absorb_h1e, contract_2e
from pyscf.tools.dump_mat import dump_rec
from pyscf.fci import cistring, spin_op

from spin_utils import coeff_rhf_to_ghf, coeff_uhf_to_ghf
from spin_utils import rotate_coeff_ghf, get_spin_avg

def get_ghf_fci_ham(ghf_obj, coeff_ghf, nelec):
    hcore = ghf_obj.get_hcore()
    nao   = hcore.shape[0] // 2
    norb  = coeff_ghf.shape[1]

    assert hcore.shape == (nao * 2, nao * 2)
    assert coeff_ghf.shape == (nao * 2, norb)

    h1e = reduce(numpy.dot, (coeff_ghf.T, ghf_obj.get_hcore(), coeff_ghf))

    coeff_alph = coeff_ghf[:nao, :]
    coeff_beta = coeff_ghf[nao:, :]
    h2e_aabb = ao2mo.kernel(ghf_obj._eri, (coeff_alph, coeff_alph, coeff_beta, coeff_beta))

    h2e  = ao2mo.kernel(ghf_obj._eri, coeff_alph)
    h2e += ao2mo.kernel(ghf_obj._eri, coeff_beta)
    h2e += h2e_aabb + h2e_aabb.T

    return h1e, h2e, absorb_h1e(h1e, h2e, norb, nelec, 0.5)

def solve_hub_bs_uhf(nelecs, nsite, u=1.0, is_pbc=False):
    nelec_alph, nelec_beta = nelecs

    mol = gto.Mole()
    mol.incore_anyway = True
    mol.nelectron = nelec_alph + nelec_beta
    mol.spin = nelec_alph - nelec_beta
    mol.build()

    h1e = numpy.zeros((nsite, nsite))
    h2e = numpy.zeros((nsite, nsite, nsite, nsite))

    for i in range(nsite - 1):
        h1e[i, i+1] = -1.0
        h1e[i+1, i] = -1.0
        h2e[i, i, i, i] = u

    if is_pbc:
        h1e[0, nsite-1] = -1.0
        h1e[nsite-1, 0] = -1.0

    h2e[nsite-1, nsite-1, nsite-1, nsite-1] = u

    rhf_obj = scf.RHF(mol)
    rhf_obj.init_guess = "1e"
    rhf_obj.get_hcore  = lambda *args: h1e
    rhf_obj.get_ovlp   = lambda *args: numpy.eye(nsite)
    rhf_obj._eri       = ao2mo.restore(8, h2e, nsite)

    rhf_obj.verbose = 0
    rhf_obj.conv_tol = 1e-12
    rhf_obj.kernel()
    assert rhf_obj.converged

    mo_coeff, mo_energy = rhf_obj.mo_coeff, rhf_obj.mo_energy
    coeff_rhf = coeff_rhf_to_ghf(mo_coeff, mo_energy)
    ene_rhf   = rhf_obj.energy_elec()[0]

    dm0 = numpy.zeros((2, nsite, nsite))
    dm0[0, 0, 0] = 1.0
    dm0[0, 2, 2] = 1.0
    dm0[1, 1, 1] = 1.0
    dm0[1, 3, 3] = 1.0

    uhf_obj = scf.UHF(mol)
    uhf_obj.get_hcore  = lambda *args: h1e
    uhf_obj.get_ovlp   = lambda *args: numpy.eye(nsite)
    uhf_obj._eri       = ao2mo.restore(8, h2e, nsite)

    uhf_obj.verbose = 0
    uhf_obj.conv_tol = 1e-12
    uhf_obj.kernel(dm0)
    assert uhf_obj.converged

    mo_coeff, mo_energy = uhf_obj.mo_coeff, uhf_obj.mo_energy
    coeff_uhf, mo_occ_uhf = coeff_uhf_to_ghf(
        uhf_obj.mo_coeff,
        uhf_obj.mo_energy,
        uhf_obj.mo_occ
    )
    ene_uhf   = uhf_obj.energy_elec()[0]

    norb_alph = mo_coeff[0].shape[1]
    norb_beta = mo_coeff[1].shape[1]
    norb = norb_alph + norb_beta

    ghf_obj = scf.GHF(mol)
    ghf_obj.get_hcore  = lambda *args: scipy.linalg.block_diag(h1e, h1e)
    ghf_obj.get_ovlp   = lambda *args: numpy.eye(nsite * 2)
    ghf_obj._eri       = ao2mo.restore(8, h2e, nsite)
    ghf_obj.verbose = 0
    ghf_obj.conv_tol = 1e-12

    dm_rhf = ghf_obj.make_rdm1(coeff_rhf, mo_occ_uhf)

    ghf_obj.kernel()
    assert ghf_obj.converged
    ene_ghf = ghf_obj.energy_elec()[0]

    print("ene_rhf", ene_rhf)
    print("ene_uhf", ene_uhf)
    print("ene_ghf", ene_ghf)


if __name__ == "__main__":
    dirpath = f"/Users/yangjunjie/work/bs-uhf/data/hub/"
    os.makedirs(dirpath, exist_ok=True)

    solve_hub_bs_uhf((2, 2), 4, u=2.0, is_pbc=False)

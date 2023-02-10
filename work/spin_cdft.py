import sys, os
from sys import stdout

import numpy, scipy
from scipy import optimize
from scipy import linalg

import pyscf
from pyscf import gto, scf, lib
from pyscf.scf import uhf
from pyscf.tools.dump_mat import dump_rec

class ConstraintMixin(object):
    pass

class SpinConstraint(ConstraintMixin):
    def __init__(self, ao_idx_list, target_spin=0.0):
        self.ao_idx_list = ao_idx_list
        self.target_spin = target_spin

    def get_spin(self, rdm1s, ovlp_ao):
        ao_idx_list = self.ao_idx_list
        ao_ix = numpy.ix_(ao_idx_list, ao_idx_list)
        ovlp_ao_block = ovlp_ao[ao_ix]
        rdm1_alph_block = rdm1s[0][ao_ix]
        rdm1_beta_block = rdm1s[1][ao_ix] 
        s  = numpy.einsum("ij,ij", ovlp_ao_block, rdm1_alph_block)
        s -= numpy.einsum("ij,ij", ovlp_ao_block, rdm1_beta_block)
        return s

    def get_res(self, rdm1s, ovlp_ao):
        s = self.get_spin(rdm1s, ovlp_ao)
        return self.get_spin(rdm1s, ovlp_ao) - self.target_spin

class UHFwithSpinConstraint(uhf.UHF):
    _spin_constraint_list = None
    _spin_lambda_list     = None

    def get_fock(self, h1e, s1e, vhf, dm, cycle=-1, diis=None, diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
        fock = uhf.get_fock(self, h1e, s1e, vhf, dm, cycle, diis, diis_start_cycle, level_shift_factor, damp_factor)
        fock_alph, fock_beta = fock
        
        _spin_constraint_list = self._spin_constraint_list
        _spin_lambda_list     = self._spin_lambda_list
        assert s1e is not None

        v_alph = numpy.zeros_like(s1e)
        v_beta = numpy.zeros_like(s1e)

        for spin_constraint, spin_lambda in zip(_spin_constraint_list, _spin_lambda_list):
            ao_idx_list = spin_constraint.ao_idx_list
            ao_ix = numpy.ix_(ao_idx_list, ao_idx_list)
            ovlp_ao_block = s1e[ao_ix]
            v_alph[ao_ix] += spin_lambda * ovlp_ao_block
            v_beta[ao_ix] -= spin_lambda * ovlp_ao_block

        fock = (fock_alph + v_alph, fock_beta + v_beta)
        return numpy.array(fock)

    def _get_res(self, rdm1s, ovlp_ao):
        res = []
        _spin_constraint_list = self._spin_constraint_list
        for spin_constraint in _spin_constraint_list:
            res.append(spin_constraint.get_res(rdm1s, ovlp_ao))
        return numpy.array(res)

    def _update_spin_lambda_list(self, _spin_lambda_list):
        num_constraints = len(self._spin_constraint_list)
        _spin_lambda_list = numpy.array(_spin_lambda_list)
        assert _spin_lambda_list.size == num_constraints
        self._spin_lambda_list = _spin_lambda_list

def solve_spin_cdft(uhf_obj, dm0=None):
    assert isinstance(uhf_obj, UHFwithSpinConstraint)
    hcore_ao = uhf_obj.get_hcore()
    ovlp_ao  = uhf_obj.get_ovlp()

    dm_list = [dm0]
    
    def res(l):
        dm = dm_list[-1]
        uhf_obj._update_spin_lambda_list(l)
        uhf_obj.kernel(dm0=dm)
        dm_list.append(uhf_obj.make_rdm1())
        r = uhf_obj._get_res(dm_list[-1], ovlp_ao)
        l_str = " ".join(["% 12.8f" % x for x in l])
        r_str = " ".join(["% 12.8f" % x for x in r])
        print("lambda: %s" % l_str)
        print("res   : %s" % r_str)
        return r

    num_constraints = len(uhf_obj._spin_constraint_list)
    assert num_constraints > 0

    l0 = numpy.zeros(num_constraints)
    if uhf_obj._spin_lambda_list is not None:
        l0 = uhf_obj._spin_lambda_list
    
    sol = optimize.newton_krylov(res, l0, verbose=1, maxiter=100)
    return sol

if __name__ == "__main__":
    r = 1.00
    atoms  = ""
    atoms += "H 0.0000 0.0000 % 12.8f\n" % ( 3.0 * r / 2.0)
    atoms += "H 0.0000 0.0000 % 12.8f\n" % (-3.0 * r / 2.0)
    atoms += "H 0.0000 0.0000 % 12.8f\n" % ( r / 2.0)
    atoms += "H 0.0000 0.0000 % 12.8f\n" % (-r / 2.0)

    mol = gto.Mole()
    mol.atom  = atoms
    mol.basis = "sto-3g"
    mol.build()

    spin_constraint_list = []
    spin_constraint_list.append(SpinConstraint([0],  -1.0))
    # spin_constraint_list.append(SpinConstraint([1], -1.0))
    # spin_constraint_list.append(SpinConstraint([2],  1.0))
    # spin_constraint_list.append(SpinConstraint([3], -1.0))
    num_constraints = len(spin_constraint_list)

    dm0 = numpy.zeros((2, mol.nao, mol.nao))
    dm0[0, 0, 0] = 1.0
    dm0[1, 1, 1] = 1.0
    dm0[0, 2, 2] = 1.0
    dm0[1, 3, 3] = 1.0

    uhf_obj = scf.UHF(mol)
    uhf_obj.verbose = 4
    uhf_obj.kernel(dm0=dm0)
    dm0 = uhf_obj.make_rdm1()
    ovlp_ao = uhf_obj.get_ovlp()

    uhf_obj = UHFwithSpinConstraint(mol)
    uhf_obj.verbose = 0
    uhf_obj._spin_constraint_list = spin_constraint_list
    uhf_obj._spin_lambda_list = numpy.zeros(num_constraints) + 0.54
    solve_spin_cdft(uhf_obj, dm0=dm0)

    # for lam in numpy.linspace(-1.0, 1.0, 101):
    #     tmp_spin_lambda_list = numpy.zeros(num_constraints)
    #     tmp_spin_lambda_list[0] = lam
        
    #     uhf_obj._spin_lambda_list = tmp_spin_lambda_list
    #     uhf_obj.kernel(dm0=dm0)
    #     dm = uhf_obj.make_rdm1()
    #     s = uhf_obj._get_res(dm, ovlp_ao)[0]
    #     print("lambda = % 6.4f, res = % 6.4f" % (lam, s))

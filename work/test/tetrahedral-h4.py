import numpy, scipy
from scipy import linalg

from itertools import combinations

from pyscf import gto, scf, lo
from pyscf.tools.dump_mat import dump_rec

from bs import get_uhf_vfci
from bs import get_ump2_vfci

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
    

if __name__ == "__main__":
    for x in numpy.linspace(0.4, 2.0, 10):
        res = make_h4(x)  
        h4  = res[0]
        h4.verbose = 0

        coeff_ao_lo = res[1]

        uhf_obj = scf.UHF(h4)
        ovlp_ao = uhf_obj.get_ovlp()

        dm0_list = make_dm0(h4)
        for dm0 in dm0_list:
            uhf_obj.kernel(dm0=dm0)
            coeff_uhf  = uhf_obj.mo_coeff
            mo_occ_uhf = uhf_obj.mo_occ

            args = (coeff_ao_lo, coeff_uhf, mo_occ_uhf, ovlp_ao, uhf_obj)
            ene_bs_uhf,  vfci_bs_uhf  = get_uhf_vfci(*args)
            ene_bs_ump2, vfci_bs_ump2 = get_ump2_vfci(*args)
        
            

            
            

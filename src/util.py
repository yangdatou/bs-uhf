import sys, os
import numpy, scipy
from sys import stdout

VMD_EXEC_PATH = "/Users/yangjunjie/Applications/VMD-1.9.4a57-x86_64-Rev12.app/Contents/vmd/vmd_MACOSXX86_64"
VMD_UTIL_PATH = "/Users/yangjunjie/work/bs-uhf/src/util.vmd"

def show_cube(inp_file :str, out_file :str = "test.png", 
              rotate_x : float = 0.0, rotate_y : float = 0.0,
              rotate_z : float = 0.0, iso : float = 0.08):
    vmd_cmd = '''
    proc show_cube {filename {isoval 0.08}} {
    #default material
    set mater Glossy
    color Display Background white
    display depthcue off
    display rendermode GLSL
    axes location Off
    color Name C tan
    color change rgb tan 0.700000 0.560000 0.360000
    material change mirror Opaque 0.15
    material change outline Opaque 4.000000
    material change outlinewidth Opaque 0.5
    material change ambient Glossy 0.1
    material change diffuse Glossy 0.600000
    material change opacity Glossy 0.75
    material change shininess Glossy 1.0
    light 3 on
    foreach i [molinfo list] {
    mol delete $i
    }
    mol new $filename
    mol modstyle 0 top CPK 0.800000 0.300000 22.000000 22.000000
    mol addrep top
    mol modstyle 1 top Isosurface $isoval 0 0 0 1 1
    mol modcolor 1 top ColorID 12
    mol modmaterial 1 top $mater
    mol addrep top
    mol modstyle 2 top Isosurface -$isoval 0 0 0 1 1
    mol modcolor 2 top ColorID 22
    mol modmaterial 2 top $mater
    display distance -2.0
    display height 10
    }
    proc cubiso {isoval} {
    mol modstyle 1 top Isosurface $isoval 0 0 0 1 1
    mol modstyle 2 top Isosurface -$isoval 0 0 0 1 1
    }
    proc cub2 {filename1 filename2 {isoval 0.05}} {
    #default material
    set mater Glossy
    color Display Background white
    display depthcue off
    display rendermode GLSL
    axes location Off
    color Name C tan
    color change rgb tan 0.700000 0.560000 0.360000
    material change mirror Opaque 0.15
    material change outline Opaque 4.000000
    material change outlinewidth Opaque 0.5
    material change ambient Glossy 0.1
    material change diffuse Glossy 0.600000
    material change opacity Glossy 0.75
    material change shininess Glossy 1.0
    light 0 on
    light 1 on
    light 2 on
    light 3 on
    foreach i [molinfo list] {
    mol delete $i
    }
    mol new $filename1
    mol modstyle 0 top CPK 0.800000 0.300000 22.000000 22.000000
    mol addrep top
    mol modstyle 1 top Isosurface $isoval 0 0 0 1 1
    mol modcolor 1 top ColorID 12
    mol modmaterial 1 top $mater
    mol new $filename2
    mol modstyle 0 top CPK 0.800000 0.300000 22.000000 22.000000
    mol addrep top
    mol modstyle 1 top Isosurface $isoval 0 0 0 1 1
    mol modcolor 1 top ColorID 22
    mol modmaterial 1 top $mater
    display distance -2.0
    display height 10
    }
    proc cub2iso {isoval} {
    foreach i [molinfo list] {
    mol modstyle 1 $i Isosurface $isoval 0 0 0 1 1
    }
    }
    show_cube {%s} %6.4f
    # rotate x to %6.4f
    rotate y to %6.4f
    # rotate z to %6.4f
    render TachyonInternal {%s}
    '''%(inp_file, iso, rotate_x, rotate_y, rotate_z, out_file)

    with open("tmp.vmd", "w") as f:
        f.write(vmd_cmd)

    os.system(f"{VMD_EXEC_PATH} < tmp.vmd")

def show_h2_spin(inp_file :str, out_file :str = "test.png",
                 atom_list = None, arrow_list = None):
    vmd_cmd = f"source {VMD_UTIL_PATH}; show_mol {inp_file}\n"

    for atm_idx, arrow in zip(atom_list, arrow_list):
        fragdx = arrow[0]
        fragdy = arrow[1]
        fragdz = arrow[2]
        vmd_cmd += f'''drawarrow \"serial {atm_idx+1}\" {fragdx: 6.4f} {fragdy: 6.4f} {fragdz: 6.4f} 1.0 0.05\n'''

    vmd_cmd += f"rotate y by {90.0: 6.4f}\n"
    vmd_cmd += f"render TachyonInternal {out_file}\n"

    with open("tmp.vmd", "w") as f:
        f.write(vmd_cmd)

    os.system(f"{VMD_EXEC_PATH} < tmp.vmd")

def show_h4_spin(inp_file :str, out_file :str = "test.png",
                 atom_list = None, arrow_list = None):
    vmd_cmd = f"source {VMD_UTIL_PATH}; show_mol {inp_file}\n"

    for atm_idx, arrow in zip(atom_list, arrow_list):
        fragdx = arrow[0]
        fragdy = arrow[1]
        fragdz = arrow[2]
        vmd_cmd += f'''drawarrow \"serial {atm_idx+1}\" {fragdx: 6.4f} {fragdy: 6.4f} {fragdz: 6.4f} 2.0 0.05\n'''

    # vmd_cmd += f"rotate y by {90.0: 6.4f}\n"
    vmd_cmd += f"rotate x by {-80.0: 6.4f}\n"
    # vmd_cmd += f"display distance 10.0\n"
    vmd_cmd += f"display height 20\n"
    vmd_cmd += f"render TachyonInternal {out_file}\n"

    with open("tmp.vmd", "w") as f:
        f.write(vmd_cmd)

    os.system(f"{VMD_EXEC_PATH} < tmp.vmd")

if __name__ == "__main__":
    from pyscf import gto, scf, lib
    from pyscf.tools import cubegen
    r = 4.0
    basis = "sto3g"

    mol = gto.Mole()
    mol.atom = f"""
    H1 0.000 0.000 {( r/2.0): 12.8f}
    H2 0.000 0.000 {(-r/2.0): 12.8f}
    """
    mol.basis = basis
    mol.build()

    dms_bs = numpy.zeros((2, mol.nao, mol.nao))
    dms_bs[0, 0, 0] = 1.0
    dms_bs[1, 1, 1] = 1.0

    uhf_obj = scf.UHF(mol)
    uhf_obj.verbose = 0
    uhf_obj.kernel(dm0=dms_bs)
    assert uhf_obj.converged

    ovlp        = uhf_obj.get_ovlp()
    mo_coeff    = uhf_obj.mo_coeff
    mo_energy   = uhf_obj.mo_energy
    _, nao, nmo = mo_coeff.shape

    h1_ao_idx = mol.search_ao_label("H1")
    h2_ao_idx = mol.search_ao_label("H2")

    pauli_matrix = []
    pauli_matrix.append(numpy.array([[0.0,   1.0], [1.0,  0.0]]))
    pauli_matrix.append(numpy.array([[0.0, -1.0j], [1.0j, 0.0]]))
    pauli_matrix.append(numpy.array([[1.0,   0.0], [0.0, -1.0]]))
    pauli_matrix = numpy.asarray(pauli_matrix)
    
    coeff = uhf_obj.mo_coeff
    dm = uhf_obj.make_rdm1()

    coeff_gso = [[coeff[0], numpy.zeros_like(coeff[0])], [numpy.zeros_like(coeff[1]), coeff[1]]]
    coeff_gso = numpy.asarray(coeff_gso)
    dm_gso    = [[dm[0], numpy.zeros_like(dm[0])], [numpy.zeros_like(dm[1]), dm[1]]]
    dm_gso    = numpy.asarray(dm_gso)

    cub_dir = f"/Users/yangjunjie/work/bs-uhf/data/h2/cub/"
    fig_dir = f"/Users/yangjunjie/work/bs-uhf/data/h2/fig/"

    os.makedirs(cub_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    for s, spin in enumerate(["alph", "beta"]):
        for p in range(nmo):
            cub_file = f"{cub_dir}/h2_mo_{spin}_{p}_{mo_energy[s,p]:6.4f}.cube"
            fig_file = f"{fig_dir}/h2_mo_{spin}_{p}_{mo_energy[s,p]:6.4f}.png"
            cubegen.orbital(mol, cub_file, mo_coeff[s][:,p])

    theta_list = numpy.linspace(0.0, numpy.pi, 21)

    for t in theta_list:
        cos_t = numpy.cos(t)
        sin_t = numpy.sin(t)
        rotate_matrix = numpy.array([[cos_t, sin_t], [-sin_t, cos_t]])
        coeff_gso_rot = numpy.einsum("abmn,bc->acmn", coeff_gso, rotate_matrix)

        occ_list = [0]
        coeff_gso_occ = coeff_gso_rot[:, :, :, occ_list]
        dm_gso_rot    = numpy.einsum("abmi,bcni->acmn", coeff_gso_occ, coeff_gso_occ)

        atom_list  = []
        arrow_list = []
        for iatm, iatm_ao_idx in enumerate([h1_ao_idx, h2_ao_idx]):
            dm_gso_iatm = dm_gso_rot[:, :, iatm_ao_idx, :][:, :, :, iatm_ao_idx]
            ovlp_iatm   = ovlp[iatm_ao_idx, :][:, iatm_ao_idx]
            s_exp = numpy.einsum("abmn,xab,mn->x", dm_gso_iatm, pauli_matrix, ovlp_iatm).real

            atom_list.append(iatm)
            arrow_list.append(s_exp)
        
        print(f"t = {t:6.4f}, s_exp = {s_exp}")

        show_arrow(
            inp_file="./cub/alph/h2_mo_0_-0.4666.cube", out_file=f"./png/t_{t:6.4f}.png", atom_list=atom_list, arrow_list=arrow_list, rotate_x=90.0, rotate_y=90.0, rotate_z=0.0
            )
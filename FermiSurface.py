#!/usr/bin/env python

import os
import numpy as np
from ase.io import read

def find_fermi_level(band_energies, kpt_weight,
                     nelect, occ=None, sigma=0.01, nedos=100,
                     soc_band=False,
                     nmax=1000):
    '''
    Locate ther Fermi level from the band energies, k-points weights and number
    of electrons. 
                                             1.0
                Ne = \sum_{n,k} --------------------------- * w_k
                                  ((E_{nk}-E_f)/sigma) 
                                e                      + 1


    Inputs:
        band_energies: The band energies of shape (nspin, nkpts, nbnds)
        kpt_weight: The weight of each k-points.
        nelect: Number of electrons.
        occ:  1.0 for spin-polarized/SOC band energies, else 2.0.
        sigma: Broadening parameter for the Fermi-Dirac distribution.
        nedos: number of discrete points in approximately locating Fermi level.
        soc_band: band energies from SOC calculations?
        nmax: maximum iteration in finding the exact Fermi level.
    '''

    if band_energies.ndim == 2:
        band_energies = band_energies[None, :]

    nspin, nkpts, nbnds = band_energies.shape

    if occ is None:
        if nspin == 1 and (not soc_band):
            occ = 2.0
        else:
            occ = 1.0

    if nbnds > nedos:
        nedos = nbnds * 5

    kpt_weight = np.asarray(kpt_weight, dtype=float)
    assert kpt_weight.shape == (nkpts,)
    kpt_weight /= np.sum(kpt_weight)

    emin = band_energies.min()
    emax = band_energies.max()
    e0 = np.linspace(emin, emax, nedos)
    de = e0[1] - e0[0]

    # find the approximated Fermi level
    nelect_lt_en = np.array([
        np.sum(occ * (band_energies <= en) * kpt_weight[None, :, None])
        for en in e0
    ])
    ne_tmp = nelect_lt_en[nedos//2]
    if (np.abs(ne_tmp - nelect) < 0.05):
        i_fermi = nedos // 2
        i_lower = i_fermi - 1
        i_upper = i_fermi + 1
    elif (ne_tmp > nelect):
        for ii in range(nedos//2-1, -1, -1):
            ne_tmp = nelect_lt_en[ii]
            if ne_tmp < nelect:
                i_fermi = ii
                i_lower = i_fermi
                i_upper = i_fermi + 1
                break
    else:
        for ii in range(nedos//2+1, nedos):
            ne_tmp = nelect_lt_en[ii]
            if ne_tmp > nelect:
                i_fermi = ii
                i_lower = i_fermi - 1
                i_upper = i_fermi
                break

    ############################################################
    # Below is the algorithm used by VASP, much slower
    ############################################################
    # find the approximated Fermi level
    # x = (e0[None, None, None, :] - band_energies[:, :, :, None]) / sigma
    # x = x.clip(-100, 100)
    # dos = 1./sigma * np.exp(x) / (np.exp(x) + 1)**2 * \
    #       kpt_weight[None, :, None, None] * de
    # ddos = np.sum(dos, axis=(0,1,2))
    #
    # nelect_from_dos_int = np.sum(ddos[:nedos/2])
    # if (np.abs(nelect_from_dos_int - nelect) < 0.05):
    #     i_fermi = nedos / 2 - 1
    #     i_lower = i_fermi - 1
    #     i_upper = i_fermi + 1
    # elif (nelect_from_dos_int > nelect):
    #     for ii in range(nedos/2, -1, -1):
    #         nelect_from_dos_int = np.sum(ddos[:ii])
    #         if nelect_from_dos_int < nelect:
    #             i_fermi = ii
    #             i_lower = i_fermi
    #             i_upper = i_fermi + 1
    #             break
    # else:
    #     for ii in range(nedos/2, nedos):
    #         nelect_from_dos_int = np.sum(ddos[:ii])
    #         if nelect_from_dos_int > nelect:
    #             i_fermi = ii
    #             i_lower = i_fermi - 1
    #             i_upper = i_fermi
    #             break

    # Locate the exact Fermi level using bisectioning
    e_lower = e0[i_lower]
    e_upper = e0[i_upper]
    lower_B = False
    upper_B = False
    for ii in range(nmax):
        e_fermi = (e_lower + e_upper) / 2.

        z = (band_energies - e_fermi) / sigma
        z = z.clip(-100, 100)
        F_nk = occ / (np.exp(z) + 1)
        N = np.sum(F_nk * kpt_weight[None, :, None])
        # print ii, e_lower, e_upper, N

        if (np.abs(N - nelect) < 1E-10):
            break
        if (np.abs(e_upper - e_lower / (np.abs(e_fermi) + 1E-10)) < 1E-14):
            raise ValueError("Cannot reach the specified precision!")

        if (N > nelect):
            if not lower_B:
                e_lower -= de
            upper_B = True
            e_upper = e_fermi
        else:
            if not upper_B:
                e_upper += de
            lower_B = True
            e_lower = e_fermi

    if (ii == nmax - 1):
        raise ValueError("Cannot reach the specified precision!")

    return e_fermi, F_nk

def get_brillouin_zone_3d(cell):
    """
    Generate the Brillouin Zone of a given cell. The BZ is the Wigner-Seitz cell
    of the reciprocal lattice, which can be constructed by Voronoi decomposition
    to the reciprocal lattice.  A Voronoi diagram is a subdivision of the space
    into the nearest neighborhoods of a given set of points. 

    https://en.wikipedia.org/wiki/Wigner%E2%80%93Seitz_cell
    https://docs.scipy.org/doc/scipy/reference/tutorial/spatial.html#voronoi-diagrams
    """

    cell = np.asarray(cell, dtype=float)
    assert cell.shape == (3, 3)

    px, py, pz = np.tensordot(cell, np.mgrid[-1:2, -1:2, -1:2], axes=[0,0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    from scipy.spatial import Voronoi
    vor = Voronoi(points)
    
    bz_facets = []
    bz_ridges = []
    bz_vertices = []

    # for rid in vor.ridge_vertices:
    #     if( np.all(np.array(rid) >= 0) ):
    #         bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
    #         bz_facets.append(vor.vertices[rid])

    for pid, rid in zip(vor.ridge_points, vor.ridge_vertices):
        # WHY 13 ????
        # The Voronoi ridges/facets are perpendicular to the lines drawn between the
        # input points. The 14th input point is [0, 0, 0].
        if( pid[0] == 13 or pid[1] == 13 ):
            bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
            bz_facets.append(vor.vertices[rid])
            bz_vertices += rid
            
    bz_vertices = list(set(bz_vertices))

    return  vor.vertices[bz_vertices], bz_ridges, bz_facets
    
class ebands3d(object):
    '''
    '''
    
    def __init__(self, inf='EIGENVAL', efermi=None, kmesh=[], symprec=1E-5):
        '''
        Init
        '''

        self._fname = inf
        # the directory containing the input file
        self._dname  = os.path.dirname(inf)
        if self._dname == '':
            self._dname = '.'

        # read bands, k-points of the irreducible Brillouin Zone
        self.read_eigenval()
        # set the Fermi energy
        self.set_efermi(efermi)
        # set the k-points mesh
        self.set_kmesh(kmesh)
        # read POSCAR
        self.atoms = read(self._dname + "/POSCAR")
        # create the grid to ir map
        self.ir_kpts_map(symprec=symprec)

    def get_fermi_ebands3d(self):
        '''
        '''

        nx, ny, nz = self.kmesh
        self.fermi_ebands3d = []
        for ispin in range(self.nspin):
            tmp = []
            for iband in self.fermi_xbands[ispin]:
                etmp = self.ir_ebands[ispin, self.grid_to_ir_map, iband]
                etmp.shape = (nx, ny, nz)
                tmp.append(etmp)

            self.fermi_ebands3d.append(tmp)


    def to_bxsf(self, prefix='ebands3d', ncol=6):
        '''
        Output the ebands3d as Xcrysden .bxsf format.
        '''

        with open(self._dname + '/{:s}.bxsf'.format(prefix), 'w') as out:
            out.write("BEGIN_INFO\n")
            out.write("  # Launch as: xcrysden --bxsf ebands3d.bxsf\n")
            out.write("  Fermi Energy: {:12.6f}\n".format(self.efermi))
            out.write("END_INFO\n\n")

            out.write("BEGIN_BLOCK_BANDGRID_3D\n")
            out.write("  band_energies\n")
            out.write("  BANDGRID_3D_BANDS\n")

            number_fermi_xbands = sum([len(xx) for xx in self.fermi_xbands])
            out.write("    {:d}\n".format(number_fermi_xbands))
            # number of data-points in each direction (i.e. nx ny nz for 3D gr)
            out.write("    {:5d}{:5d}{:5d}\n".format(*(x for x in self.kmesh)))
            # origin of the bandgrid.
            # Warning: origin should be (0,0,0) (i.e. Gamma point)
            out.write("    {:16.8f}{:16.8f}{:16.8f}\n".format(0.0, 0.0, 0.0))
            # Reciprocal lattice vector
            out.write(
                    '\n'.join(["    " + ''.join(["%16.8f" % xx for xx in row])
                               for row in self.atoms.get_reciprocal_cell()])
                    )

            # write the band grid for each band, the values inside a band grid
            # are specified in row-major (i.e. C) order. This means that values
            # are written as:
            #
            # C-syntax:
            #   for (i=0; i<nx; i++)
            #     for (j=0; j<ny; j++)
            #       for (k=0; k<nz; k++)
            #         printf("%f",value[i][j][k]);
            #
            # FORTRAN syntax:
            #   write(*,*) (((value(ix,iy,iz),iz=1,nz),iy=1,ny),ix=1,nx)

            for ispin in range(self.nspin):
                sign = 1 if ispin == 0 else -1
                for ii in range(len(self.fermi_xbands[ispin])):
                    iband = self.fermi_xbands[ispin][ii]
                    b3d = self.fermi_ebands3d[ispin][ii].copy()
                    nx, ny, nz = b3d.shape
                    b3d.shape = (nx * ny, nz)

                    out.write("\n    BAND: {:5d}\n".format(iband * sign))
                    out.write(
                            '\n'.join(["    " + ''.join(["%16.8e" % xx for xx in row])
                                       for row in b3d])
                            )
                    # np.savetxt(out, b3d, fmt='%16.8E')

            out.write("\n  END_BANDGRID_3D\n")
            out.write("END_BLOCK_BANDGRID_3D\n")

    def ir_kpts_map(self, symprec=1E-5):
        '''
        Get irreducible k-points in BZ and the mapping between the mesh points
        and the ir kpoints.
        '''

        import spglib

        cell = (
                self.atoms.cell,
                self.atoms.get_scaled_positions(),
                self.atoms.get_atomic_numbers()
            )
        # mapping: a map between the grid points and ir k-points in grid
        mapping, grid = spglib.get_ir_reciprocal_mesh(self.kmesh, cell,
                is_shift=[0, 0, 0], symprec=symprec)

        self.kmesh_grid = grid
        ir_kpts = grid[np.unique(mapping)] / self.kmesh.astype(float)
        assert (ir_kpts.shape == self.ir_kpath.shape) and \
               (np.allclose(self.ir_kpath, ir_kpts)), \
               "Irreducible k-points generated by Spglib and VASP inconsistent!\n Try to reduce symprec, e.g. 1E-4."

        uniq_grid_index = np.unique(mapping)
        dump = np.zeros((grid.shape[0]), dtype=int)
        dump[uniq_grid_index] = np.arange(uniq_grid_index.size, dtype=int)
        # mapping between the grid points and the ir k-points
        self.grid_to_ir_map = dump[mapping]

    def set_kmesh(self, kmesh):
        '''
        Set the k-points mesh, Read from KPOINTS if not given.
        '''

        if kmesh:
            self.kmesh = kmesh
        else:
            with open(self._dname + "/KPOINTS") as k:
                dat = [l.split() for l in k if l.strip()]
                assert int(dat[1][0]) == 0, "Not automatic k-mesh generation in KPOINTS!"
                assert dat[2][0][0].upper() == 'G', "Please use Gamma center mesh!"
                self.kmesh = np.array([int(x) for x in dat[3]], dtype=int)
                self.kshift = np.array([float(x) for x in dat[4]], dtype=float)
                assert np.allclose(self.kshift, [0,0,0]), "K mesh shift should be 0!"

    def set_efermi(self, efermi):
        '''
        Set a new Fermi energy.
        '''

        if efermi is None:
            self.efermi, _ = find_fermi_level(self.ir_ebands, self.ir_kptwt, self.nelect)
        else:
            self.efermi = efermi
        self.find_fermicrossing_bands()

    def find_fermicrossing_bands(self):
        '''
        Find the index of the bands that cross the Fermi level.
        '''

        band_index = np.arange(self.nbnds, dtype=int)

        band_energy_max = np.max(self.ir_ebands, axis=1)
        band_energy_min = np.min(self.ir_ebands, axis=1)
        fermi_cross_band = (band_energy_min < self.efermi) & (self.efermi < band_energy_max)
        
        self.fermi_xbands = [band_index[fermi_cross_band[ii]] for ii in range(self.nspin)]
        if not np.all([True if x else False for x in self.fermi_xbands]):
            raise ValueError("No surface found at {:8.4f} eV!".format(self.efermi))
        
    def read_eigenval(self):
        '''
        Read band energies from VASP EIGENVAL file.
        '''

        with open(self._fname) as inp:
            # read all the data.
            dat = np.array([line.strip() for line in inp if line.strip()])

            # extract the needed info in the header.
            self.nspin = int(dat[0].split()[-1])
            self.nelect, self.ir_nkpts, self.nbnds = map(int, dat[5].split())

            # remove the header
            dat = dat[6:]

            # extract the k-points info
            dump = np.array([xx.split() for xx in dat[::self.nspin * self.nbnds + 1]], dtype=float)
            self.ir_kpath = dump[:self.ir_nkpts, :3]
            self.ir_kptwt = dump[:self.ir_nkpts, -1]

            # extract the ebands info
            ebands_flag = np.ones(dat.size, dtype=bool)
            ebands_flag[::self.nspin * self.nbnds + 1] = 0
            if self.nspin == 1:
                ebands = np.array([xx.split()[1] for xx in dat[ebands_flag]],
                        dtype=float)
            else:
                ebands = np.array([xx.split()[1:3] for xx in dat[ebands_flag]],
                        dtype=float)
            ebands.shape = (self.ir_nkpts, self.nspin, self.nbnds)
            self.ir_ebands = ebands.swapaxes(0, 1)

if __name__ == "__main__":
    """
    TEST
    """

    # cell = np.array([
    #     [np.sqrt(3)/2., 0.5, 0.0],
    #     [0.000000000,   1.0,  0.000000000],
    #     [0.000000000,   0.000000000,  1.0]])
    #
    # cell = np.sqrt(2) * np.array([
    #     [0.5, 0.5, 0.0],
    #     [0.0, 0.5, 0.5],
    #     [0.5, 0.0, 0.5]])
    #
    # # print(np.linalg.det(cell))
    # # cell = np.eye(3)
    #
    # p, l, f = get_brillouin_zone_3d(cell)
    #
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.set_aspect('equal')
    #
    # for xx in l:
    #     ax.plot(xx[:,0], xx[:,1], xx[:,2], color='r', alpha=0.5, lw=1.0)
    #
    # art = Poly3DCollection(f, facecolor='k', alpha=0.1)
    # ax.add_collection3d(art)
    #
    # ax.set_xlim(-1,1)
    # ax.set_ylim(-1,1)
    # ax.set_zlim(-1,1)
    #
    # plt.show()

    # kpath, kptwt, ebands, nelect = read_eigenval()
    # efermi, fnk = find_fermi_level(ebands, kptwt, nelect, sigma=0.1)
    # print("Fermi Energy: {:8.4f} eV".format(efermi))

    xx = ebands3d()
    xx.get_fermi_ebands3d()
    xx.to_bxsf()
    

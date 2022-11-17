import numpy as np
import subprocess
import re
import os
import functools
import time
import fortranformat as ff
from pyscf import gto, scf, qmmm, lib

from vqe_qmmm.util import utils
from vqe_qmmm.vqe import vqemcscf
from vqe_qmmm.resp import resp
from vqe_qmmm.qmmm import electrostatic

_hartree2kcal = 627.5095
_hartree2kj = 627.5095*4.184
_hartree2kj = 2625.4996394798765 #RISIMICAL
_hartree2kcal = 627.509474063068  # RISMICAL

def stopwatchPrint(func):

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    print(f"{func.__name__}: {end - start:.3f} s.")
    return result

  return wrapper

def gen_grid(N,dr):
    x = np.mgrid[-(N)/2:(N-1)/2]*dr
    y = np.mgrid[-(N)/2:(N-1)/2]*dr
    z = np.mgrid[-(N)/2:(N-1)/2]*dr

    grid_coords = np.array(np.meshgrid(x,y,z)).T.reshape(-1,3)

    return grid_coords # in angstrom

def read_qv(fname):
    fd = open(fname,'r')

    _re_comments = re.compile(r'^#')
    _re_num_points = re.compile(r"Number\sof\spoints:")

    charge_list = []
    npoints = 0
    idx = 0
    for line in fd:
        if _re_comments.match(line) is None:
            tokens = line.split()
            if len(tokens) == 4:
                x = float(tokens[0])
                y = float(tokens[1])
                z = float(tokens[2])
                q = float(tokens[3])
                charge_list.append([x,y,z,q])
        elif _re_num_points.search(line) is not None:
            tokens = line.split()
            npoints = int(tokens[-1])

    assert npoints == len(charge_list)

    charge_arrays = np.array(charge_list)

    return charge_arrays #(npoints, 4)

@stopwatchPrint
def write_esp(mf, fname, grid_coords, resp_charges,
              potentials, dr, cwf=None):

    resp_line = ff.FortranRecordWriter('(D20.10)')
    grid_coord_line = ff.FortranRecordWriter('(F14.4,F12.4,F12.4)')
    grid_pot_line = ff.FortranRecordWriter('(E19.8)')

    fd = open(fname,'w')

    fd.write("## RESP charges  \n")
    fd.write("%i\n"%len(resp_charges[:-1]))
    for iresp in resp_charges[:-1]:
        fd.write(resp_line.write([iresp]))
        fd.write("\n")
    npoints = grid_coords.shape[0]
    fd.write("## Grid        %i   %f\n" %(np.cbrt(npoints), dr))
    fd.write("##  Electrostatic potential\n")
    fd.write("##  REMARKS    2\n")
    fd.write("##     Number of points:          %i\n" %npoints)
    fd.write("##     x[Ang]      y[Ang]      z[Ang]          V[J/mol/e]\n")
    potentials *= _hartree2kj*1000.0
    for icoord, ivpot in zip(grid_coords, potentials):
        fd.write(grid_coord_line.write(icoord))
        if (np.isnan(ivpot) or np.isinf(ivpot)):
          ivpot = 0.0
        fd.write(grid_pot_line.write([ivpot]))
        fd.write("\n")

    return 0

class pyscf_3drism:


    def __init__(self,
                 myinput):
        """initialize rism-pyscf

        Args:
          myinput: a dictionary object, containing input information

        """
        #
        # load input file information
        #
        self._input = myinput
        if "base" in self._input:
            self.basename = self._input["base"]["name"]
        else:
            print ("Error: \"base\" is mandatory!")
            quit()
        # rism
        self.ngrid3d = 128
        self.rdelta3d = 0.5
        self.ljparam = None
        self.rism_inp_file = None
        if "rism" in self._input:
            self._set_grid_info()
        # io
        self.geom_file = None
        self.esp_file = None
        self._set_file_name()
        # geometry
        self.geom = None
        self._set_geom()
        # pyscf information
        self.charge = 0
        self.spin = 0
        self.basis = 'sto-3g'
        self.method = "vqe"
        self.norb = 2
        self.nelec = 2
        self.doGrad = False
        self.doRI = False
        self.doShift_to_centroid  = False
        self.verbose = 1
        if "pyscf" in self._input:
            self._set_pyscf_info()
        #
        # Generate Grid
        #
        self.grid_coords = None
        self._gen_grid()
        self.npoints = self.grid_coords.shape[0]
        #
        # Set Hartree-Fock object
        #
        self.mol = gto.M(atom=self.geom,
                         basis=self.basis,
                         charge=self.charge,
                         spin=self.spin)
        #self.mol.cart = False
        if self.doShift_to_centroid:
          self._shift_to_centroid()
        if self.doRI:
          self.mf = scf.RHF(self.mol).density_fit()
        else:
          self.mf = scf.RHF(self.mol)
        self.mf.verbose = self.verbose
        self.mf.conv_tol = 1.0e-8
        #print ("N-N Replusion", gto.energy_nuc(self.mol))
        self.cwf = None
        #
        self.resp_charges = None
        self.vpot_mm = None
        self.vpot_qm = None
        self.solv_ene = [] #Solvation Free Energy
        self.qm_ene = [] #
        self.rism_scf_ene = []
        self.isConv = False
        self.thresh_ene = 5.0e-04
        self.rism_charges = None
        #
        self.mf_qmmm = None

    @stopwatchPrint
    def kernel(self):

        # 1. Perform Initial Hartree-Fock caluculation
        self.mf.chkfile = self.pyscf_chk_file
        self.mf.run()
        # 2. Perform Initial VQE calculations
        if self.method == 'vqe':
            self.run_vqeci()
            self.qm_ene.append(self.cwf.e_tot+0.0)
            self.rism_scf_ene.append(self.cwf.e_tot+0.0)
        else:
            self.qm_ene.append(self.mf.e_tot+0.0)
            self.rism_scf_ene.append(self.mf.e_tot+0.0)

        # 3. Compute initial RESP and ESP. Then, save them.
        self.compute_and_save_esp_resp()
        print (self.mf.e_tot)
        print (self.mf.dip_moment(unit='Debye'))
        print (self.mf.energy_elec())
        print (self.mf.energy_elec()[0]+self.mf.energy_nuc())
        print (self.mf.energy_tot())
        print ('TWO ELECTRON ENERGY =%20.10f' %self.mf_qmmm.energy_elec()[1])


        print ("Enter RISM-SCF loop")
        # LOOP Start
        idx = 0
        while self.isConv == False:
            idx = idx + 1
            print ("\n\n###############################")
            print ("### RISM-SCF iteration %5i###" %idx)
            print ("###############################\n\n")
            #
            # 4. perform the RISM,
            # extract solvation free energy and RISM_MM charges
            #
            self.get_solv_free_energy()
            #self.rism_mm_charges = read_qv(self.rism_qv_file)
            self.rism_mm_charges = self.get_rism_mm_charges()

            #
            # 5. perform VQE with RISM-MM charges
            #
            self.run_vqe_with_rism_mm_charges()

            #
            # 6. check convergence & if yes exit
            #
            print ("\nRISM-SCF Energy in hartree at %i-th iteration " %idx)
            print ("%15.5f %10.5e" %(self.rism_scf_ene[-1],
                                     self.rism_scf_ene[-1]
                                     -self.rism_scf_ene[-2]))
            print ("without solvation free energy")
            print ("%15.5f %10.5e\n\n" %(self.qm_ene[-1],
                                     self.qm_ene[-1]
                                     -self.qm_ene[-2]))
            if np.abs(self.rism_scf_ene[-1]
                      - self.rism_scf_ene[-2]) < self.thresh_ene:
                print ("RISM-SCF is get converged!!")
                print ("Convergence behavior of RISM-SCF energy")
                print ("with Solvation Free Energy")
                print ("Iteration  Energy(hartree)    Energy(kcal/mol)")
                for idx, iene in enumerate(self.rism_scf_ene):
                    print ("%5i %15.5f %15.5f" %(idx,iene,iene*_hartree2kcal))
                print ("")
                print ("without Solvtion Free Energy")
                print ("Iteration  Energy(hartree)    Energy(kcal/mol)")
                for idx, iene in enumerate(self.qm_ene):
                    print ("%5i %15.5f %15.5f" %(idx,iene,iene*_hartree2kcal))
                self.isConv = True

            #
            # 8. comptue electrostatic potentials & RESP charges
            #
            self.compute_and_save_esp_resp()
        # Loop End

    def get_rism_mm_charges(self):
        qv_charges = read_qv(self.rism_qv_file)
        #print (qv_charges.shape)
        for i in range(self.mol.natm):
          r = lib.norm(self.mol.atom_coord(i)-qv_charges[:,:3],
                       axis=1)
          qv_charges = np.delete(qv_charges, np.where(r < 0.1e-0)[0], axis=0)
        #print (qv_charges.shape)
        return qv_charges

    @stopwatchPrint
    def get_solv_free_energy(self):
        # RUN RISM
        self.run_rism()
        # Extract Solvation Free Energy
        fd = open(self.rism_log_file,'r')
        regex = re.compile("Solvation Free Energy         :")
        for iline in fd:
            if re.search(regex, iline):
                terms = iline.split()
                solv_ene = float(terms[4])
                solv_ene /= _hartree2kj
                self.solv_ene.append(solv_ene) # in hartree

    @stopwatchPrint
    def run_rism(self):
        print ("Enter RISM")
        with open('rismical.log', 'w') as Fd:
          rism_process = subprocess.run(['rismical.x',"3d",self.rism_inp_file], stdout=Fd)


    @stopwatchPrint
    def run_vqeci(self):
        self.mf.run()
        self.cwf = vqemcscf.VQECASCI(self.mf,
                                     self.norb,
                                     self.nelec)
        self.cwf.kernel()
        self.qm_ene.append(self.cwf.e_tot)

    @stopwatchPrint
    def run_vqe_with_rism_mm_charges(self):
            self.mf_qmmm =  qmmm.mm_charge(self.mf,
                                           self.rism_mm_charges[:,:3],
                                           self.rism_mm_charges[:,-1],
                                           unit='Agnstrom')
            self.mf_qmmm.chkfile = self.pyscf_chk_file
            self.mf_qmmm.init_guess = 'chk'
            self.mf_qmmm.run()
            self.mf_qmmm.energy_nuc = self.mf.energy_nuc
            print (self.mf_qmmm.dip_moment(unit='Debye'))
            #print (self.mf_qmmm.energy_elec())
            #print (self.mf_qmmm.energy_elec()[0]+self.mf.energy_nuc())
            print ("RHF-ENERGY: ", self.mf_qmmm.energy_tot())
            print ('TWO ELECTRON ENERGY =%20.10f' %self.mf_qmmm.energy_elec()[1])
            print ('Solvation Free energy (kcal/mol)', self.solv_ene[-1]*_hartree2kcal)
            #quit()
            #
            mf = scf.RHF(self.mf.mol)
            mf.__dict__.update(scf.chkfile.load(self.pyscf_chk_file, 'scf'))
            print ('\n<HF|H_{iso}|HF>+(solvation free energy)', mf.energy_tot()+self.solv_ene[-1])
            print ('<HF|H_{iso}|HF>', mf.energy_tot())
            print ('<HF|V_solv|HF>\n', self.mf_qmmm.energy_tot()-mf.energy_tot())
            #quit()
            if self.method == 'vqe':
               self.cwf = vqemcscf.VQECASCI(self.mf_qmmm,
                                            self.norb,
                                            self.nelec)
               self.cwf.kernel()
               dm1 = self.cwf.make_rdm1()
               e_rism_qm = self.mf_qmmm.energy_tot(dm1)-mf.energy_tot(dm1)
               print ('<VQE|V_solv|VQE>\n', e_rism_qm)
               e_vqe_iso = self.cwf.e_tot - e_rism_qm
               print ('<VQE|H_{iso}|VQE>', e_vqe_iso)
               print ('<VQE|H_{iso}|VQE>+(solvation free energy)', e_vqe_iso+self.solv_ene[-1])
               #quit()
               self.qm_ene.append(self.cwf.e_tot)
               self.rism_scf_ene.append(self.cwf.e_tot
                                        +self.solv_ene[-1])
            else:
               self.qm_ene.append(self.mf_qmmm.energy_tot())
               self.rism_scf_ene.append(self.mf_qmmm.energy_tot()
                                        +self.solv_ene[-1])

    @stopwatchPrint
    def compute_and_save_esp_resp(self):
        #
        # Compute the electrostatic potentials at mm charges
        #
        if self.mf_qmmm is None:
          self.mf_qmmm = self.mf
        mf_qmmm =  qmmm.mm_charge(self.mf_qmmm,
                                  self.grid_coords,
                                  np.zeros(self.npoints),
                                  unit='Agnstrom')
        self.vpot_mm = electrostatic.vpot_pyscf(mf_qmmm, cwf=self.cwf) #MM regions
        #
        # Compute the RESP charges
        #
        self.resp_charges = resp.compute_resp(mf_qmmm, cwf=self.cwf) #do not use mf_qmmm?
        print ("RESP CHARGES", self.resp_charges)
        q = 0.0 
        for iq in self.resp_charges[:-1]:
          q += iq
        print ("Sum of RESP Charges", q)
#        quit()
#        self.resp_charges[0] = -0.6168
#        self.resp_charges[1] =  0.3084
#        self.resp_charges[2] =  0.3084
#        self.resp_charges[3] =  0.0000
#
        #
        # Write down the ESP charges and electrostatic potentials
        #
        write_esp(self.mf, self.esp_file,
                  self.grid_coords, self.resp_charges,
                  self.vpot_mm,
                  self.rdelta3d, cwf=self.cwf)

    def _gen_grid(self):
        self.grid_coords = gen_grid(self.ngrid3d,
                                    self.rdelta3d)

    def _set_geom(self):
        fd = open(self.geom_file)
        self.geom = utils.read_xyz(fd)


    def _set_file_name(self):
        if "geometry" in self._input["base"]:
            self.geom_file = self._input["base"]["geometry"]
        else:
            self.geom_file = self.basename + ".xyz"
        if "esp" in self._input["base"]:
            self.esp_file = self._input["base"]["esp"]
        else:
            self.esp_file = self.basename + ".esp"
        if "rism" in self._input:
            if 'ljparam' in self._input["rism"]:
                self.ljparam = self._input["rism"]["ljparam"]
            else:
                self.ljparam = self.basename + ".lj"
        self.rism_qv_file = self.basename + ".qv"
        self.rism_log_file = 'rismical.log'
        self.rism_inp_file = self.basename + ".inp"
        self.pyscf_chk_file = self.basename + ".pyscf.chk"

    def _set_grid_info(self):
        if "grid" in self._input["rism"]:
           grid_info = self._input["rism"]["grid"]
           if type(grid_info) is str:
               if grid_info.lower() == 'fine':
                   self.ngrid3d = 256
                   self.rdelta3d = 0.25
               elif grid_info.lower() == 'std':
                   self.ngrid3d = 128
                   self.rdelta3d = 0.5
               elif grid_info.lower() == 'test':
                   self.ngrid3d = 64
                   self.rdelta3d = 1.0
           if "ngrid3d" in grid_info:
               self.ngrid3d = int(grid_info["ngrid3d"])
           if "rdelta3d" in grid_info:
               self.rdelta3d = float(grid_info["rdelta3d"])


    def _set_pyscf_info(self):
        #
        pyscf_info = self._input["pyscf"]
        if "basis" in pyscf_info:
            self.basis = pyscf_info["basis"]
        if "charge" in pyscf_info:
            self.charge = int(pyscf_info["charge"])
        if "spin" in pyscf_info:
            self.spin = int(pyscf_info["spin"])
        if "active electrons" in pyscf_info:
            self.nelec = int(pyscf_info["active electrons"])
        if "active orbitals" in pyscf_info:
            self.norb = int(pyscf_info["active orbitals"])
        if "verbose" in pyscf_info:
            self.verbose = int(pyscf_info["verbose"])
        if "method" in pyscf_info:
            self.method = (pyscf_info["method"]).lower()
        if "compute grad" in pyscf_info:
            if (pyscf_info["compute grad"]).lower() == 'false':
                self.doGrad = False
            else:
                self.doGrad = True
        if "density fit" in pyscf_info:
            if (pyscf_info["density fit"]).lower() == 'false':
                self.doRI = False
            else:
                self.doRI = True
        if "centroid" in pyscf_info:
            if (pyscf_info["centroid"]).lower() == 'false':
                self.doShift_to_centroid = False
            else:
                self.doShift_to_centroid = True

    def _shift_to_centroid(self):
      self.mol = utils.set_centroid(self.mol)
      if os.path.isfile(self.geom_file):
        os.rename(self.geom_file, self.geom_file+".bak")
      utils.write_xyz(self.geom_file, self.mol,
                  comments='The solute molecule has been translated so that its centroid is origin.')



if __name__=="__main__":

    import sys
    import json
    # 最初の引数でインプットファイルを渡す
    input_fname = sys.argv[1]
    # インプットファイルはjson形式を使うこととする
    # json形式のファイルをdictionary形式に変換する
    myinput = json.load(open(input_fname))
    #
    mycalc = pyscf_3drism(myinput)
    mycalc.kernel()

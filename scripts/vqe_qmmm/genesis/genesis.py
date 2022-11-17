from vqe_qmmm.util import gaussian_parser


def get_basename_of_inp_file(fname):
   basename = ''
   for iterm in fname.split('.')[:-1]:
       basename += iterm + '.'
    return basename[:-1]



class qmmm_genesis_pyscf():

    def __init__(self, gaussian_input_fname, pyscf_options):
        #
        self._set_file_name(gaussian_input_fname)

        #
        self._input = pyscf_options
        self._set_pyscf_options()

        # Parse Gaussian-style input file
        (self.qm_geometry,
         self.mm_pos,
         self.mm_charges) = gaussian_parser.get_coords_from_gaussian_input(fname)

        # Set a PySCF qmmm object
        self._set_pyscf_qmmm()


    def kernel(self):
        self.mf_qmmm.kernel()
        if self.method == 'vqe':
            self.cwf = vqemcscf.VQECASCI(self.mf_qmmm,
                                         2, 2)
            self.cwf.kernel()

    def compute_esp_resp_field_dip(self):
        self.q_vqe = resp.compute_resp(self.mf,
                                  self.cwf) #do not use mf_qmmm
        # compute the electrostatic potentials at mm charges
        self.vpot = electrostatic.vpot_pyscf(self.mf_qmmm,
                                        self.cwf) #MM regions
        self.vpot_qm = electrostatic.vpot_qm_pyscf(self.mf_qmmm,
                                              self.cwf) #QM regions
        self.vpot_all = np.concatenate([vpot_qm,vpot])

        # compute_electric_field
        self.field = electrostatic.compute_electric_field(self.mf_qmmm,
                                                          self.cwf)

        # compute self-energy of the charges
        self.mm_self_energy = electrostatic.compute_mm_selfenergy(self.mf_qmmm,
                                                                  self.cwf)

        # get dipole moment in atomic unit
        self.dip = self.mf_qmmm.dip_moment(unit='AU')


    def write_gaussian_style_log(self):
        fd = open(self.gaussian_log_file,'w')
        fd.write("Self energy of the charges =  %15.10f a.u. \n"%self.mm_self_energy)
        fd.write("    Center     Electric         -------- Electric Field --------\n")
        fd.write("               Potential          X             Y             Z\n")
        fd.write("-----------------------------------------------------------------\n")
        idx = 0
        for ifield, ivpot in zip(self.field, self.vpot_all):
            idx += 1
            print ("%5i     %14.6f%14.6f%14.6f%14.6f"%(idx,ivpot,
                                                       ifield[0],
                                                       ifield[1],
                                                       ifield[2]))
        fd.write("-----------------------------------------------------------------\n")
        fd.write("Normal termination of Gaussian 09 at xxxxx (well, actually, PySCF)")



    def write_gaussian_style_fchk(self):
        fd = open(self.gaussian_fchk_file,'w')

        self.tot_ene = self.mm_self_energy + self.
        print ("Total Energy                               R    %23.15E\n"%self.tot_ene)
        nesp = len(self.resp_charges[:-1])
        print ('ESP Charges                                R   N=%12i'%nesp)
        nline = nesp//5
        for iline in range(nline):
            ioff = iline*5
            print ('%16.8E%16.8E%16.8E%16.8E'%(q_vqe[ioff],
                                               q_vqe[ioff+1],
                                               q_vqe[ioff+2],
                                               q_vqe[ioff+3],
                                               q_vqe[ioff+4]))
        lastline = ''
        for iterm in range(np.mod(nesp,5)):
            ioff = nline*5
            lastline += '%16.8E'%(q_vqe[ioff+iterm])
        print (lastline)

        self.gradient = mc_g[1].flatten()
        ngrad = len(self.gradient)
        print ('Cartesian Gradient                         R   N=%12i'%ngrad)
        nline = ngrad//5
        for iline in range(nline):
            ioff = iline*5
            print ('%16.8E%16.8E%16.8E%16.8E%16.8E'%(self.gradient[ioff],
                                               self.gradient[ioff+1],
                                               self.gradient[ioff+2],
                                               self.gradient[ioff+3],
                                               self.gradient[ioff+4]))
        lastline = ''
        for iterm in range(np.mod(ngrad,5)):
            ioff = nline*5
            lastline += '%16.8E'%(self.gradient[ioff+iterm])
        print (lastline)

        print ('Dipole Moment                              R   N=%12i'%3)
        lastline = ''
        for iterm in range(3):
            lastline += '%16.8E'%(self.dip[iterm])
        print (lastline)



    def _set_pyscf_qmmm(self):
        self.mol = gto.M(atom=self.qm_geometry,
                    basis=self.basis,
                    charge=self.charge,
                    spin=self.spin)
        self.mf = scf.RHF(self.mol).density_fit()
        self.mf_qmmm = qmmm.mm_charge(self.mf,
                                      self.mm_pos,
                                      self.mm_charges,
                                      unit='Agnstrom')


    def _set_pyscf_options(self):
        pyscf_info = self._input["pyscf"]
        if "basis" in pyscf_info:
            self.basis = pyscf_info["basis"]
        if "charge" in pyscf_info:
            self.charge = int(pyscf_info["charge"])
        if "spin" in pyscf_info:
            self.charge = int(pyscf_info["spin"])
        if "active electrons" in pyscf_info:
            self.nelec = int(pyscf_info["active electrons"])
        if "active orbitals" in pyscf_info:
            self.norb = int(pyscf_info["active orbitals"])
        if "method" in pyscf_info:
            self.method = (pyscf_info["method"]).lower()
        if "compute grad" in pyscf_info:
            if (pyscf_info["compute grad"]).lower() == 'false':
                self.doGrad = False
            else:
                self.doGrad = True


    def _set_file_name(self, gaussian_input_fname):
        self.basename = get_basename_of_inp_file(gaussian_input_fname)
        self.gaussian_input_file = gaussian_input_fname
        self.gaussian_fchk_file = self.basename + ".Fchk"
        self.gaussian_log_file = self.basename + ".log"

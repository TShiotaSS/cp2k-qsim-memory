from pyscf.mcscf import casci, mc1step
from vqe_qmmm.vqe.vqeci import VQECI


class VQECASCI(casci.CASCI):

    def __init__(self, mf, ncas, nelecas, ncore=None):
        casci.CASCI.__init__(self, mf, ncas, nelecas, ncore)
        self.fcisolver =  VQECI(mf.mol)



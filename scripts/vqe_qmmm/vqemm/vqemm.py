from vqe_qmmm.qmmm import electrostatic
from vqe_qmmm.vqe import vqemcscf


class VQECASCIMM(vqemcscf.VQECASCI):

    def __init__(self, mf, ncas, nelecas, ncore=None):
        vqemcscf.VQECASCI(mf, ncas, nelecas, ncore=None)
        #self._scf is mf



class VQECASCIMM(vqemcscf.VQECASCI):

    def __init__(self, mf, ncas, nelecas, ncore=None):
        vqemcscf.VQECASCI(mf, ncas, nelecas, ncore=None)




import numpy
import sys
from fcidump_helpers import read_fcidump, write_fcidump, calc_energy, \
    get_hf_rdms, calc_reference_energy

import fqe
from fqe import restricted_hamiltonian, sso_hamiltonian
from fqe.algorithm import davidson

numpy.set_printoptions(precision=8, linewidth=255, floatmode="fixed", suppress=True)


def fqe_perform_davidson(filename: str, output: str) -> None:
    """
    This code performs Davidson diagonalization and stores rdm
    """
    ecore, h, eri, norb, nelec, ms2, uhf = read_fcidump(filename)
    if uhf:
        h1 = numpy.zeros((norb * 2,) * 2, dtype=h.dtype)
        h1[:norb, :norb] = h[0]
        h1[norb:, norb:] = h[1]

        v2 = numpy.zeros((norb * 2,) * 4, dtype=eri.dtype)
        v2[:norb, :norb, :norb, :norb] = eri[0]
        v2[:norb, :norb, norb:, norb:] = eri[1]
        v2[norb:, norb:, :norb, :norb] = numpy.transpose(eri[1], (2, 3, 0, 1))
        v2[norb:, norb:, norb:, norb:] = eri[2]
    else:
        h1 = h[0]
        v2 = eri[0]
    v3 = -0.5 * numpy.transpose(v2, (0, 2, 1, 3))
    nalpha, nbeta = (nelec + ms2) // 2, (nelec - ms2) // 2

    if uhf:
        ham = sso_hamiltonian.SSOHamiltonian((h1, v3), e_0=ecore)
    else:
        ham = restricted_hamiltonian.RestrictedHamiltonian((h1, v3), e_0=ecore)

    # Fully occupied or empty
    if nalpha % norb == 0 and nbeta % norb == 0:
        v = [fqe.Wavefunction([[nelec, ms2, norb]])]
        v[0].set_wfn(strategy='random')
        v[0].normalize()
        w = [numpy.real(v[0].expectationValue(ham, brawfn=v[0]))]
    else:
        w, v = davidson.davidson_diagonalization(ham, nalpha, nbeta, 1)
    print(" * CASCI finished. The ground state energy: " + str(w[0]))
    wavefunction = v[0]

    if uhf:
        # Needed for correct rdm1 and rdm2 unfortunately...
        # Dummy wavefunction with all electrons being alpha and double amount of orbitals
        opdm, tpdm = wavefunction.sector((nelec, ms2)).get_openfermion_rdms()
        # Openfermion orders as a b a b a b a b ...
        # We need a a a a ... b b b b ...
        idx_swap = list(2 * ii for ii in range(norb)) + list(2 * ii + 1 for ii in range(norb))
        rdm1 = opdm[idx_swap, :][:, idx_swap]
        rdm2 = tpdm[idx_swap, :, :, :][:, idx_swap, :, :][:, :, idx_swap, :][:, :, :, idx_swap]
    else:
        rdm1 = wavefunction.rdm('i^ j')
        rdm2 = wavefunction.rdm('i^ j^ k l')
    rdm2 = -numpy.transpose(rdm2, (0, 2, 1, 3))
    if not numpy.allclose(rdm1.imag, 0, atol=1e-6):
        print(" WARNING - RDM1 has an element with non-zero imaginary part")
    if not numpy.allclose(rdm2.imag, 0, atol=1e-6):
        print(" WARNING - RDM2 has an element with non-zero imaginary part")
    rdm1, rdm2 = rdm1.real, rdm2.real

    assert numpy.isclose(nelec * (nelec - 1), numpy.einsum('iijj', rdm2))
    assert numpy.isclose(nelec, numpy.einsum('ii', rdm1))
    if numpy.allclose(rdm1, numpy.einsum('ijll', rdm2) / (nelec - 1)):
        print(" * returned RDM1 and RDM2 consistent.")
    else:
        raise Exception("RDM1 and RDM2 are inconsistent with each other (partial trace)")

    assert numpy.isclose(calc_energy(ecore, h1, v2, rdm1, rdm2), w[0])

    test = h1 @ rdm1 + numpy.tensordot(v2, rdm2, axes=((1, 2, 3), (1, 2, 3)))
    print(test)
    reference_rdms = get_hf_rdms(nelec, ms2, norb, uhf)
    reference_energy = calc_energy(ecore, h1, v2, *reference_rdms)
    assert numpy.isclose(reference_energy, calc_reference_energy(ecore, h, eri, nelec, ms2, uhf))

    # For later
    correlation_energy = w[0] - reference_energy
    rdm1 -= reference_rdms[0]
    rdm2 -= reference_rdms[1]
    assert numpy.isclose(0, numpy.einsum('iijj', rdm2))
    assert numpy.isclose(0, numpy.einsum('ii', rdm1))
    if numpy.allclose(rdm1, numpy.einsum('ijll', rdm2) / (nelec - 1)):
        print(" * returned diff_RDM1 and diff_RDM2 consistent.")
    else:
        raise Exception("diff_RDM1 and diff_RDM2 are inconsistent with each other (partial trace)")

    print(f" * Correlation energy: {correlation_energy}")

    write_fcidump(output, correlation_energy, rdm1, rdm2, norb, nelec, ms2, uhf)


if __name__ == '__main__':
    arg = sys.argv
    if len(arg) != 3:
        print("expecting exactly two arguments (input/output file names)")
        sys.exit(5)

    fqe_perform_davidson(arg[1], arg[2])

import re
import numpy
import sys
from typing import Tuple


def write_fcidump(filename: str, energy: float, rdm1: numpy.ndarray,
                  rdm2: numpy.ndarray, norb: int, nelec: int, ms2: int, uhf: bool=False) -> None:

    with open(filename, "w") as fp:
        # Writing header
        print(f"&FCI NORB= {norb},NELEC= {nelec},MS2= {ms2},", file=fp)
        print(','.join(('1',) * norb), file=fp)
        print('ISYM=0,', file=fp)
        print(' /', file=fp)

        for (ii, jj, kk, ll), val in numpy.ndenumerate(rdm2):
            if not numpy.isclose(val, 0, atol=1e-12):
                print(f"{val:22.15e} {ii+1:3d} {jj+1:3d} {kk+1:3d} {ll+1:3d}", file=fp)

        for (ii, jj), val in numpy.ndenumerate(rdm1):
            if not numpy.isclose(val, 0, atol=1e-12):
                print(f"{val:22.15e} {ii+1:3d} {jj+1:3d} {0:3d} {0:3d}", file=fp)
        print(f"{energy:22.15e} {0:3d} {0:3d} {0:3d} {0:3d}", file=fp)


def read_header(fp):
    header = ''
    for line in fp:
        header += line.strip()
        if line.strip() == '/':
            break

    m = re.search(r'&FCI\s+NORB=\s*([0-9]+),\s*NELEC=\s*([0-9]+),\s*MS2=\s*([0-9]+)', header)
    norb = int(m.group(1))
    nelec = int(m.group(2))
    ms2 = int(m.group(3))

    if m is None:
        raise Exception("fci dump is not properly formatted")

    if nelec % 2 != ms2 % 2:
        raise Exception(f"Inconcistent nelec={nelec} and ms2={ms2}")

    m = re.search(r'UHF=([^,]+),', header)
    uhf = (m is not None) and (bool(int(m.group(1))))

    return norb, nelec, ms2, uhf


def read_integrals(fp, norb, uhf=False, is_integral=True):
    nspins = 2 if uhf else 1
    ld_eri = nspins * (nspins + 1) // 2  # leading dimension eri

    def set_el(matrix, index, element):
        spins, index = list(zip(*[((x - 1) // norb, (x - 1) % norb) for x in index]))
        # Check spins are one on one same
        if set(s1 == s2 for s1, s2 in zip(spins[::2], spins[1::2])) != {True}:
            raise ValueError(f'invalid index combination {index} for norb={norb} and uhf={uhf}')
        if len(spins) == 4 and spins[0] > spins[2]:
            # Ignore wrongly ordered spins. Beta spin should be second electron in ERI
            return

        ld_index = spins[0] if len(spins) == 2 else spins[0] + spins[2]
        index = (ld_index,) + index
        if matrix[index] != 0.0 and not numpy.isclose(matrix[index], element):
            print(f"Warning {index} already set to {matrix[index]} instead of {element}")
        matrix[index] = element

    ecore = 0.0
    h = numpy.zeros((nspins, norb, norb), dtype=numpy.float64)
    eri = numpy.zeros((ld_eri, norb, norb, norb, norb), dtype=numpy.float64)

    for line in fp:
        m = re.search(r'([0-9eE\-\+\.]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)', line)
        if m is None:
            raise Exception(f"FCIDUMP is not properly formatted. Could not process \"{line}\"")

        el = float(m.group(1))
        ii, jj, kk, ll = tuple(int(m.group(x)) for x in range(2, 6))

        if ii != 0 and jj != 0 and kk != 0 and ll != 0:
            set_el(eri, (ii, jj, kk, ll), el)
            set_el(eri, (jj, ii, ll, kk), el)
            set_el(eri, (kk, ll, ii, jj), el)
            set_el(eri, (ll, kk, jj, ii), el)
            if is_integral:
                set_el(eri, (jj, ii, kk, ll), el)
                set_el(eri, (ii, jj, ll, kk), el)
                set_el(eri, (kk, ll, jj, ii), el)
                set_el(eri, (ll, kk, ii, jj), el)
        elif ii != 0 and jj != 0 and kk == 0 and ll == 0:
            set_el(h, (ii, jj), el)
            set_el(h, (jj, ii), el)
        elif ii == 0 and jj == 0 and kk == 0 and ll == 0:
            ecore = el
        else:
            raise Exception(f"FCIDUMP is not properly formatted. Could not process \"{line}\"")

    return ecore, h, eri


def read_fcidump(filename: str, reference_check: bool = True, is_integral=True) \
        -> Tuple[float, numpy.ndarray, numpy.ndarray, int, int, int, int]:

    with open(filename, "r") as fp:
        norb, nelec, ms2, uhf = read_header(fp)
        print(f"Read FCIDUMP. norb={norb}, nelec={nelec}, ms2={ms2}, uhf={uhf}")
        ecore, h, v = read_integrals(fp, norb, uhf=uhf, is_integral=is_integral)

    # for debug I calculate the reference Hartree-Fock energy
    if reference_check:
        eref = calc_reference_energy(ecore, h, v, nelec, ms2, uhf)
        print(f" * The reference ground state energy: {eref}")
    return ecore, h, v, norb, nelec, ms2, uhf


def calc_energy(ecore, h, eri, rdm1, rdm2):
    return ecore + rdm1.flatten() @ h.flatten() + 0.5 * rdm2.flatten() @ eri.flatten()


def get_hf_rdms(nelec, ms2, norb, uhf):
    nalpha = (nelec + ms2)//2
    nbeta = (nelec - ms2)//2
    dim = norb * 2 if uhf else norb
    rdm1 = numpy.zeros((dim,) * 2)
    rdm2 = numpy.zeros((dim,) * 4)
    electrons = [(0, ii) for ii in range(nalpha)] + \
        [(1, ii + norb * uhf) for ii in range(nbeta)]

    for spini, ii in electrons:
        rdm1[ii, ii] += 1.0
        for spinj, jj in electrons:
            rdm2[ii, ii, jj, jj] += 1.0
            if spini == spinj:
                rdm2[ii, jj, jj, ii] -= 1.0
    return rdm1, rdm2


def calc_reference_energy(ecore, h, v, nelec, ms2, uhf):
    alpha_occs = (nelec + ms2) // 2
    beta_occs = (nelec - ms2) // 2
    a_idx = 0
    b_idx = 1 if uhf else 0
    aa_idx = 0
    ab_idx = 1 if uhf else 0
    bb_idx = 2 if uhf else 0

    r_alpha = range(alpha_occs)
    r_beta = range(beta_occs)

    eref = ecore + h[a_idx, r_alpha, r_alpha].sum() + h[b_idx, r_beta, r_beta].sum()
    # Coulomb
    eref += 0.5 * v[aa_idx, r_alpha, r_alpha, :, :][:, r_alpha, r_alpha].sum()
    eref += 1.0 * v[ab_idx, r_alpha, r_alpha, :, :][:, r_beta, r_beta].sum()
    eref += 0.5 * v[bb_idx, r_beta, r_beta, :, :][:, r_beta, r_beta].sum()
    # Exchange
    eref -= 0.5 * v[aa_idx, r_alpha, :, :, r_alpha][:, r_alpha, r_alpha].sum()
    eref -= 0.5 * v[bb_idx, r_beta, :, :, r_beta][:, r_beta, r_beta].sum()
    return eref


if __name__ == '__main__':
    if len(sys.argv) == 1:
        read_fcidump('water-test-1_0.fcidump')
    else:
        read_fcidump(sys.argv[1])

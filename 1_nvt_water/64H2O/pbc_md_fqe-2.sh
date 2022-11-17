#!/bin/bash
#PBS -l nodes=8:ppn=96
#PBS -N UCCNVE_FQE_10ps

NPROCS=768
NTHREADSPERPROC=1

export OMP_NUM_THREADS=${NTHREADSPERPROC}

NCORES=`wc -l < ${PBS_NODEFILE}`
if [ ${NCORES} -ne $(( NPROCS * NTHREADSPERPROC )) ]; then echo Check NPROCS and NTHREADSPERPROC; exit 1; fi
NNODES=`uniq ${PBS_NODEFILE} | wc -l`
NCORESPERNODE=$(( NCORES / NNODES ))
if [ $(( NCORESPERNODE % NTHREADSPERPROC )) -ne 0 ]; then echo Check NPROCS and NTHREADSPERPROC; exit 1; fi

echo 'NCORES =' ${NCORES}
echo 'NNODES =' ${NNODES}
echo 'NCORESPERNODE =' ${NCORESPERNODE}
echo 'NPROCS =' ${NPROCS}
echo 'NTHREADSPERPROC =' ${NTHREADSPERPROC}

cd $PBS_O_WORKDIR
source ~/.bashrc

#path for cp2k-fqe
CP2K=/home/shiota/CP2K/cp2k_8.2_qs_v1_new_10c/exe/local/cp2k.psmp
FQE=/home/shiota/CP2K/cp2k_QSimulate/scripts

ulimit -s unlimited

for i in `seq 0 19`;do
mpirun --hostfile ${PBS_NODEFILE} -np 768 -x OMP_NUM_THREADS=1 ${CP2K} -i h2o-aimd-2_$i.inp -o h2o-aimd-2_$i.out
mv H2O-64-1.restart H2O-64-1_$i.restart
mv H2O-64-RESTART.wfn H2O-64-RESTART_$i.wfn 
mkdir fcidump_$i
mv *fcidump fcidump_$i
done



#!/bin/bash
#SBATCH --output=aims.out 
#SBATCH --nodes=1
#SBATCH --ntasks=20

mpirun /work1/spavlov/aims.241109.scalapack.mpi.x >&aims.out

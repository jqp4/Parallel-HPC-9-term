CC = mpicxx
CPP_FLAGS_LOCAL = -lc++ -std=c++17 -Wall -Werror -pthread
CPP_FLAGS = -std=c++11 -Wall -Werror -pthread


all: clear compile run_local

clear:
	rm -rf ./main
	clear

compile: main.cpp
	$(CC) main.cpp $(CPP_FLAGS) -o main

submit:
	bsub < ./lsf/Batcher_MPI_job_c1_s100000.lsf
	bsub < ./lsf/Batcher_MPI_job_c2_s100000.lsf
	bsub < ./lsf/Batcher_MPI_job_c4_s100000.lsf
	bsub < ./lsf/Batcher_MPI_job_c8_s100000.lsf
	bsub < ./lsf/Batcher_MPI_job_c16_s100000.lsf
	bsub < ./lsf/Batcher_MPI_job_c32_s100000.lsf
	bsub < ./lsf/Batcher_MPI_job_c40_s100000.lsf
	bsub < ./lsf/Batcher_MPI_job_c1_s1000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c2_s1000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c4_s1000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c8_s1000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c16_s1000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c32_s1000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c40_s1000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c1_s10000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c2_s10000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c4_s10000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c8_s10000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c16_s10000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c32_s10000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c40_s10000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c1_s100000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c2_s100000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c4_s100000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c8_s100000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c16_s100000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c32_s100000000.lsf
	bsub < ./lsf/Batcher_MPI_job_c40_s100000000.lsf
	bjobs

run_local:
	mpiexec -np 1 ./main 1000000
	mpiexec -np 2 ./main 1000000
	mpiexec -np 4 ./main 1000000
	mpiexec -np 8 ./main 1000000
	mpiexec -np 16 ./main 1000000

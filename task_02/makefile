CC = mpicxx
CPP_FLAGS_LOCAL = -lc++ -std=c++17 -Wall -Werror -pthread
CPP_FLAGS = -std=c++11 -Wall -Werror -pthread


all: clear compile run

clear:
	rm -rf ./main
	clear

compile:
	$(CC) main.cpp $(CPP_FLAGS) -o main

# submit:
# 	mpirun -n 36 --oversubscribe ./main

run:
	mpiexec -np 2 ./main 1000000

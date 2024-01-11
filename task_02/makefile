CC = mpicxx
# CPP_FLAGS = -lc++ -std=c++17 -Wall -Werror -pthread # local
CPP_FLAGS = -std=c++11 -Wall -Werror -pthread
N = 8

all: clear compile run

clear:
	rm -rf ./main
	clear

compile:
	$(CC) main.cpp $(CPP_FLAGS) -o main

submit:
	mpirun -n $(N) --oversubscribe ./main

run:
	mpiexec -np $(N) ./main

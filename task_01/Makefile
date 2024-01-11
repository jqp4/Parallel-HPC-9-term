# CPP_FLAGS = -std=c++17 -Wall -Werror -pthread
# как надо? - mpic++ bsort.cpp -o bsort -std=c++11

CC = clang
CPP_FLAGS = -lc++ -std=c++17 -Wall -Werror -pthread

all: clean compile run

clean:
	rm -rf main
	clear

compile: main.cpp
	$(CC) main.cpp $(CPP_FLAGS) -o main

run: main
	./main 5
	./main 8
 
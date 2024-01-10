#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "mpi.h"

#define assertm(exp, msg) assert(((void)msg, exp))

using ElementType = __uint16_t;

void printVector(const std::vector<ElementType>& array) {
    std::cout << "Vector(" << array.size() << "): ";
    for (ElementType element : array) {
        std::cout << (long long)(element) << " ";
    }

    std::cout << std::endl;
}

class BatcherSortingNetwork {
   public:
    struct Comparator {
        size_t a;
        size_t b;
    };

    class Tact {
       public:
        Tact() {}

        Tact(Comparator firstComparator) { addComparator(firstComparator); }

        bool containIndicesOf(Comparator comparator) {
            for (Comparator innerComparator : _comparators) {
                if (innerComparator.a == comparator.a || innerComparator.a == comparator.b ||
                    innerComparator.b == comparator.a || innerComparator.b == comparator.b) {
                    return true;
                }
            }

            return false;
        }

        void addComparator(Comparator comparator) {
            assertm(!containIndicesOf(comparator),
                    "The added comparator has indices contained in the sequence of comparators of "
                    "this network tact.");

            _comparators.push_back(comparator);
        }

        std::vector<Comparator> getComparators() const { return _comparators; }

       private:
        std::vector<Comparator> _comparators;
    };

    BatcherSortingNetwork(const size_t n) {
        _n = n;
        _B(0, 1, _n);
        _calculateNetworkTacts();
    }

    std::vector<Tact> getTacts() const { return _tactsVector; }

    void printComparatorsSummary() const {
        std::cout << (long long)(_n) << " 0 0" << std::endl;

        for (Comparator comparator : _comparatorsVector) {
            std::cout << (long long)(comparator.a) << " ";
            std::cout << (long long)(comparator.b) << std::endl;
        }

        std::cout << _comparatorsVector.size() << std::endl;
        std::cout << _tactsVector.size() << std::endl;

        std::cout << "\nNetwork tacts:\n";
        for (Tact tact : _tactsVector) {
            for (Comparator comparator : tact.getComparators()) {
                std::cout << (long long)(comparator.a) << "_";
                std::cout << (long long)(comparator.b) << ", ";
            }

            std::cout << std::endl;
        }
    }

    void sortVector(std::vector<ElementType>& array) const {
        assertm(array.size() == _n, "Incorrect vector size.");

        for (Comparator comparator : _comparatorsVector) {
            if (array[comparator.a] > array[comparator.b]) {
                std::swap(array[comparator.a], array[comparator.b]);
            }
        }
    }

    bool checkTheSortingIsCorrect(std::vector<ElementType>& array) const {
        assertm(array.size() == _n, "Incorrect vector size.");

        std::vector<ElementType> correctlySortedArray(array);
        std::sort(correctlySortedArray.begin(), correctlySortedArray.end());
        sortVector(array);

        // проверка
        // printVector(array);
        // printVector(correctlySortedArray);

        return array == correctlySortedArray;
    }

    bool checkThatSortingIsCorrectForAllCases() const {
        std::vector<ElementType> array(_n);

        for (unsigned long arrayNum = 0; arrayNum < 1 << _n; arrayNum++) {
            size_t i = _n - 1;
            unsigned long _num = arrayNum;
            while (_num > 0) {
                array[i] = ElementType(_num & 1);
                _num >>= 1;
                i--;
            }

            if (!checkTheSortingIsCorrect(array)) {
                return false;
            }
        }
        return true;
    }

   private:
    void _addComparator(const size_t a, const size_t b) {
        _comparatorsVector.push_back(Comparator{a, b});
    }

    // Рекурсивная процедура слияния двух групп линий (a, step, n) и (b, step, m)
    void _S(const size_t a, const size_t b, const size_t step, const size_t n, const size_t m) {
        if (n * m < 1) {
            return;
        } else if (n == 1 && m == 1) {
            _addComparator(a, b);
            return;
        }

        size_t i;
        size_t n1 = n - n / 2;  // количество нечетных строк в массиве a
        size_t m1 = m - m / 2;  // количество   четных строк в массиве b

        // объединить нечетные линии
        _S(a, b, 2 * step, n1, m1);

        // объединить четные линии
        _S(a + step, b + step, 2 * step, n - n1, m - m1);

        // далее добавить цепочку компараторов, начиная со второй линии

        // компараторы между линиями первого массива
        for (i = 1; i < n - 1; i += 2) {
            _addComparator(a + step * i, a + step * (i + 1));
        }

        if (n % 2 == 0) {
            // компаратор между массивами
            _addComparator(a + step * (n - 1), b);
            i = 1;
        } else {
            i = 0;
        }

        // компараторы между линиями второго массива
        for (; i < m - 1; i += 2) {
            _addComparator(b + step * i, b + step * (i + 1));
        }
    }

    // Процедура рекурсивного построения сети сортировки группы линий (first, step, n)
    void _B(const size_t first, const size_t step, const size_t n) {
        if (n < 2) {
            return;
        } else if (n == 2) {
            _addComparator(first, first + step);
            return;
        }

        // число элементов в первой половине массива
        size_t n1 = std::ceil(n / 2);

        // число элементов во второй половине массива
        size_t n2 = n - n1;

        // упорядочить первую половину массива
        _B(first, step, n1);

        // упорядочить вторую половину массива
        _B(first + step * n1, step, n2);

        // объединить упорядоченные части
        _S(first, first + step * n1, step, n1, n2);
    }

    // Функция оптимального рассчета тактов сети сортировки
    void _calculateNetworkTacts() {
        assertm(!_comparatorsVector.empty(), "Network comparators vector is empty.");
        _tactsVector.push_back(Tact());

        for (Comparator comparator : _comparatorsVector) {
            if (!_tactsVector.back().containIndicesOf(comparator)) {
                _tactsVector.back().addComparator(comparator);
            } else {
                _tactsVector.push_back(Tact(comparator));
            }
        }
    }

    size_t _n;
    std::vector<Comparator> _comparatorsVector;
    std::vector<Tact> _tactsVector;
};

void batcherSortingNetworkTest() {
    std::cout << "Batcher Sorting Network Test\n";
    std::cout << "[1] Testing all possible combinations"
              << " of arrays of length N = 1..24:\n";

    for (size_t n = 1; n <= 24; n++) {
        std::cout << " N = " << (long long)(n) << ": ";

        BatcherSortingNetwork sortingNetwork(n);
        bool result = sortingNetwork.checkThatSortingIsCorrectForAllCases();

        std::cout << (result ? "PASSED" : "FAILED") << std::endl;
        assertm(result, "Batcher Sorting Network test failed.");
    }

    const size_t nMax = 10000;
    std::cout << "[2] Testing random arrays of length N = " << nMax << ":\n";

    BatcherSortingNetwork sortingNetwork(nMax);
    for (unsigned i = 0; i < 10; i++) {
        std::cout << " try #" << i << ": ";

        std::vector<ElementType> bigArray(nMax);
        std::srand(unsigned(std::time(nullptr)));
        std::generate(bigArray.begin(), bigArray.end(), std::rand);
        bool result = sortingNetwork.checkTheSortingIsCorrect(bigArray);

        std::cout << (result ? "PASSED" : "FAILED") << std::endl;
        assertm(result, "Batcher Sorting Network test failed.");
    }
}

void generateArray(std::vector<double>* array, size_t n, size_t extendedN) {
    const double maxValue = 100;

    for (int i = 0; i < n; i++) {
        (*array)[i] = (double)rand() / RAND_MAX * maxValue;
    }

    for (int i = n; i < extendedN; i++) {
        (*array)[i] = __DBL_MAX__;
    }
}

void shareArray(size_t n, size_t m, std::vector<double>* myArray, int rank, int general_size) {
    if (rank == 0) {
        // размер расширенного массива
        size_t extendedN = m * general_size;
        // Основной массив, создается сразу расширенным
        std::vector<double> mainArray(extendedN);
        std::srand(unsigned(std::time(nullptr)));
        generateArray(&mainArray, n, extendedN);

        // BatcherSortingNetwork sortingNetwork(n);
        // sortingNetwork.printComparatorsSummary();

        for (int workerRank = 1; workerRank < general_size; workerRank++) {
            unsigned i0 = workerRank * m;
            MPI_Send(&mainArray[i0], m, MPI_DOUBLE, workerRank, 0, MPI_COMM_WORLD);
        }

        for (int i = 0; i < m; i++) {
            (*myArray)[i] = mainArray[i];
        }
    } else {
        // первым действием принимаем свою часть массива
        MPI_Recv(&(*myArray)[0], m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // std::cout << rank << ":\n";
        // for (int i = 0; i < m; i++) {
        //     std::cout << myArray[i] << " ";
        // }
        // std::cout << std::endl;
    }

    std::sort((*myArray).begin(), (*myArray).end());
}

int main(int argc, char* argv[]) {
    const size_t n = 30;

    // создаем группу процессов и область связи
    int rc = MPI_Init(&argc, &argv);

    if (rc) {
        std::cout << "Ошибка запуска " << rc << ", выполнение остановлено\n";
        MPI_Abort(MPI_COMM_WORLD, rc);
        return rc;
    }

    int rank, general_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &general_size);

    // размер куска массива для каждого процесса
    size_t m = (int)std::ceil((double)n / general_size);
    std::vector<double> myArray(m);
    // std::vector<double> otherArray(m);
    shareArray(n, m, &myArray, rank, general_size);

    std::cout << rank << ":\n";
    for (int i = 0; i < m; i++) {
        std::cout << myArray[i] << " ";
    }
    std::cout << std::endl;

    // выключаем MPI
    MPI_Finalize();
    return 0;
}

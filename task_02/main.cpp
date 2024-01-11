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

#define sleep(ms) usleep((useconds_t)ms * 1000)
#define assertm(exp, msg) assert(((void)msg, exp))

using ElementType = __uint16_t;

const bool DEBUG_MODE = true;
const useconds_t SLEEP_TIME_AFTER_BARRIER = 100;

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

        bool contain(size_t index) { return index == a || index == b; }
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

    std::vector<Comparator> getComparators() const { return _comparatorsVector; }

    void printComparatorsSummary() const {
        std::cout << (long long)(_n) << " 0 0" << std::endl;

        for (Comparator comparator : _comparatorsVector) {
            std::cout << (long long)(comparator.a) << " ";
            std::cout << (long long)(comparator.b) << std::endl;
        }

        std::cout << _comparatorsVector.size() << std::endl;
        std::cout << _tactsVector.size() << std::endl;
    }

    void printTactsSummary() const {
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
    void _addComparator(const size_t a, const size_t b) { _comparatorsVector.push_back(Comparator{a, b}); }

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

void shareArray(size_t n, size_t m, std::vector<double>* myArray, int rank, int processCount) {
    // Пусть процесс 0 временно побудет мастер-процессом,
    // который разошлет исходный массив всем процессам по частям
    if (rank == 0) {
        // Размер расширенного массива
        size_t extendedN = m * processCount;
        // Основной массив, создается сразу расширенным
        std::vector<double> mainArray(extendedN);
        // Заполняем первые N элементов случайными числами
        std::srand(unsigned(std::time(nullptr)));
        generateArray(&mainArray, n, extendedN);

        // Рассылаем кусочки основного массива одинаковой длины
        for (int workerRank = 1; workerRank < processCount; workerRank++) {
            unsigned i0 = workerRank * m;
            MPI_Send(&mainArray[i0], m, MPI_DOUBLE, workerRank, 0, MPI_COMM_WORLD);
        }

        // Для 0 процесса отдельно копируем первый кусочек
        for (int i = 0; i < m; i++) {
            (*myArray)[i] = mainArray[i];
        }
    } else {
        // Принимаем свою часть массива (для каждого процесса)
        MPI_Recv(&(*myArray)[0], m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // В каждом процессе сортируем полученные кусочки массива
    std::sort((*myArray).begin(), (*myArray).end());

    // Подождем, пока все отсортируются
    MPI_Barrier(MPI_COMM_WORLD);
}

void distributeValues(size_t m, std::vector<double>* array1, std::vector<double>* array2) {
    std::vector<double> tmpArray(m * 2);

    for (size_t i = 0; i < m; i++) {
        tmpArray[i] = (*array1)[i];
        tmpArray[m + i] = (*array2)[i];
    }

    std::sort(tmpArray.begin(), tmpArray.end());

    for (size_t i = 0; i < m; i++) {
        (*array1)[i] = tmpArray[i];
        (*array2)[i] = tmpArray[m + i];
    }
}

int main(int argc, char* argv[]) {
    // Задаем размер исходного массива
    const size_t n = 30;

    // Создаем группу процессов и область связи
    int rc = MPI_Init(&argc, &argv);
    if (rc) {
        std::cout << "Ошибка запуска " << rc << ", выполнение остановлено\n";
        MPI_Abort(MPI_COMM_WORLD, rc);
        return rc;
    }

    // Узнаем номер текущего процесса
    int rank, processCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    // std::cout << 1 << std::endl;
    // MPI_Barrier(MPI_COMM_WORLD);
    // sleep(SLEEP_TIME_AFTER_BARRIER);
    // std::cout << 2 << std::endl;
    // return 0;

    // Получаем такты сортировки
    BatcherSortingNetwork sortingNetwork(processCount);
    std::vector<BatcherSortingNetwork::Tact> networkTacts = sortingNetwork.getTacts();
    std::vector<BatcherSortingNetwork::Comparator> networkComparators = sortingNetwork.getComparators();

    // Размер куска массива для каждого процесса
    size_t m = (int)std::ceil((double)n / processCount);
    std::vector<double> myArray(m);
    std::vector<double> otherArray(m);
    shareArray(n, m, &myArray, rank, processCount);

    if (DEBUG_MODE) {
        std::cout << rank << ": ";
        for (int i = 0; i < m; i++) {
            std::cout << myArray[i] << " ";
        }
        std::cout << std::endl;

        if (rank == 0) {
            sleep(10);
            sortingNetwork.printTactsSummary();
            std::cout << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    sleep(SLEEP_TIME_AFTER_BARRIER);

    int i = 0;

    for (BatcherSortingNetwork::Tact networkTact : networkTacts) {
        for (BatcherSortingNetwork::Comparator comparator : networkTact.getComparators()) {
            // if (comparator.contain(rank)) {
            //     std::cout << "tact: " << i << "  rank: " << rank << std::endl;
            // }

            if (rank == comparator.a) {
                // отправляем массив этого процесса на сравнение другому процессу
                MPI_Send(&myArray[0], m, MPI_DOUBLE, comparator.b, 0, MPI_COMM_WORLD);
                MPI_Recv(&myArray[0], m, MPI_DOUBLE, comparator.b, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::cout << "tact " << i << ": " << (long long)(comparator.a) << " -> "
                          << (long long)(comparator.b) << " done" << std::endl;

            } else if (rank == comparator.b) {
                // получаем массив от другого процесса на сравнение
                // std::vector<double> otherArray(m);
                MPI_Recv(&otherArray[0], m, MPI_DOUBLE, comparator.a, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // otherArray должен забрать меньшие элементы, а myArray - бОльшие
                distributeValues(m, &otherArray, &myArray);

                // Отправляем полученные массив обратно
                MPI_Send(&otherArray[0], m, MPI_DOUBLE, comparator.a, 0, MPI_COMM_WORLD);
            }
        }

        // Подождем, пока все закончат сравниваться на этом такте
        MPI_Barrier(MPI_COMM_WORLD);
        sleep(SLEEP_TIME_AFTER_BARRIER);

        i++;
    }

    if (DEBUG_MODE) {
        std::cout << rank << ": ";
        for (int i = 0; i < m; i++) {
            std::cout << myArray[i] << " ";
        }
        std::cout << std::endl;
    }

    // Завершаем MPI
    MPI_Finalize();
    return 0;
}

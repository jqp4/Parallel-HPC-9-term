#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "mpi.h"

#define sleep(ms) usleep((useconds_t)ms * 1000)
#define assertm(exp, msg) assert(((void)msg, exp))

const int MASTER_RANK = 0;
const bool DEBUG_MODE = false;

using ElementType = double;
const ElementType ELEMENT_TYPE_MAX = __DBL_MAX__;
#define MPI_ELEMENT_TYPE OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_mpi_double)

class BatcherSortingNetwork {
   public:
    struct Comparator {
        size_t a;
        size_t b;

        bool containIndex(size_t index) const { return index == a || index == b; }

        bool containIndicesOf(Comparator otherComparator) const {
            return containIndex(otherComparator.a) || containIndex(otherComparator.b);
        }
    };

    class Tact {
       public:
        Tact() {}

        Tact(Comparator firstComparator) { addComparator(firstComparator); }

        void addComparator(Comparator comparator) {
            assertm(!containIndicesOf(comparator),
                    "The added comparator has indices contained in the sequence of comparators of "
                    "this network tact.");

            _comparators.push_back(comparator);
        }

        bool containIndicesOf(Comparator comparator) const {
            for (Comparator innerComparator : _comparators) {
                if (innerComparator.containIndicesOf(comparator)) {
                    return true;
                }
            }

            return false;
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
            bool newTactIsRequired = true;
            std::vector<Tact>::iterator tactIt = std::end(_tactsVector) - 1;

            // Шаг 1: ищем последний такт, в котором встречаются индексы из нового компаратора
            for (; tactIt != std::begin(_tactsVector); tactIt--) {
                if (tactIt->containIndicesOf(comparator)) {
                    break;
                }
            }

            // Шаг 2: ищем ближайшую позицию после этого такта, куда можно вставить новый компаратор
            for (; tactIt != std::end(_tactsVector); tactIt++) {
                if (!tactIt->containIndicesOf(comparator)) {
                    tactIt->addComparator(comparator);
                    newTactIsRequired = false;
                    break;
                }
            }

            // Если такой позиции не нашлось, создаем новый такт
            if (newTactIsRequired) {
                _tactsVector.push_back(Tact(comparator));
            }
        }
    }

    size_t _n;
    std::vector<Comparator> _comparatorsVector;
    std::vector<Tact> _tactsVector;
};

void printVector(const std::vector<ElementType>& array) {
    std::cout << "std::vector(" << array.size() << "): ";
    for (ElementType element : array) {
        std::cout << element << " ";
    }

    std::cout << std::endl;
}

void generateArray(std::vector<ElementType>* array, size_t n, size_t extendedN) {
    const ElementType maxValue = 100;
    std::srand(unsigned(std::time(nullptr)));

    for (int i = 0; i < n; i++) {
        (*array)[i] = (ElementType)rand() / RAND_MAX * maxValue;
    }

    for (int i = n; i < extendedN; i++) {
        (*array)[i] = ELEMENT_TYPE_MAX;
    }
}

// Функция распределения массива по процессам
void shareArray(size_t n, size_t m, std::vector<ElementType>* myArray, int rank, int processCount) {
    // Пусть процесс 0 временно побудет мастер-процессом,
    // который разошлет исходный массив всем процессам по частям
    if (rank == MASTER_RANK) {
        // Размер расширенного массива
        size_t extendedN = m * processCount;
        // Основной массив, создается сразу расширенным
        std::vector<ElementType> mainArray(extendedN);
        // Заполняем первые N элементов случайными числами
        generateArray(&mainArray, n, extendedN);

        // Рассылаем кусочки основного массива одинаковой длины
        for (int workerRank = 1; workerRank < processCount; workerRank++) {
            unsigned i0 = workerRank * m;
            MPI_Send(&mainArray[i0], m, MPI_ELEMENT_TYPE, workerRank, 0, MPI_COMM_WORLD);
        }

        // Для 0 процесса отдельно копируем первый кусочек
        for (int i = 0; i < m; i++) {
            (*myArray)[i] = mainArray[i];
        }
    } else {
        // Принимаем свою часть массива (для каждого процесса)
        MPI_Recv(&(*myArray)[0], m, MPI_ELEMENT_TYPE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// Функция распределения значений в 2 массивах длинны M. array1 получает меньшие элементы, а array2 - большие
double distributeValues(size_t m,
                        std::vector<ElementType>* array1,
                        std::vector<ElementType>* array2,
                        std::vector<ElementType>* tmpArray) {
    double timestampStart = MPI_Wtime();
    size_t i = 0, j = 0, k = 0;

    while (i < m && j < m) {
        if ((*array1)[i] < (*array2)[j]) {
            (*tmpArray)[k++] = (*array1)[i++];
        } else {
            (*tmpArray)[k++] = (*array2)[j++];
        }
    }

    while (i < m) {
        (*tmpArray)[k++] = (*array1)[i++];
    }

    while (j < m) {
        (*tmpArray)[k++] = (*array2)[j++];
    }

    // Копируем обратно
    for (i = 0; i < m; i++) {
        (*array1)[i] = (*tmpArray)[i];
        (*array2)[i] = (*tmpArray)[m + i];
    }

    return MPI_Wtime() - timestampStart;
}

bool checkArrayIsSortedCorrectly(std::vector<ElementType> array) {
    ElementType prevElement = array.front();

    for (ElementType element : array) {
        if (element < prevElement) {
            return false;
        }
    }

    return true;
}

long long getIntTime() {
    return (long long)(MPI_Wtime() * 100000000000);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "There are not enough command line arguments.\n";
        return -1;
    }

    // Задаем размер исходного массива
    int _n = std::stoi(argv[1]);
    if (_n < 1) {
        std::cout << "n must be greater than 0.\n";
        return -1;
    }

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

    size_t i = 0;
    size_t n = (size_t)_n;
    size_t m = (size_t)std::ceil((ElementType)n / processCount);
    double timestampStart, timestampFinish, timestampCompareStart, timestampCompareFinish;

    // если процесс всего один
    if (processCount == 1) {
        std::vector<ElementType> array(_n);
        generateArray(&array, _n, _n);

        timestampStart = MPI_Wtime();
        std::sort(array.begin(), array.end());
        timestampFinish = MPI_Wtime();

        std::cout << "procCount_size_time " << 1 << " " << _n << " " << timestampFinish - timestampStart
                  << std::endl;

        MPI_Finalize();
        return 0;
    }

    // Получаем такты сортировки
    BatcherSortingNetwork sortingNetwork(processCount);
    std::vector<BatcherSortingNetwork::Tact> networkTacts = sortingNetwork.getTacts();
    std::vector<BatcherSortingNetwork::Comparator> networkComparators = sortingNetwork.getComparators();

    // Для каждого процесса создаем кусочек массива размера m = n / processCount, и заполняем его
    std::vector<ElementType> myArray(m);
    shareArray(n, m, &myArray, rank, processCount);

    // Создаем вспомогательные массивы
    std::vector<ElementType> otherArray(m);
    std::vector<ElementType> tmpArrayForMerge(m * 2);
    std::vector<ElementType> finalArray;
    if (rank == MASTER_RANK) {
        finalArray.resize(m * processCount);
    }

    // Ввывод полезной информации (1/2), если включен DEBUG_MODE
    if (DEBUG_MODE) {
        if (m < 50) {
            if (rank == MASTER_RANK) {
                std::cout << "Parts of the array in processes before sorting:" << std::endl;
            }

            sleep(10 + rank * 2);
            std::cout << "r" << rank << ", ";
            printVector(myArray);
            sleep(10 + (processCount - rank - 1) * 2);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (rank == MASTER_RANK) {
            sortingNetwork.printTactsSummary();
            std::cout << "\nAlgorithm logs:" << std::endl;
        }
    }

    // Подождем, пока все процессы закончат приготовления
    MPI_Barrier(MPI_COMM_WORLD);

    // Записываем время старта работы основного алгоритма
    timestampStart = MPI_Wtime();

    // В каждом процессе сортируем полученные кусочки массива
    std::sort(myArray.begin(), myArray.end());

    // Основная часть программы: сеть обменной сортировки со слиянием Бэтчера
    for (BatcherSortingNetwork::Tact networkTact : networkTacts) {
        for (BatcherSortingNetwork::Comparator comparator : networkTact.getComparators()) {
            timestampCompareStart = MPI_Wtime();
            double mergeTime = 0;

            if (rank == comparator.a) {
                // Отправляем массив этого процесса на сравнение другому процессу
                MPI_Send(&myArray[0], m, MPI_ELEMENT_TYPE, comparator.b, 0, MPI_COMM_WORLD);
                // Получаем массив этого процесса с новыми значениями
                MPI_Recv(&myArray[0], m, MPI_ELEMENT_TYPE, comparator.b, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            } else if (rank == comparator.b) {
                // Получаем массив от другого процесса на сравнение
                MPI_Recv(&otherArray[0], m, MPI_ELEMENT_TYPE, comparator.a, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                // otherArray должен забрать меньшие элементы, а myArray - бОльшие
                mergeTime = distributeValues(m, &otherArray, &myArray, &tmpArrayForMerge);
                // Отправляем полученные массив обратно
                MPI_Send(&otherArray[0], m, MPI_ELEMENT_TYPE, comparator.a, 0, MPI_COMM_WORLD);
            }

            timestampCompareFinish = MPI_Wtime();
            if (DEBUG_MODE && (rank == comparator.a || rank == comparator.b)) {
                std::cout << getIntTime() << " Tact " << i << ", proc " << rank << ": r"
                          << (long long)(comparator.a) << " -> r" << (long long)(comparator.b) << ", done in "
                          << timestampCompareFinish - timestampCompareStart << " seconds"
                          << " (merge: " << mergeTime << " s)" << std::endl;
            }
        }

        // Подождем, пока все закончат сравниваться на этом такте
        // upd: не требуется, вспе и так работает
        // MPI_Barrier(MPI_COMM_WORLD);
        i++;
    }

    // Записываем время завершения основного основного алгоритма
    timestampFinish = MPI_Wtime();

    // Проверка на корректность сортировки, собираем массив обратно
    MPI_Gather(myArray.data(), m, MPI_ELEMENT_TYPE, finalArray.data(), m, MPI_ELEMENT_TYPE, MASTER_RANK,
               MPI_COMM_WORLD);

    // Ввывод полезной информации (2/2), если включен DEBUG_MODE
    if (DEBUG_MODE) {
        if (m < 50) {
            sleep(10);
            if (rank == MASTER_RANK) {
                std::cout << "\nParts of the array in processes after sorting:" << std::endl;
            }

            sleep(10 + rank * 2);
            std::cout << "r" << rank << ", ";
            printVector(myArray);
            sleep(10 + (processCount - rank - 1) * 2);
            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == MASTER_RANK) {
                std::cout << "\nFinal array, ";
                printVector(finalArray);
            }
        }
    }

    // Выводим полученные данные о работе алгоритма
    if (rank == MASTER_RANK) {
        std::cout << std::endl;

        if (checkArrayIsSortedCorrectly(finalArray)) {
            std::cout << "The array is sorted correctly." << std::endl;
        } else {
            std::cout << "The array is sorted incorrectly!" << std::endl;
        }

        std::cout << "Elapsed time is " << timestampFinish - timestampStart << " seconds." << std::endl;
        std::cout << std::endl;
    }

    // Еще раз выводим полученные данные в виде статистики
    // cat *.out | grep "procCount_size_time"
    if (rank == MASTER_RANK) {
        std::cout << "procCount_size_time " << processCount << " " << _n << " "
                  << timestampFinish - timestampStart << std::endl
                  << std::endl;
    }

    // Завершаем MPI
    MPI_Finalize();
    return 0;
}

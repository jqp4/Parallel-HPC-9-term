#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "There are not enough command line arguments.\n";
        return 0;
    }

    int n = std::stoi(argv[1]);
    if (n < 0) {
        std::cout << "n must be greater than 0.\n";
        return 0;
    }

    BatcherSortingNetwork sortingNetwork(n);
    sortingNetwork.printComparatorsSummary();
    // sortingNetwork.printTactsSummary();

    // Раскомментировать для тестирования сети сортировки
    // batcherSortingNetworkTest();

    return 0;
}

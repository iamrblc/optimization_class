#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

int main() {
    // Timer start
    auto start = std::chrono::high_resolution_clock::now();

    double total = 0.0;

    for (int i = 0; i < 100000000; ++i) {
        total += static_cast<double>(rand()) / RAND_MAX; // float between 0 and 1
    }

    // Timer end
    auto end = std::chrono::high_resolution_clock::now();

    // Output
    std::cout << "Total: " << total << std::endl;
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " s" << std::endl;

    return 0;
}

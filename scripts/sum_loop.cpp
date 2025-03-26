#include <iostream>
#include <chrono>
int main() {
    auto start = std::chrono::high_resolution_clock::now();
    long long total = 0;
    for (int i = 0; i < 1000000000; ++i) total += i;
    std::cout << total << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " s\n";
    return 0;
}

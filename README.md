# Introduction to Optimization

## Picking the right language

Let's see a problem that takes time, adding up the first billion numbers. 

In python you can do it by using a loop. 

```python
import time
N = 1_000_000_000

start = time.time()
total_loop = 0

for i in range(N):
    total_loop += i

print(f"Loop total: {total_loop}")
print(f"Loop time: {time.time() - start} s")
```

Note that it took some time even on a strong machine. Let's try the same with python's native sum() function.

```python
start = time.time()
total_sum = sum(range(N))
print(f"Sum total: {total_sum}")
print(f"Sum time: {time.time() - start} s")
```
This was much faster. When python runs this function, there is a low level language running in the background, similar to this C++ script:

```cpp
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
```

### Interpreted vs Compiled languages

A **compiled language** (like C++) translates the entire code into machine instructions **before** it runs. So when you execute it, it's already in the fastest form your computer understands.

An **interpreted language** (like Python) reads and executes code **line by line**, while the program is running. This makes it more flexible and easy to write, but slower because it's doing more work at runtime.

That’s why Python is often used for prototyping. It may be too slow for really big tasks, but it's easy to write and debug. Once the code is running as intended, you can "translate" it to a low level language. 

### JIT - Just In Time Compilation
There are languages that were developed specifically for scientific computing, such as Matlab or Julia. The latter is especially known for being "easy as Python, fast as C". 

Julia looks like an interpreted language (just like Python), but in reality it uses **just in time compliation (JIT)**. It doesn't compile the entire code before it runs (like C), but compiles parts of it as needed, on the go. 

While Juila or Matlab were designed to be able to do this there are also Python packages (such as numba) that enable JIT.

Let's see what Julia can do:

```julia
@time total = sum(0:999_999_999)
println(total)
```
(Note that Julia uses 1-indexing instead of 0-indexing, similar to R.)

You may notice that the runtime was extremely low. But how can it be even faster than a language close to machine code?

In scientific computing languages like Julia, many mathematical patterns are recognized and mapped to hardcoded optimizations. In this case, Julia detects that ranges like (n:m) are a special case for sum() and uses a fast, pre-optimized formula instead of looping.

Conceptually, this is similar to the trick the little Gauss allegedly used to sum numbers quickly:

$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$

Take home message: using your brain can speed up your code. 🙃

```python
start = time.time()
n = 999_999_999
total_sum = n * (1 + n) // 2
print(f"Sum total: {total_sum}")
print(f"Sum time: {time.time() - start} s")
```

See?


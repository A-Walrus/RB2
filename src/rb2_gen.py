import numpy as np
import timeit

N = 11520

m = np.fromfile('lut', dtype=np.uint16)

THREADS = 1000
ITERATIONS = 100


def foo():
    c = np.random.randint(0, N, (THREADS, ITERATIONS))
    s = np.zeros(THREADS, dtype=int)

    for i in range(ITERATIONS):
        s = m[c[:,i] + s*N]

n = 10
result = timeit.timeit(stmt='foo()', globals=globals(), number=n)

print(f"Execution time is {result / n / THREADS / ITERATIONS * 1e9} ns per Clifford")


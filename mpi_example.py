
import itertools
from mpi4py.futures import MPIPoolExecutor


def say_hi(stuff):

    print("Hello, world! You gave me: ", stuff)

    return(sum(stuff))


if __name__ == '__main__':

    print("Starting executor.")

    with MPIPoolExecutor() as executor:

        a = range(1)
        b = [16, 32]
        c = [32, 64]
        d = [1, 2]

        # Produce the Cartesian set of configurations.
        some_stuff = itertools.product(a, b, c, d)

        print("About to map")

        # output = executor.map(say_hi, [[] for _ in range(10)])
        output = executor.map(say_hi, some_stuff)

        for thing in output:
            print("The output is: ", thing)


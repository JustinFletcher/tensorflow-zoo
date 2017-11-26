
import itertools
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor


rank = comm.Get_rank()

print(comm)
print(rank)

def say_hi(stuff):

    print("Hello, world! You gave me: ", stuff)

    return(sum(stuff))


if __name__ == '__main__':

    print("Starting executor.")

    with MPIPoolExecutor() as executor:

        thread_counts = [16, 32]
        batch_sizes = [32, 64]
        reps = range(1)

        # Produce the Cartesian set of configurations.
        experimental_configurations = itertools.product(thread_counts,
                                                        batch_sizes,
                                                        reps)

        print("About to map")

        # output = executor.map(say_hi, [[] for _ in range(10)])
        output = executor.map(say_hi, experimental_configurations)

        for thing in output:
            print("The output is: ", thing)

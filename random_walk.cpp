#include <iostream>
#include <cstdlib> // For atoi, rand, srand
#include <ctime>   // For time
#include <mpi.h>

void walker_process();
void controller_process();

int domain_size;
int max_steps;
int world_rank;
int world_size;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 3)
    {
        if (world_rank == 0)
        {
            std::cerr << "Usage: mpirun -np <p> " << argv[0] << " <domain_size> <max_steps>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    domain_size = atoi(argv[1]);
    max_steps = atoi(argv[2]);

    if (world_rank == 0)
    {
        controller_process();
    }
    else
    {
        walker_process();
    }

    MPI_Finalize();
    return 0;
}

void walker_process()
{
    srand(time(NULL) + world_rank);

    int position = 0;
    int steps_taken = 0;

    for (steps_taken = 0; steps_taken < max_steps; ++steps_taken)
    {
        int step = (rand() % 2 == 0) ? -1 : +1;
        position += step;

        if (position < -domain_size || position > domain_size)
        {
            // Send steps taken (plus 1 because steps_taken is zero-indexed) to controller
            int result = steps_taken + 1;
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            return;
        }
    }

    // If max_steps reached without going out of bounds
    MPI_Send(&max_steps, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

void controller_process()
{
    int num_walkers = world_size - 1;
    int *results = new int[world_size]; // To hold results for each rank (only walkers)
    for (int i = 0; i < world_size; ++i) results[i] = -1; // Initialize

    int finished = 0;
    MPI_Status status;

    // Receive from each walker
    while (finished < num_walkers)
    {
        int steps;
        MPI_Recv(&steps, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        int rank = status.MPI_SOURCE;
        results[rank] = steps;
        finished++;
    }

    // Now print in rank order
    for (int rank = 1; rank < world_size; ++rank)
    {
        std::cout << "Rank " << rank << ": Walker finished in " << results[rank] << " steps." << std::endl;
    }

    std::cout << "Controller: All " << num_walkers << " walkers have finished." << std::endl;

    delete[] results;
}

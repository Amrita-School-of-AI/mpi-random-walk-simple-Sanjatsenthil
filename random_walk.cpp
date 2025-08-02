#include <iostream>
#include <cstdlib>   // For atoi, rand, srand
#include <ctime>     // For time
#include <mpi.h>

void walker_process();
void controller_process();

int domain_size;
int max_steps;
int world_rank;
int world_size;

int main(int argc, char **argv)
{
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes and the rank of this process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 3)
    {
        if (world_rank == 0)
        {
            std::cerr << "Usage: mpirun -np <p> " << argv[0] << " <domain_size> <max_steps>\n";
        }
        MPI_Finalize();
        return 1;
    }

    domain_size = std::atoi(argv[1]);
    max_steps = std::atoi(argv[2]);

    if (world_rank == 0)
    {
        // Rank 0 is the controller
        controller_process();
    }
    else
    {
        // All other ranks are walkers
        walker_process();
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}

void walker_process()
{
    // Seed the random number generator using rank for uniqueness
    srand(time(NULL) + world_rank);

    int position = 0;
    int steps = 0;

    while (steps < max_steps)
    {
        // Random step: -1 or +1
        int step = (rand() % 2 == 0) ? -1 : 1;
        position += step;
        steps++;

        // Check if walker is out of domain
        if (position < -domain_size || position > domain_size)
        {
            // Print finished message (required for autograder)
            std::cout << "Rank " << world_rank << ": Walker finished in " << steps << " steps." << std::endl;

            // Send step count to controller (rank 0)
            MPI_Send(&steps, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            break;
        }
    }

    // Also finish if max_steps is reached but still in domain
    if (position >= -domain_size && position <= domain_size && steps == max_steps)
    {
        std::cout << "Rank " << world_rank << ": Walker finished in " << steps << " steps." << std::endl;
        MPI_Send(&steps, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

void controller_process()
{
    int num_walkers = world_size - 1;
    int steps_taken;

    for (int i = 0; i < num_walkers; ++i)
    {
        MPI_Recv(&steps_taken, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Optionally: Print which walker completed
        // std::cout << "Controller: Received completion from a walker in " << steps_taken << " steps." << std::endl;
    }

    std::cout << "Controller: All " << num_walkers << " walkers have completed their walks." << std::endl;
}


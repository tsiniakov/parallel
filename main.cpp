#define _USE_MATH_DEFINES
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "mpi.h"

#define TARGET 1.0/24.0

using namespace std;

int main(int argc, char *argv[])
{
    int i, rank, size, iter = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double x, y, z;
    double
            partial_sum = 0.0,
            total_sum = 0.0,
            result = 0.0,
            eps = atof(argv[1]);

    int
            seed = 100000,
            block_size = 50.0 / (size * eps);

    block_size *= 16;
    drand48_data rand_buf;
    drand48_data rand_arr[16];

    int block_r = size * block_size / 16;
    for(i = 0; i < 16; ++i) {
        srand48_r(seed + i, &rand_arr[i]);
    }

    double start = MPI_Wtime();

    while(fabs(TARGET - result/iter) >= eps){
        partial_sum = 0.0;
        for(i = 0; i < block_size; ++i) {
            drand48_r(&rand_arr[(i/block_r)*size + rank], &x);
            drand48_r(&rand_arr[(i/block_r)*size + rank], &y);
            drand48_r(&rand_arr[(i/block_r)*size + rank], &z);

            partial_sum += (x - 1.0) * (x - 1.0) * (x - 1.0) * (y - 1.0) * (y - 1.0) * (z - 1.0)
                           / (size * block_size);
        }
        MPI_Allreduce(&partial_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        result += total_sum;
        ++iter;
    }
    double time = MPI_Wtime() - start;
    double total_time;

    MPI_Reduce(&time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Integral: " << result/(iter-1) << endl;
        cout << "Eps: " << abs(TARGET - result/(iter-1)) << endl;
        cout << "N points: " << size * block_size * (iter-1) << endl;
        cout << "Time: " << total_time << endl;
    }

    MPI_Finalize();

    return 0;
}
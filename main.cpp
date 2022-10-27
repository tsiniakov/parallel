#define _USE_MATH_DEFINES

#include <iostream>
#include <cstdlib>
#include <cmath>
#include "mpi.h"

#define TARGET_VALUE 1.0/24.0

using namespace std;

int main(int argc, char *argv[]) {

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double x, y, z, partialSum, totalSum, integral;
    double eps = atof(argv[1]);
    int N = (int) 10.0 / (size * eps);
    int iterations = 1;
    double start = MPI_Wtime(), end;
    srand(9*rank + 800);
    while (true) {
        partialSum = 0.0;
        for (int i = 0; i < N; ++i) {
            x = -1.0 + (double) rand() / RAND_MAX;
            y = -1.0 + (double) rand() / RAND_MAX;
            z = -1.0 + (double) rand() / RAND_MAX;
            partialSum += x * x * x * y * y * z / (N * size);
        }
        MPI_Allreduce(&partialSum, &totalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        integral += totalSum;
        if (fabs(TARGET_VALUE - integral / iterations) < eps) {
          break;
        }
        ++iterations;
    }
    end = MPI_Wtime();
    double totalTime, time = end - start;
    MPI_Reduce(&time, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << "size: " << size << endl;
        cout << "N points: " << size * N * (iterations - 1) << endl;
        cout << "Time: " << totalTime << endl;
        cout << "Amount of iterations: " << iterations - 1 << endl;
        cout << "Epsilon: " << eps << endl;
        cout << "Integral: " << integral / (iterations - 1) << endl;
        cout << "Eps: " << abs(TARGET_VALUE - integral / (iterations - 1)) << endl;
        cout << "True: " << TARGET_VALUE << endl;
        cout << endl;
    }
    MPI_Finalize();
    return 0;
}

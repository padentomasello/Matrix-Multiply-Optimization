#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cblas.h>
#include <omp.h>
#include <nmmintrin.h>

/* Your function must have the following signature: */

void sgemm(int m, int n, int d, float *A, float *C);

/* The reference code */
void sgemm_reference(int m, int n, float *A, float *C) {
#pragma omp parallel for
	for (int i = 0; i < n; i++)
		for (int k = 0; k < m; k++)
			for (int j = 0; j < n; j++)
				C[i + j * n] += A[i + k * (n)] * A[j * (n + 1) + k * (n)];
}

/* The benchmarking program */

int main(int argc, char **argv) {
	srand(time(NULL ));
	int counter = 0;
	float sum = 0;
//	int nStart = 32;
//	int nEnd = 100;
//	int mStart = 32;
//	int mEnd = 100;
		int nStart = 400;
		int nEnd = 1500;
		int mStart = 32;
		int mEnd = 100;
	int loopEnd = 1;
	if (argc == 3) {
		nStart = atoi(argv[1]);
		nEnd = nStart + 1;
		mStart = atoi(argv[2]);
		mEnd = mStart + 1;
		loopEnd = 10;
	}

	for (int loop = 0; loop < loopEnd; loop++)
		for (int n = nStart; n < nEnd; n = n + 1 +n/3) {
			/* Try different m */
			for (int m = mStart; m < mEnd; m = m + 1 + m/3) {
				/* Allocate and fill 2 random matrices A, C */
				//      printf("test5\n");
				float *A = (float*) malloc((n + m) * n * sizeof(float));
				//    printf("test6\n");
				float *C = (float*) malloc(n * n * sizeof(float));
				//    printf("test7\n");
				float *C_ref = (float*) malloc(n * n * sizeof(float));

				for (int i = 0; i < (n + m) * n; i++)
					A[i] = 2 * drand48() - 1;

				/* Ensure that error does not exceed the theoretical error bound */

				/* Calculate A*B from C using reference */
				memset( C_ref, 0, sizeof( float ) * n * n);
				sgemm_reference(m, n, A, C_ref);
				/* Set initial C to 0 and do matrix multiply of A*B */
				//     printf("test8\n");
				memset( C, 0, sizeof( float ) * n * n);
//      printf("test9\n");
				sgemm(m, n, m, A, C);
				//    printf("test10\n");

				/* Subtract the maximum allowed roundoff from each element of C */
				for (int i = 0; i < n * n; i++) {
					C[i] -= C_ref[i];
					//printf("C[i] = %f\n", C[i]);
				}

				/* After this test if any element in C is still positive something went wrong in square_sgemm */
				for (int i = 0; i < n * n; i++) {
					//  printf("C[i] = %f\n", C[i]);
					if (C[i] * C[i] > 0.0001) {
						printf(
								"FAILURE: error in matrix multiply exceeds an acceptable margin\n");
						printf(
								"Off by: %f, from the reference: %f, at n = %d, m = %d, i = %d\n",
								C[i], C_ref[i], n, m, i);
						return -1;
					}
				}

				/* measure Gflop/s rate; time a sufficiently long sequence of calls to eliminate noise */
				double Gflop_s, seconds = -1.0;
				for (int n_iterations = 1; seconds < 0.1; n_iterations *= 2) {
					/* warm-up */
					sgemm(m, n, m, A, C);

					/* measure time */
					struct timeval start, end;
					gettimeofday(&start, NULL );
					for (int i = 0; i < n_iterations; i++)
						sgemm(m, n, m, A, C);
					gettimeofday(&end, NULL );
					seconds = (end.tv_sec - start.tv_sec)
							+ 1.0e-6 * (end.tv_usec - start.tv_usec);

					/* compute Gflop/s rate */
					Gflop_s = 2e-9 * n_iterations * m * n * n / seconds;
				}

				printf("n = %d, m =  %d \t %g Gflop/s\n", n, m, Gflop_s);

				counter++;
				sum += Gflop_s;

				/* release memory */

				free(C_ref);
				//   printf("test2\n");
				free(C);
				//    printf("test3\n");
				free(A);
				//   printf("test4\n");
			}
		}
	printf("Average Gflops: %f\n", sum / counter);
	return 0;
}

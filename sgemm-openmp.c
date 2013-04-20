/*
 * sgemm-small.c
 *
 *  Created on: Apr 11, 2013
 *      Author: padentomasello
 */
#include <nmmintrin.h>
#include <stdio.h>
#include <string.h>
void sgemm(int m, int n, int d, float *A, float *C) {
	int ps = n;
	//	printf("test11, n: %d, m: %d\n", n, m);
	float At[m*n];
	//	{
	for(int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {

			*(At+i+j*n) = *(A+i*(n+1)+j*n);
		}
	}

	//  printf("test13 A2:%d, A:%d, C:%d, C:%d\n", A2, A, C, C);
	__m128 a1, a2, a3, a4, a5, c5, b, c1, c2, c3, c4;
	float c1sum;
	int k, j, i;
	//#pragma omp for private(a1, a2, a3, a4, a5, c5, b, c1, c2, c3, c4, c1sum, k, i ,j)
	{
		for( int k = 0; k+3 < m; k+=4 ) {
			for( int j = 0; j < n; j++ ) {
				b = _mm_load1_ps(At + j + k*n);
				for( int i = 0; i < n; i++ ) {
					c1 = _mm_loadu_ps(C+i+j*n);
					a1 = _mm_loadu_ps(A + i+ k*n);
					c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
					//C[i+j*n] += A[i+k*n] * At[j+k*n];
					_mm_storeu_ps(C+i+j*n, c1);
				}
			}
		}
	}


	//	}



	//   printf("test14\n");
	//		printf("test1\n");
	//}

}


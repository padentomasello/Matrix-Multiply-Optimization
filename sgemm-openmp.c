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
	int i, j, k;
	//	printf("test11, n: %d, m: %d\n", n, m);
	float At[n*m];
	float At2[n*m];
//#pragma omp parallel

//#pragma omp for private(i, j)
		for(i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				*(At2+i+j*m) = *(A+j+i*n);

			}

		}
		for(i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {

				*(At+j+i*m) = *(A+i*(n+1)+j*n);
			}
		}

		__m128 a1, a2, a3, a4, a5, c5, b, c1, c2, c3, c4;
		float c1sum;
		//#pragma omp for private(a1, a2, a3, a4, a5, c5, b, c1, c2, c3, c4, c1sum, k, i ,j)
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				c1 = _mm_setzero_ps();
				for (k=0; k+3 < m; k++) {
					b = _mm_loadu_ps(At+k+j*m);
					a1 = _mm_loadu_ps(At2+k+i*m);
					c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);

					//C[i+j*n] += At2[k+i*m] * At[k+j*m];
				}
				__m128 temp = _mm_add_ps(_mm_movehl_ps(c1, c1), c1);
				_mm_store_ss(C+i+j*n, _mm_add_ss(temp, _mm_shuffle_ps(temp, 1)));

			}
		}



		//   printf("test14\n");
		//		printf("test1\n");
		//}

	}



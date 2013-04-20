/*
 * sgemm-small.c
 *
 *  Created on: Apr 11, 2013
 *      Author: padentomasello
 */
#include <nmmintrin.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
void sgemm(int m, int n, int d, float *A, float *C) {
	//	printf("test11, n: %d, m: %d\n", n, m);
	float At[m*n];
	int i;
#pragma omp parallel
	{
#pragma omp for
		for(i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {

				*(At+i+j*n) = *(A+i*(n+1)+j*n);
			}
		}
		__m128 a1, a2, a3, a4, a5, b, c1, c2, c3, c4, c5;
		int k, j;
#pragma omp for private(a1, a2, a3, a4, a5, b, c1, c2, c3, c4, c5, k, i ,j)
		for(j = 0; j < n; j++ ) {
			if (j==0) {
				//printf("numofthreads: %d\n", omp_get_num_threads());
			}
			for(k = 0; k < m; k ++) {
				b = _mm_load1_ps(At + j + k*n);
				for(i = 0; i+19 < n; i += 20 ) {
					a1 = _mm_loadu_ps(A + i+ k*n);

					a2 = _mm_loadu_ps(A + (i+4)+ k*n);

					a3 = _mm_loadu_ps(A + (i+8)+ k*n);
					a4 = _mm_loadu_ps(A + (i+12)+ k*n);

					a5 = _mm_loadu_ps(A + (i+16)+ k*n);

					c1 = _mm_loadu_ps(C+i+(j*n));
					c2 = _mm_loadu_ps(C+(i+4)+(j*n));
					c3 = _mm_loadu_ps(C+(i+8)+(j*n));
					c5 = _mm_loadu_ps(C+(i+16)+(j*n));
					c4 = _mm_loadu_ps(C+(i+12)+(j*n));

					c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
					c2 = _mm_add_ps(_mm_mul_ps(a2, b), c2);
					c3 = _mm_add_ps(_mm_mul_ps(a3, b), c3);
					c4 = _mm_add_ps(_mm_mul_ps(a4, b), c4);
					c5 = _mm_add_ps(_mm_mul_ps(a5, b), c5);

					_mm_storeu_ps((C+(i+12)+(j*n)), c4);
					_mm_storeu_ps((C+(i+8)+(j*n)), c3);
					_mm_storeu_ps((C+(i+4)+(j*n)), c2);
					_mm_storeu_ps((C+i+(j*n)), c1);
					_mm_storeu_ps((C+(i+16)+(j*n)), c5);

				}
				//							for(; i+15 < n; i+=16 ) {
				//
				//								c1 = _mm_loadu_ps(C+i+(j*n));
				//								a1 = _mm_loadu_ps(A + i+ k*n);
				//								c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
				//								_mm_storeu_ps((C+i+(j*n)), c1);
				//								c1 = _mm_loadu_ps(C+(i+4)+(j*n));
				//								a1 = _mm_loadu_ps(A + (i+4)+ k*n);
				//								c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
				//								_mm_storeu_ps((C+(i+4)+(j*n)), c1);
				//								c1 = _mm_loadu_ps(C+(i+8)+(j*n));
				//								a1 = _mm_loadu_ps(A + (i+8)+ k*n);
				//								c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
				//								_mm_storeu_ps((C+(i+8)+(j*n)), c1);
				//								c1 = _mm_loadu_ps(C+(i+12)+(j*n));
				//								a1 = _mm_loadu_ps(A + (i+12)+ k*n);
				//								c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
				//								_mm_storeu_ps((C+(i+12)+(j*n)), c1);
				//
				//							}

				for (; i < n; i +=1) {
					{
						C[i+j*n] += A[i+k*n] * At[j+k*n];
					}
				}
			}

		}
	}


	//	}



	//   printf("test14\n");
	//		printf("test1\n");
	//}

}


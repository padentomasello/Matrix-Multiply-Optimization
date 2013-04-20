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
	//	printf("test11, n: %d, m: %d\n", n, m);
	//	float* At = (float *) malloc(n*m*sizeof(float));
	__m128 a1, a2, a3, a4, a5, c5, b, c1, c2, c3, c4;
	float c1sum;
	int i;
	int k;
	int j;
	float *atemp, *ctemp;
	float At[n*m];
	int btemp, cinter;
#pragma omp parallel
	{
#pragma omp for private(i, j)
		for(j = 0; j < n; j++) {
			btemp = j*m;
			cinter = j*(n+1);
			for (int k = 0; k < m; k++) {
				*(At+k+btemp) = *(A+cinter+k*n);
			}
		}



		//  printf("test13 A2:%d, A:%d, C:%d, C:%d\n", A2, A, C, C);

#pragma omp for private(a1, a2, a3, a4, a5, c5, b, c1, c2, c3, c4, c1sum, k, i ,j, btemp, ctemp, atemp, cinter)
		for (j = 0; j < n; j++) { //Goes through column of C
			i = 0;
			btemp = (j*m);
			cinter = (j*n);
			for (i = 0; i + 19 < n ; i += 20) { //Goes through 4 rows at a time of C and A.

				c1 = _mm_setzero_ps();
				c2 = c1;
				c3 = c1;
				c4 = c1;
				c5 = c1;
				ctemp = C + (cinter) + i;

				//		printf("test4. k:%d, i:%d, ps:%d\n", k, i, ps);
				for (k = 0; k < m; k += 1) { //Goes through Goes through width of m, data strip.
					atemp = A + (i) + (k * n);
					b = _mm_load1_ps((At+k+btemp));
					a1 = _mm_loadu_ps(atemp);
					a2 = _mm_loadu_ps(atemp+4);
					a3 = _mm_loadu_ps(atemp+8);
					a4 = _mm_loadu_ps(atemp+12);
					a5 = _mm_loadu_ps(atemp+16);
					c5 = _mm_add_ps(_mm_mul_ps(a5, b), c5);
					c4 = _mm_add_ps(_mm_mul_ps(a4, b), c4);
					c3 = _mm_add_ps(_mm_mul_ps(a3, b), c3);
					c2 = _mm_add_ps(_mm_mul_ps(a2, b), c2);
					c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
				}
				_mm_storeu_ps(ctemp, c1);
				_mm_storeu_ps(ctemp+4, c2);
				_mm_storeu_ps(ctemp+8, c3);
				_mm_storeu_ps(ctemp+12, c4);
				_mm_storeu_ps(ctemp+16, c5);
			}

			for (; i + 7 < n ; i += 8) { //Goes through 4 rows at a time of C and A.
				//	printf("test3\n");
				c1 = _mm_setzero_ps();
				c2 = c1;
				ctemp = C + cinter + i;
				for (int j = 0; j < m; j += 1) { //Goes through Goes through width of m, data strip.
					atemp = A + (i) + (j * n);
					b = _mm_load1_ps(At+j+btemp);
					a1 = _mm_loadu_ps(atemp);
					a2 = _mm_loadu_ps(atemp+4);
					c2 = _mm_add_ps(_mm_mul_ps(a2, b), c2);
					c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
				}
				_mm_storeu_ps(ctemp, c1);
				_mm_storeu_ps(ctemp + 4, c2);
			}
			//			}
			//	printf("test18, i: %d, k: %d, ps:%d\n", i, k, ps);
			while (i + 3 < n) {
				//printf("test2\n");
				c1 = _mm_setzero_ps();
				for (int j = 0; j < m; j += 1) { //Goes through Goes through width of m, data strip.
					//	printf("test4. k:%d, i:%d, ps:%d, j:%d, bi:%d\n", k, i, ps, j, (j * ps) + (i * (ps + 1)));
					b = _mm_load1_ps(At+j+btemp);
					//	printf("test5\n");
					a1 = _mm_loadu_ps((A + i + (j * n)));
					//	printf("test6\n");
					c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
					//	printf("test7\n");

				}
				//	printf("test16\n");
				_mm_storeu_ps((C + cinter + i), c1);
				i += 4;

			}
			while (i < n) {

				c1sum = 0;
				for (k = 0; k < m; k +=1) {
					float a1 = A[i + (k * n)];
					float b = At[k+btemp];
					c1sum += a1*b;
				}
				*(C + cinter + i) = c1sum;
				i += 1;
			}
		}
	}



	//   printf("test14\n");
	//		printf("test1\n");
	//}

}



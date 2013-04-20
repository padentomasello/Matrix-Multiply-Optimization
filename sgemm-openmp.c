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
	//	float* At = (float *) malloc(n*m*sizeof(float));
	__m128 a1, a2, a3, a4, a5, c5, b, c1, c2, c3, c4;
	float c1sum;
	int k, j, i;
	float *atemp, *ctemp;
	float At[n*m];
#pragma omp parallel
	{
#pragma omp for private(i, j)
		for(int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {

				*(At+j+i*m) = *(A+i*(n+1)+j*n);
			}
		}


		//  printf("test13 A2:%d, A:%d, C:%d, C:%d\n", A2, A, C, C);
		__m128 a1, a2, a3, a4, a5, c5, b, c1, c2, c3, c4;
		float c1sum;
		int k, j, i;
		float *atemp, *ctemp;
		int btemp;
#pragma omp for private(a1, a2, a3, a4, a5, c5, b, c1, c2, c3, c4, c1sum, k, i ,j)
		for (i = 0; i < n; i++) { //Goes through column of C
			k = 0;
			for (k = 0; k + 19 < ps ; k += 20) { //Goes through 4 rows at a time of C and A.

				c1 = _mm_setzero_ps();
				c2 = c1;
				c3 = c1;
				c4 = c1;
				c5 = c1;
				ctemp = C + (i * ps) + k;
				btemp = (i*ps+1);
				//		printf("test4. k:%d, i:%d, ps:%d\n", k, i, ps);
				for (j = 0; j < m; j += 1) { //Goes through Goes through width of m, data strip.
					atemp = A + (k) + (j * ps);
					b = _mm_load1_ps((At+j+i*m));
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

			for (; k + 7 < ps ; k += 8) { //Goes through 4 rows at a time of C and A.
				//	printf("test3\n");
				c1 = _mm_setzero_ps();
				c2 = c1;
				ctemp = C + (i * ps) + k;
				for (int j = 0; j < m; j += 1) { //Goes through Goes through width of m, data strip.
					atemp = A + (k) + (j * ps);
					b = _mm_load1_ps((A + (j * ps) + (i * (ps + 1))));
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
			while (k + 3 < ps) {
				//printf("test2\n");
				c1 = _mm_setzero_ps();
				for (int j = 0; j < m; j += 1) { //Goes through Goes through width of m, data strip.
					//	printf("test4. k:%d, i:%d, ps:%d, j:%d, bi:%d\n", k, i, ps, j, (j * ps) + (i * (ps + 1)));
					b = _mm_load1_ps((A + (j * ps) + (i * (ps + 1))));
					//	printf("test5\n");
					a1 = _mm_loadu_ps((A + k + (j * ps)));
					//	printf("test6\n");
					c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
					//	printf("test7\n");

				}
				//	printf("test16\n");
				_mm_storeu_ps((C + (i * ps) + k), c1);
				k += 4;

			}
			while (k < ps) {

				c1sum = 0;
				for (int j = 0; j < m; j +=1) {
					float a1 = A[k + (j * ps)];
					float b = A[(j * ps) + (i * (ps + 1))];
					c1sum += a1*b;
				}
				*(C + (i * ps) + k) = c1sum;
				k += 1;
			}
		}
	}



	//   printf("test14\n");
	//		printf("test1\n");
	//}

}



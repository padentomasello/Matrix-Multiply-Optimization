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
	float* A2;
	float* C2;
	int ps = n;
	//	printf("test11, n: %d, m: %d\n", n, m);
	float* At = (float *) malloc(n*m*sizeof(float));
	for(int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {

			*(At+i+j*n) = *(A+i*(n+1)+j*n);
		}

	}
	if ((n % 4) != 0) {
		ps = ((n / 4) + 1) * 4;
		//		printf("\n\n ps=%d\n\n", ps);
		A2 = (float*) malloc((n + m) * ps * sizeof(float));
		C2 = (float*) malloc(ps * ps * sizeof(float));
		//      printf("test12\n");
		for (int i = 0; i < (d + n); i++) {
			memcpy(A2+(ps*i), A+(i*n), 4*n);
			//		memset(A2+((i*ps) +n), 0, 4*(ps-n));
			//			int k;
			//			for (k = 0; k < n; k++) {
			//				*(A2+k+(i*ps)) = *(A+k+(i*n));
			//			}
			//			for (; k < ps; k++) {
			//				*(A2+k+(i*ps)) = 0;
			//			}
		}
	} else {
		A2 = A;
		C2 = C;
	}

	//  printf("test13 A2:%d, A:%d, C2:%d, C:%d\n", A2, A, C2, C);
	__m128 a1, a2, a3, a4, a5, c5, b, c1, c2, c3, c4;
	int k;
	for (int i = 0; i < n; i++) { //Goes through column of C
		for (k = 0; k + 4 < ps / 4; k += 5) { //Goes through 4 rows at a time of C and A.
			c1 = _mm_setzero_ps();
			c2 = c1;
			c3 = c1;
			c4 = c1;
			c5 = c1;
			for (int j = 0; j < m; j += 1) { //Goes through Goes through width of m, data strip.
				b = _mm_load1_ps((A2 + (j * ps) + (i * (ps + 1))));
				a1 = _mm_load_ps((A2 + (k * 4) + (j * ps)));
				a2 = _mm_load_ps((A2 + ((k + 1) * 4) + (j * ps)));
				a3 = _mm_load_ps((A2 + ((k + 2) * 4) + (j * ps)));
				a4 = _mm_load_ps((A2 + ((k + 3) * 4) + (j * ps)));
				a5 = _mm_load_ps((A2 + ((k + 4) * 4) + (j * ps)));
				c5 = _mm_add_ps(_mm_mul_ps(a5, b), c5);
				c4 = _mm_add_ps(_mm_mul_ps(a4, b), c4);
				c3 = _mm_add_ps(_mm_mul_ps(a3, b), c3);
				c2 = _mm_add_ps(_mm_mul_ps(a2, b), c2);
				c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
			}
			_mm_store_ps((C2 + (i * ps) + (4 * k)), c1);
			_mm_store_ps((C2 + (i * ps) + (4 * (k + 1))), c2);
			_mm_store_ps((C2 + (i * ps) + (4 * (k + 2))), c3);
			_mm_store_ps((C2 + (i * ps) + (4 * (k + 3))), c4);
			_mm_store_ps((C2 + (i * ps) + (4 * (k + 4))), c5);
		}
		//		printf("test17, i: %d, k: %d, ps:%d\n", i, k, ps);
		for (; k + 1 < ps / 4; k += 2) { //Goes through 4 rows at a time of C and A.
			c1 = _mm_setzero_ps();
			c2 = c1;
			for (int j = 0; j < m; j += 1) { //Goes through Goes through width of m, data strip.
				b = _mm_load1_ps((A2 + (j * ps) + (i * (ps + 1))));
				a1 = _mm_load_ps((A2 + (k * 4) + (j * ps)));
				a2 = _mm_load_ps((A2 + ((k + 1) * 4) + (j * ps)));
				c2 = _mm_add_ps(_mm_mul_ps(a2, b), c2);
				c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
			}
			_mm_store_ps((C2 + (i * ps) + (4 * k)), c1);
			_mm_store_ps((C2 + (i * ps) + (4 * (k + 1))), c2);
		}
		//	printf("test18, i: %d, k: %d, ps:%d\n", i, k, ps);
		if (k < ps / 4) {
			c1 = _mm_setzero_ps();
			for (int j = 0; j < m; j += 1) { //Goes through Goes through width of m, data strip.

				b = _mm_load1_ps((A2 + (j * ps) + (i * (ps + 1))));
				a1 = _mm_load_ps((A2 + (k * 4) + (j * ps)));
				c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);

			}
			//	printf("test16\n");
			_mm_store_ps((C2 + (i * ps) + (4 * k)), c1);

		}
	}
	//   printf("test14\n");
	if ((n % 4) != 0) {
		for (int i = 0; i < n; i++) {
			memcpy(C+(i*n), C2+(ps*i), 4*n);
		}
		free(A2);
		free(C2);
		//		printf("test1\n");
	}
}

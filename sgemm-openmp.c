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
	static int blocksize = 25;
	static int TWENTY = 40;
#pragma omp parallel num_threads(8)
	{
		__m128 c5, b, c1, c2, c3, c4, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15;
		float bsmall[blocksize*m];
		float small[40*m];
		int i, k, j, l,ln, temp, temp2, cinter;
		float *atemp, *ctemp, *bpoint;

		float c1sum;




#pragma omp for private(c13, c14, c15, c5, b, c1, c2, c3, c4, c6, c7, c8, c9, c10, c11, c12, c1sum, k, i ,j, ctemp, atemp, cinter, l, ln, temp, temp2, small, bpoint, bsmall) schedule(dynamic)
		for (j = 0; j < n; j+= blocksize) { //Goes through column of C
			cinter = (j*n);
			//		printf("test2 j: %d, n:$d,\n,", j, n);
			for(i = 0; i < blocksize && i+j < n; i++) {
				temp = i*m;
				temp2 = (j+i)*(n+1);
				for (k = 0; k < m; k++) {
					*(bsmall+k+temp) = *(A+temp2+k*n);
				}
			}
			//			//	printf("test3\n");
			//			for (i = 0; i + 59 < n ; i += 60) { //Goes through 4 rows at a time of C and A.
			//				ctemp = C + (cinter) + i;
			//				for (ln = 0; ln < m; ln++){
			//					temp = ln*60;
			//					temp2 = i+(n*ln);
			//					for (l = 0; l < 60; l++) {
			//						*(Asmall+l+(temp)) = *(A+l+temp2);
			//					}
			//				}
			//				for (l = 0; (l < blocksize) && (l+j < n); l++) {
			//					ln = l*n;
			//					c1 = _mm_setzero_ps();
			//					c2 = c1;
			//					c3 = c1;
			//					c4 = c1;
			//					c5 = c1;
			//					c6 = c1;
			//					c7 = c1;
			//					c8 = c1;
			//					c9 = c1;
			//					c10 = c1;
			//					c11 = c1;
			//					c12 = c1;
			//					c13 = c1;
			//					c14 = c1;
			//					c15 = c1;
			//					bpoint = Bsmall+m*l;
			//					for (k = 0; k < m; k += 1) { //Goes through Goes through width of m, data strip.
			//						atemp = Asmall+60*k;
			//						b = _mm_load1_ps(bpoint+k);
			//						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);
			//
			//						c2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+4), b), c2);
			//
			//						c3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+8), b), c3);
			//
			//						c4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+12), b), c4);
			//
			//						c5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+16), b), c5);
			//						c6 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+20), b), c6);
			//						c7 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+24), b), c7);
			//						c8 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+28), b), c8);
			//						c9 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+32), b), c9);
			//						c10 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+36), b), c10);
			//						c11 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+40), b), c11);
			//						c12 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+44), b), c12);
			//						c13 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+48), b), c13);
			//						c14 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+52), b), c14);
			//						c15 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+56), b), c15);
			//
			//
			//
			//
			//					}
			//					_mm_storeu_ps(ctemp+ln, c1);
			//					_mm_storeu_ps(ctemp+4+ln, c2);
			//					_mm_storeu_ps(ctemp+8+ln, c3);
			//					_mm_storeu_ps(ctemp+12+ln, c4);
			//					_mm_storeu_ps(ctemp+16+ln, c5);
			//					_mm_storeu_ps(ctemp+20+ln, c6);
			//					_mm_storeu_ps(ctemp+24+ln, c7);
			//					_mm_storeu_ps(ctemp+28+ln, c8);
			//					_mm_storeu_ps(ctemp+32+ln, c9);
			//					_mm_storeu_ps(ctemp+36+ln, c10);
			//					_mm_storeu_ps(ctemp+40+ln, c11);
			//					_mm_storeu_ps(ctemp+44+ln, c12);
			//					_mm_storeu_ps(ctemp+48+ln, c13);
			//					_mm_storeu_ps(ctemp+52+ln, c14);
			//					_mm_storeu_ps(ctemp+56+ln, c15);
			//				}
			//			}
			for (i = 0; i + 39 < n ; i += 40) { //Goes through 4 rows at a time of C and A.
				ctemp = C + (cinter) + i;
				for (ln = 0; ln < m; ln++){
					temp = ln*40;
					temp2 = i+(n*ln);
					for (l = 0; l < 40; l++) {
						small[l+(temp)] = *(A+l+temp2);
					}
				}
				for (l = 0; (l < blocksize) && (l+j < n); l++) {
					ln = l*n;
					c1 = _mm_setzero_ps();
					c2 = c1;
					c3 = c1;
					c4 = c1;
					c5 = c1;
					c6 = c1;
					c7 = c1;
					c8 = c1;
					c9 = c1;
					c10 = c1;

					bpoint = (bsmall+m*l);
					for (k = 0; k < m; k += 1) { //Goes through Goes through width of m, data strip.
						atemp = small+40*k;
						b = _mm_load1_ps(bpoint+k);
						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);

						c2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+4), b), c2);

						c3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+8), b), c3);

						c4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+12), b), c4);

						c5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+16), b), c5);
						c6 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+20), b), c6);
						c7 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+24), b), c7);
						c8 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+28), b), c8);
						c9 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+32), b), c9);
						c10 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+36), b), c10);
					}
					_mm_storeu_ps(ctemp+ln, c1);
					_mm_storeu_ps(ctemp+4+ln, c2);
					_mm_storeu_ps(ctemp+8+ln, c3);
					_mm_storeu_ps(ctemp+12+ln, c4);
					_mm_storeu_ps(ctemp+16+ln, c5);
					_mm_storeu_ps(ctemp+20+ln, c6);
					_mm_storeu_ps(ctemp+24+ln, c7);
					_mm_storeu_ps(ctemp+28+ln, c8);
					_mm_storeu_ps(ctemp+32+ln, c9);
					_mm_storeu_ps(ctemp+36+ln, c10);

				}
			}
			for (; i + 19 < n ; i += 20) { //Goes through 4 rows at a time of C and A.
				ctemp = C + (cinter) + i;
				for (ln = 0; ln < m; ln++){
					temp = ln*20;
					temp2 = i+(n*ln);
					for (l = 0; l < 20; l++) {
						*(small+l+(temp)) = *(A+l+temp2);
					}
				}
				for (l = 0; (l < blocksize) && (l+j < n); l++) {
					ln = l*n;
					c1 = _mm_setzero_ps();
					c2 = c1;
					c3 = c1;
					c4 = c1;
					c5 = c1;

					bpoint = bsmall+m*l;
					for (k = 0; k < m; k += 1) { //Goes through Goes through width of m, data strip.
						atemp = small+20*k;
						b = _mm_load1_ps(bpoint+k);
						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);

						c2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+4), b), c2);

						c3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+8), b), c3);

						c4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+12), b), c4);

						c5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+16), b), c5);

					}
					_mm_storeu_ps(ctemp+ln, c1);
					_mm_storeu_ps(ctemp+4+ln, c2);
					_mm_storeu_ps(ctemp+8+ln, c3);
					_mm_storeu_ps(ctemp+12+ln, c4);
					_mm_storeu_ps(ctemp+16+ln, c5);


				}
			}

			//			for (; i + 7 < n ; i += 8) { //Goes through 4 rows at a time of C and A.
			//				//	printf("test3\n");
			//				c1 = _mm_setzero_ps();
			//				c2 = c1;
			//				ctemp = C + cinter + i;
			//				for (int j = 0; j < m; j += 1) { //Goes through Goes through width of m, data strip.
			//					atemp = A + (i) + (j * n);
			//					b = _mm_load1_ps(At+j+btemp);
			//					a1 = _mm_loadu_ps(atemp);
			//					a2 = _mm_loadu_ps(atemp+4);
			//					c2 = _mm_add_ps(_mm_mul_ps(a2, b), c2);
			//					c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
			//				}
			//				_mm_storeu_ps(ctemp, c1);
			//				_mm_storeu_ps(ctemp + 4, c2);
			//			}
			//			//			}
			//			//	printf("test18, i: %d, k: %d, ps:%d\n", i, k, ps);
			while (i + 3 < n) {
				ctemp = C + (cinter) + i;
				for (ln = 0; ln < m; ln++){
					temp = ln*4;
					temp2 = i+(n*ln);
					for (l = 0; l < 4; l++) {
						*(small+l+(temp)) = *(A+l+temp2);
					}
				}
				for (l = 0; (l < blocksize) && (l+j < n); l++) {
					c1 = _mm_setzero_ps();
					bpoint = bsmall+m*l;
					for (int k = 0; k < m; k += 1) { //Goes through Goes through width of m, data strip.
						//	printf("test4. k:%d, i:%d, ps:%d, j:%d, bi:%d\n", k, i, ps, j, (j * ps) + (i * (ps + 1)));
						atemp = small+4*k;
						b = _mm_load1_ps(bpoint +k);
						//	printf("test5\n");
						//	printf("test6\n");
						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);
						//	printf("test7\n");

					}
					//	printf("test16\n");
					_mm_storeu_ps((C + cinter + i + l*n), c1);

				}
				i += 4;
			}
			//	printf("test3\n");
			while (i < n) {
				ctemp = C + (cinter) + i;
				for (ln = 0; ln < m; ln++){
					*(small+(ln)) = *(A+i+(n*ln));

				}
				for (l = 0; (l < blocksize) && (l+j < n); l++) {
					c1sum = 0;
					for (k = 0; k < m; k +=1) {
						c1sum += *(small+k)*(*(bsmall+k+m*l));
					}
					*(C + cinter + i+l*n) = c1sum;

				}
				i += 1;
			}
			//	printf("test5\n");
		}

	}



	//   printf("test14\n");
	//		printf("test1\n");
	//}

}



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
	__m128 c5, b, c1, c2, c3, c4, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15;
	int i, k, j, l,ln, temp, temp2;
	float *atemp, *ctemp, *bpoint;
	float *Bsmall;
	static int blocksize = 25;
	float *Asmall;
	static int TWENTY = 40;
	float c1sum;
#pragma omp parallel num_threads(8)
	{
		//#pragma omp for private(i, j)
		//		for(j = 0; j < n; j++) {
		//			btemp = j*m;
		//			cinter = j*(n+1);
		//			for (int k = 0; k < m; k++) {
		//				*(At+k+btemp) = *(A+cinter+k*n);
		//			}
		//		}


		//	printf("test1\n");
		//  printf("test13 A2:%d, A:%d, C:%d, C:%d\n", A2, A, C, C);
		float bsmall[blocksize*m];
		float small[40*m];
		//	printf("test2\n");
#pragma omp for private(c13, c14, c15, c5, b, c1, c2, c3, c4, c6, c7, c8, c9, c10, c11, c12, c1sum, k, i ,j, ctemp, atemp, l, ln, temp, temp2, Asmall, small, Bsmall, bpoint, bsmall) schedule(static)
		for (j = 0; j < n; j+= blocksize) { //Goes through column of C
			//		printf("test2 j: %d, n:$d,\n,", j, n);
			int block = (blocksize < (n-j) ? blocksize : n-j);
			Bsmall = bsmall;
			for(i = (blocksize < (n-j) ? blocksize : n-j); --i >= 0;) {
				temp = i*m;
				temp2 = (j+i)*(n+1);
				for (k = m; --k >=0;) {
					*(Bsmall+k+temp) = *(A+temp2+k*n);
				}
			}
			Asmall = small;
			for (i = 0; i + 39 < n ; i += 40) { //Goes through 4 rows at a time of C and A.
				ctemp = C + (j*n) + i;

				for (ln = m; --ln >= 0;){
					temp = ln*40;
					temp2 = i+(n*ln);
					for (k = 40; --k >= 0;) {
						*(Asmall+k+(temp)) = *(A+k+temp2);
					}
				}
				for (l = block; --l >=0;) {
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

					bpoint = Bsmall+m*l;
					for (k = m; --k >=0;) { //Goes through Goes through width of m, data strip.
						atemp = Asmall+40*k;
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
			for (; i + 15 < n ; i += 16) { //Goes through 4 rows at a time of C and A.
				ctemp = C + (j*n) + i;
				for (ln = m; --ln>=0;){
					temp = (ln << 4);
					temp2 = i+(n*ln);
					for (l = 16; --l>=0;) {
						*(Asmall+l+(temp)) = *(A+l+temp2);
					}
				}

				for (l = block; --l>=0;) {
					ln = l*n;
					c1 = _mm_setzero_ps();
					c2 = c1;
					c3 = c1;
					c4 = c1;
					c5 = c1;

					bpoint = Bsmall+m*l;
					for (k = m; --k >= 0;) { //Goes through Goes through width of m, data strip.
						atemp = Asmall+(k << 4);
						b = _mm_load1_ps(bpoint+k);
						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);

						c2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+4), b), c2);

						c3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+8), b), c3);

						c4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+12), b), c4);

						//									c5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+16), b), c5);

					}
					_mm_storeu_ps(ctemp+ln, c1);
					_mm_storeu_ps(ctemp+4+ln, c2);
					_mm_storeu_ps(ctemp+8+ln, c3);
					_mm_storeu_ps(ctemp+12+ln, c4);
					//								_mm_storeu_ps(ctemp+16+ln, c5);


				}
			}
			while (i + 3 < n) {
				ctemp = C + (j*n) + i;
				for (ln = m; --ln >= 0;){
					temp = (ln << 2);
					temp2 = i+(n*ln);
					for (l = 0; l < 4; l++) {
						*(Asmall+l+(temp)) = *(A+l+temp2);
					}
				}
				temp = n-j;
				for (l = block; --l>=0;) {
					c1 = _mm_setzero_ps();
					bpoint = Bsmall+m*l;

					for (int k = m; --k >= 0;) { //Goes through Goes through width of m, data strip.
						//	printf("test4. k:%d, i:%d, ps:%d, j:%d, bi:%d\n", k, i, ps, j, (j * ps) + (i * (ps + 1)));
						atemp = Asmall+(k << 2);
						b = _mm_load1_ps(bpoint +k);
						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);
					}
					_mm_storeu_ps((ctemp + l*n), c1);

				}
				i += 4;
			}
			while (i < n) {
				for (ln = m; --ln >=0;){
					*(Asmall+(ln)) = *(A+i+(n*ln));
				}

				for (l = block; --l >=0;) {
					c1sum = 0;
					temp = m*l;
					for (k = m; --k>=0;) {
						c1sum += small[k]*bsmall[k+temp];
					}
					*(C + j*n + i+l*n) = c1sum;
				}
				++i;
			}
		}

	}



	//   printf("test14\n");
	//		printf("test1\n");
	//}

}



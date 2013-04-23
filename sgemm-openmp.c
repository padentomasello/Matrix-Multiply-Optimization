/*
 * sgemm-small.c
 *
 *  Created on: Apr 11, 2013
 *      Author: padentomasello, forrestlaine
 */
#include <nmmintrin.h>
#include <stdio.h>
#include <string.h>
void sgemm(int m, int n, int d, float *A, float *C) {
	//	printf("test11, n: %d, m: %d\n", n, m);
	//	float* At = (float *) malloc(n*m*sizeof(float));
	float c1sum;
	__m128 b, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10;
	int i, k, j, l,ln, temp2, block;
	float *atemp, *ctemp, *bpoint;
	static int blocksize = 25;

#pragma omp parallel num_threads(8)
	{

		float bsmall[blocksize*m];
		float Asmall[40*m];
		//	printf("test2\n");
#pragma omp for private(b, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c1sum, k, i ,j, ctemp, block, atemp, l, ln, temp2, bpoint) schedule(dynamic)
		for (j = 0; j < n; j+= blocksize) { //Goes through column of C
			block = ((blocksize < (n - j)) ? blocksize : n-j);
			for(i = block; --i>=0;) {
				ln = i*m;
				temp2 = (j+i)*(n+1);
				for (k = 0; k < m; k++) {
					bsmall[k+ln] = *(A+temp2+k*n);
				}
			}

			for (i = 0; i + 39 < n ; i += 40) { //Goes through 40 rows at a time of C and A.
				ctemp = C + j*n + i;
				for (ln = 0; ln < m; ln++){
					k = ln*40;
					temp2 = i+(n*ln);
					for (l = 0; l < 40; l++) {
						Asmall[l+k] = *(A+l+temp2);
					}
				}
				for (l = 0; l < block; ++l) {
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

					bpoint = (&bsmall[m*l]);
					for (k = 0; k < m; k += 1) { //Goes through Goes through width of m, data strip.
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
			for (; i + 35 < n ; i += 36) { //Goes through 32 rows at a time of C and A.
				ctemp = C + j*n + i;
				for (ln = 0; ln < m; ln++){
					k = ln*36;
					temp2 = i+(n*ln);
					for (l = 0; l < 36; l++) {
						*(Asmall+l+(k)) = *(A+l+temp2);
					}
				}
				for (l = 0; l < block; ++l) {
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


					bpoint = &bsmall[m*l];
					for (k = 0; k < m; k++) { //Goes through Goes through width of m, data strip.
						atemp = Asmall+36*k;

						b = _mm_load1_ps(bpoint+k);

						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);
						;
						c2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+4), b), c2);

						c3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+8), b), c3);

						c4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+12), b), c4);

						c5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+16), b), c5);

						c6 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+20), b), c6);

						c7 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+24), b), c7);
						c8 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+28), b), c8);
						c9 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+32), b), c9);


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


				}
			}
			for (; i + 31 < n ; i += 32) { //Goes through 32 rows at a time of C and A.
				ctemp = C + j*n + i;
				for (ln = 0; ln < m; ln++){
					k = ln*32;
					temp2 = i+(n*ln);
					for (l = 0; l < 32; l++) {
						*(Asmall+l+(k)) = *(A+l+temp2);
					}
				}
				for (l = 0; l < block; ++l) {
					ln = l*n;
					c1 = _mm_setzero_ps();
					c2 = c1;
					c3 = c1;
					c4 = c1;
					c5 = c1;
					c6 = c1;
					c7 = c1;
					c8 = c1;

					bpoint = &bsmall[m*l];
					for (k = 0; k < m; k++) { //Goes through Goes through width of m, data strip.
						atemp = Asmall+32*k;

						b = _mm_load1_ps(bpoint+k);

						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);
						;
						c2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+4), b), c2);

						c3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+8), b), c3);

						c4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+12), b), c4);

						c5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+16), b), c5);

						c6 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+20), b), c6);

						c7 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+24), b), c7);

						c8 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+28), b), c8);

					}

					_mm_storeu_ps(ctemp+ln, c1);
					_mm_storeu_ps(ctemp+4+ln, c2);
					_mm_storeu_ps(ctemp+8+ln, c3);
					_mm_storeu_ps(ctemp+12+ln, c4);
					_mm_storeu_ps(ctemp+16+ln, c5);
					_mm_storeu_ps(ctemp+20+ln, c6);
					_mm_storeu_ps(ctemp+24+ln, c7);
					_mm_storeu_ps(ctemp+28+ln, c8);


				}
			}
			for (; i + 27 < n ; i += 28) { //Goes through 32 rows at a time of C and A.
				ctemp = C + j*n + i;
				for (ln = 0; ln < m; ln++){
					k = ln*28;
					temp2 = i+(n*ln);
					for (l = 0; l < 28; l++) {
						*(Asmall+l+(k)) = *(A+l+temp2);
					}
				}
				for (l = 0; l < block; ++l) {
					ln = l*n;
					c1 = _mm_setzero_ps();
					c2 = c1;
					c3 = c1;
					c4 = c1;
					c5 = c1;
					c6 = c1;
					c7 = c1;


					bpoint = &bsmall[m*l];
					for (k = 0; k < m; k++) { //Goes through Goes through width of m, data strip.
						atemp = Asmall+28*k;

						b = _mm_load1_ps(bpoint+k);

						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);
						;
						c2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+4), b), c2);

						c3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+8), b), c3);

						c4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+12), b), c4);

						c5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+16), b), c5);

						c6 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+20), b), c6);

						c7 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+24), b), c7);


					}
					_mm_storeu_ps(ctemp+ln, c1);
					_mm_storeu_ps(ctemp+4+ln, c2);
					_mm_storeu_ps(ctemp+8+ln, c3);
					_mm_storeu_ps(ctemp+12+ln, c4);
					_mm_storeu_ps(ctemp+16+ln, c5);
					_mm_storeu_ps(ctemp+20+ln, c6);
					_mm_storeu_ps(ctemp+24+ln, c7);


				}
			}
			for (; i + 23 < n ; i += 24) { //Goes through 32 rows at a time of C and A.
				ctemp = C + j*n + i;
				for (ln = 0; ln < m; ln++){
					k = ln*24;
					temp2 = i+(n*ln);
					for (l = 0; l < 24; l++) {
						*(Asmall+l+(k)) = *(A+l+temp2);
					}
				}
				for (l = 0; l < block; ++l) {
					ln = l*n;
					c1 = _mm_setzero_ps();
					c2 = c1;
					c3 = c1;
					c4 = c1;
					c5 = c1;
					c6 = c1;


					bpoint = &bsmall[m*l];
					for (k = 0; k < m; k++) { //Goes through Goes through width of m, data strip.
						atemp = Asmall+24*k;
						b = _mm_load1_ps(bpoint+k);
						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);

						c2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+4), b), c2);

						c3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+8), b), c3);

						c4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+12), b), c4);

						c5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+16), b), c5);

						c6 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+20), b), c6);

					}
					_mm_storeu_ps(ctemp+ln, c1);
					_mm_storeu_ps(ctemp+4+ln, c2);
					_mm_storeu_ps(ctemp+8+ln, c3);
					_mm_storeu_ps(ctemp+12+ln, c4);
					_mm_storeu_ps(ctemp+16+ln, c5);
					_mm_storeu_ps(ctemp+20+ln, c6);


				}
			}
			for (; i + 19 < n ; i += 20) { //Goes through 32 rows at a time of C and A.
				ctemp = C + j*n + i;
				for (ln = 0; ln < m; ln++){
					k = ln*20;
					temp2 = i+(n*ln);
					for (l = 0; l < 20; l++) {
						*(Asmall+l+(k)) = *(A+l+temp2);
					}
				}
				for (l = 0; l < block; ++l) {
					ln = l*n;
					c1 = _mm_setzero_ps();
					c2 = c1;
					c3 = c1;
					c4 = c1;
					c5 = c1;


					bpoint = &bsmall[m*l];
					for (k = 0; k < m; k++) { //Goes through Goes through width of m, data strip.
						atemp = Asmall+20*k;
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
			for (; i + 15 < n ; i += 16) { //Goes through 32 rows at a time of C and A.
				ctemp = C + j*n + i;
				for (ln = 0; ln < m; ln++){
					k = ln*16;
					temp2 = i+(n*ln);
					for (l = 0; l < 16; l++) {
						*(Asmall+l+(k)) = *(A+l+temp2);
					}
				}
				for (l = 0; l < block; ++l) {
					ln = l*n;
					c1 = _mm_setzero_ps();
					c2 = c1;
					c3 = c1;
					c4 = c1;


					bpoint = &bsmall[m*l];
					for (k = 0; k < m; k++) { //Goes through Goes through width of m, data strip.
						atemp = Asmall+16*k;
						b = _mm_load1_ps(bpoint+k);
						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);

						c2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+4), b), c2);

						c3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+8), b), c3);

						c4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+12), b), c4);


					}
					_mm_storeu_ps(ctemp+ln, c1);
					_mm_storeu_ps(ctemp+4+ln, c2);
					_mm_storeu_ps(ctemp+8+ln, c3);
					_mm_storeu_ps(ctemp+12+ln, c4);


				}
			}
			for (; i + 11 < n ; i += 12) { //Goes through 32 rows at a time of C and A.
				ctemp = C + j*n + i;
				for (ln = 0; ln < m; ln++){
					k = ln*12;
					temp2 = i+(n*ln);
					for (l = 0; l < 12; l++) {
						*(Asmall+l+(k)) = *(A+l+temp2);
					}
				}
				for (l = 0; l < block; ++l) {
					ln = l*n;
					c1 = _mm_setzero_ps();
					c2 = c1;
					c3 = c1;


					bpoint = &bsmall[m*l];
					for (k = 0; k < m; k++) { //Goes through Goes through width of m, data strip.
						atemp = Asmall+12*k;
						b = _mm_load1_ps(bpoint+k);
						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);

						c2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+4), b), c2);

						c3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+8), b), c3);



					}
					_mm_storeu_ps(ctemp+ln, c1);
					_mm_storeu_ps(ctemp+4+ln, c2);
					_mm_storeu_ps(ctemp+8+ln, c3);


				}
			}
			for (; i + 7 < n ; i += 8) { //Goes through 32 rows at a time of C and A.
				ctemp = C + j*n + i;
				for (ln = 0; ln < m; ln++){
					k = ln*8;
					temp2 = i+(n*ln);
					for (l = 0; l < 8; l++) {
						*(Asmall+l+(k)) = *(A+l+temp2);
					}
				}
				for (l = 0; l < block; ++l) {
					ln = l*n;
					c1 = _mm_setzero_ps();
					c2 = c1;


					bpoint = &bsmall[m*l];
					for (k = 0; k < m; k++) { //Goes through Goes through width of m, data strip.
						atemp = Asmall+8*k;
						b = _mm_load1_ps(bpoint+k);
						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);

						c2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+4), b), c2);


					}
					_mm_storeu_ps(ctemp+ln, c1);
					_mm_storeu_ps(ctemp+4+ln, c2);



				}
			}
			while (i + 3 < n) {
				ctemp = C + j*n + i;
				for (ln = 0; ln < m; ln++){
					k = ln << 2;
					temp2 = i+(n*ln);
					for (l = 0; l < 4; l++) {
						*(Asmall+l+k) = *(A+l+temp2);
					}
				}
				for (l = 0; l < block; ++l) {
					c1 = _mm_setzero_ps();
					bpoint = &bsmall[m*l];
					for (int k = 0; k < m; k++) { //Goes through Goes through width of m, data strip.
						atemp = Asmall+(k<<2);
						b = _mm_load1_ps(bpoint +k);
						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);

					}
					//	printf("test16\n");
					_mm_storeu_ps((C + j*n + i + l*n), c1);

				}
				i += 4;
			}
			//	printf("test3\n");
			if (i < n) {
				ctemp = C + j*n + i;
				for (ln = 0; ln < m; ln++){
					temp2 = n*ln;
					for (l = 0; l < (n - i); l++) {
						*(Asmall+ln+m*l) = *(A+i+l+(temp2));
					}
				}
				for (l = 0; l < (n-i); l++) {
					for (ln = 0; ln < block; ln++) {
						c1sum = 0.0;
						temp2 = m*ln;
						for (k = 0; k < m; k++) {
							c1sum += Asmall[k+l*m]*bsmall[k+(temp2)];
						}
						*(ctemp+l+ln*n) = c1sum;
					}


				}


			}

		}
	}

}

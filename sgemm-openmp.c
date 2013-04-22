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
#pragma omp parallel num_threads(8)
	{
		__m128 c5, b, c1, c2, c3, c4, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15;
		int i, k, j, l, temp, temp2, cinter;
		float *atemp, *ctemp, *bpoint;
		float *Bsmall;

		float *Asmall;
		static int TWENTY = 40;
		static int blocksize = 2;
		float c1sum;
		float bsmall[blocksize*m];
		float small[40*m];
#pragma omp for private(c13, c14, c15, c5, b, c1, c2, c3, c4, c6, c7, c8, c9, c10, c11, c12, c1sum, k, i ,j, ctemp, atemp, cinter, l, temp, temp2, Asmall, small, Bsmall, bpoint, bsmall) schedule(static)
		for (j = 0; j < n; j+= blocksize) { //Goes through column of C
			cinter = (j*n);
			Bsmall = bsmall;
			for(i = 0; i < blocksize && i+j < n; i++) {
				temp = i*m;
				temp2 = (j+i)*(n+1);
				for (k = 0; k < m; k++) {
					*(Bsmall+k+temp) = *(A+temp2+k*n);
				}
			}
			Asmall = small;
			for (i = 0; i + 39 < n ; i += 40) { //Goes through 4 rows at a time of C and A.
				ctemp = C + (cinter) + i;
				for (k = 0; k < m; k++){
					temp = k*40;
					temp2 = i+(n*k);
					for (l = 0; l < 40; l++) {
						*(Asmall+l+(temp)) = *(A+l+temp2);
					}
				}
				for (l = 0; (l < blocksize) && (l+j < n); l++) {
					temp = l*n;
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
					_mm_storeu_ps(ctemp+temp, c1);
					_mm_storeu_ps(ctemp+4+temp, c2);
					_mm_storeu_ps(ctemp+8+temp, c3);
					_mm_storeu_ps(ctemp+12+temp, c4);
					_mm_storeu_ps(ctemp+16+temp, c5);
					_mm_storeu_ps(ctemp+20+temp, c6);
					_mm_storeu_ps(ctemp+24+temp, c7);
					_mm_storeu_ps(ctemp+28+temp, c8);
					_mm_storeu_ps(ctemp+32+temp, c9);
					_mm_storeu_ps(ctemp+36+temp, c10);

				}
			}
			for (; i + 19 < n ; i += 20) { //Goes through 4 rows at a time of C and A.
				ctemp = C + (cinter) + i;
				for (k = 0; k < m; k++){
					temp = k*20;
					temp2 = i+(n*k);
					for (l = 0; l < 20; l++) {
						*(Asmall+l+(temp)) = *(A+l+temp2);
					}
				}
				for (l = 0; (l < blocksize) && (l+j < n); l++) {
					temp = l*n;
					c1 = _mm_setzero_ps();
					c2 = c1;
					c3 = c1;
					c4 = c1;
					c5 = c1;

					bpoint = Bsmall+m*l;
					for (k = 0; k < m; k += 1) { //Goes through Goes through width of m, data strip.
						atemp = Asmall+20*k;
						b = _mm_load1_ps(bpoint+k);
						c1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp), b), c1);

						c2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+4), b), c2);

						c3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+8), b), c3);

						c4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+12), b), c4);

						c5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(atemp+16), b), c5);

					}
					_mm_storeu_ps(ctemp+temp, c1);
					_mm_storeu_ps(ctemp+4+temp, c2);
					_mm_storeu_ps(ctemp+8+temp, c3);
					_mm_storeu_ps(ctemp+12+temp, c4);
					_mm_storeu_ps(ctemp+16+temp, c5);


				}
			}

			while (i + 3 < n) {
				ctemp = C + (cinter) + i;
				for (k = 0; k < m; k++){
					temp = (k <<2);
					temp2 = i+(n*k);
					for (l = 0; l < 4; l++) {
						*(Asmall+l+(temp)) = *(A+l+temp2);
					}
				}
				for (l = 0; (l < blocksize) && (l+j < n); l++) {
					c1 = _mm_setzero_ps();
					bpoint = Bsmall+m*l;
					for (int k = 0; k < m; k += 1) { //Goes through Goes through width of m, data strip.
						//	printf("test4. k:%d, i:%d, ps:%d, j:%d, bi:%d\n", k, i, ps, j, (j * ps) + (i * (ps + 1)));
						atemp = Asmall+(k<<2);
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
				for (k = 0; k < m; k++){
					*(Asmall+(k)) = *(A+i+(n*k));

				}
				for (l = 0; (l < blocksize) && (l+j < n); l++) {
					c1sum = 0;
					for (k = 0; k < m; k +=1) {
						c1sum += small[k]*bsmall[k+(m*l)];
					}
					*(ctemp+l*n) = c1sum;

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



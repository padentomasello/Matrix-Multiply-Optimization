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
	int i, k, j, l, t, ln, temp, temp2, blocksize, btemp, cinter;
	float *atemp, *ctemp, *bpoint;
	float *Bsmall;
	blocksize = 25;
	float *Asmall;
	static int TWENTY = 20;
#pragma omp parallel
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
		float small[20*m];
		//	printf("test2\n");
#pragma omp for private(a1, a2, a3, a4, a5, c5, b, c1, c2, c3, c4, c1sum, k, i ,j, btemp, ctemp, atemp, cinter, l, ln, t, temp, temp2, Asmall, small, Bsmall, bpoint, bsmall)
		for (j = 0; j < n; j+= blocksize) { //Goes through column of C
			btemp = (j*m);
			cinter = (j*n);
			//		printf("test2 j: %d, n:$d,\n,", j, n);
			Bsmall = bsmall;
			for(i = 0; i < blocksize && i+j < n; i++) {
				temp = i*m;
				temp2 = (j+i)*(n+1);
				for (k = 0; k < m; k++) {
					*(Bsmall+k+temp) = *(A+temp2+k*n);
				}
			}
			Asmall = small;
			//	printf("test3\n");
			for (i = 0; i + 19 < n ; i += TWENTY) { //Goes through 4 rows at a time of C and A.
				ctemp = C + (cinter) + i;
				for (t = 0; t < m; t++){
					temp = t*20;
					temp2 = i+(n*t);
					for (l = 0; l < 20; l++) {
						*(Asmall+l+(temp)) = *(A+l+temp2);
					}
				}
				for (l = 0; (l < blocksize) && (l+j < n); l++) {
					ln = l*n;
					c1 = _mm_setzero_ps();
					c2 = c1;
					c3 = c1;
					c4 = c1;
					c5 = c1;
					bpoint = Bsmall+m*l;
					for (k = 0; k < m; k += 1) { //Goes through Goes through width of m, data strip.
						atemp = Asmall+20*k;
						b = _mm_load1_ps(bpoint+k);
						a1 = _mm_loadu_ps(atemp);
						c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
						a2 = _mm_loadu_ps(atemp+4);
						c2 = _mm_add_ps(_mm_mul_ps(a2, b), c2);
						a3 = _mm_loadu_ps(atemp+8);
						c3 = _mm_add_ps(_mm_mul_ps(a3, b), c3);
						a4 = _mm_loadu_ps(atemp+12);
						c4 = _mm_add_ps(_mm_mul_ps(a4, b), c4);
						a5 = _mm_loadu_ps(atemp+16);
						c5 = _mm_add_ps(_mm_mul_ps(a5, b), c5);




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
				//printf("test2\n");
				for (l = 0; (l < blocksize) && (l+j < n); l++) {
					c1 = _mm_setzero_ps();
					bpoint = Bsmall+m*l;
					for (int k = 0; k < m; k += 1) { //Goes through Goes through width of m, data strip.
						//	printf("test4. k:%d, i:%d, ps:%d, j:%d, bi:%d\n", k, i, ps, j, (j * ps) + (i * (ps + 1)));
						b = _mm_load1_ps(bpoint +k);
						//	printf("test5\n");
						a1 = _mm_loadu_ps((A + i + (k * n)));
						//	printf("test6\n");
						c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
						//	printf("test7\n");

					}
					//	printf("test16\n");
					_mm_storeu_ps((C + cinter + i + l*n), c1);

				}
				i += 4;
			}
			//	printf("test3\n");
			while (i < n) {
				for (l = 0; (l < blocksize) && (l+j < n); l++) {
					c1sum = 0;
					for (k = 0; k < m; k +=1) {
						float a1 = A[i + (k * n)];
						float b = bsmall[k+(m*l)];
						c1sum += a1*b;
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



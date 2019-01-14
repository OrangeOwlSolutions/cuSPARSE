#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <cuda_runtime.h>
#include <cusparse.h>

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
// --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) { exit(code); }
	}
}

void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

/***************************/
/* CUSPARSE ERROR CHECKING */
/***************************/
static const char *_cusparseGetErrorEnum(cusparseStatus_t error)
{
	switch (error)
	{

	case CUSPARSE_STATUS_SUCCESS:
		return "CUSPARSE_STATUS_SUCCESS";

	case CUSPARSE_STATUS_NOT_INITIALIZED:
		return "CUSPARSE_STATUS_NOT_INITIALIZED";

	case CUSPARSE_STATUS_ALLOC_FAILED:
		return "CUSPARSE_STATUS_ALLOC_FAILED";

	case CUSPARSE_STATUS_INVALID_VALUE:
		return "CUSPARSE_STATUS_INVALID_VALUE";

	case CUSPARSE_STATUS_ARCH_MISMATCH:
		return "CUSPARSE_STATUS_ARCH_MISMATCH";

	case CUSPARSE_STATUS_MAPPING_ERROR:
		return "CUSPARSE_STATUS_MAPPING_ERROR";

	case CUSPARSE_STATUS_EXECUTION_FAILED:
		return "CUSPARSE_STATUS_EXECUTION_FAILED";

	case CUSPARSE_STATUS_INTERNAL_ERROR:
		return "CUSPARSE_STATUS_INTERNAL_ERROR";

	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

	case CUSPARSE_STATUS_ZERO_PIVOT:
		return "CUSPARSE_STATUS_ZERO_PIVOT";
	}

	return "<unknown>";
}

inline void __cusparseSafeCall(cusparseStatus_t err, const char *file, const int line)
{
	if (CUSPARSE_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUSPARSE error in file '%s', line %d, error %s\nterminating!\n", __FILE__, __LINE__, \
			_cusparseGetErrorEnum(err)); \
			assert(0); \
	}
}

extern "C" void cusparseSafeCall(cusparseStatus_t err) { __cusparseSafeCall(err, __FILE__, __LINE__); }

/********/
/* MAIN */
/********/
int main()
{
	// --- Initialize cuSPARSE
	cusparseHandle_t handle;	cusparseSafeCall(cusparseCreate(&handle));

	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/
	const int N = 4;				// --- Number of rows and columns

	// --- Host side dense matrices
	double *h_A_dense = (double*)malloc(N * N * sizeof(*h_A_dense));
	double *h_B_dense = (double*)malloc(N * N * sizeof(*h_B_dense));
	double *h_C_dense = (double*)malloc(N * N * sizeof(*h_C_dense));

	// --- Column-major ordering
	h_A_dense[0] = 0.4612;  h_A_dense[4] = -0.0006;		h_A_dense[8] = 0.3566;		h_A_dense[12] = 0.0;
	h_A_dense[1] = -0.0006; h_A_dense[5] = 0.4640;		h_A_dense[9] = 0.0723;		h_A_dense[13] = 0.0;
	h_A_dense[2] = 0.3566;  h_A_dense[6] = 0.0723;		h_A_dense[10] = 0.7543;		h_A_dense[14] = 0.0;
	h_A_dense[3] = 0.;		h_A_dense[7] = 0.0;			h_A_dense[11] = 0.0;		h_A_dense[15] = 0.1;

	// --- Column-major ordering
	h_B_dense[0] = 0.;		h_B_dense[4] = 0.;			h_B_dense[8] = 1.;			h_B_dense[12] = 0.;
	h_B_dense[1] = 1.;		h_B_dense[5] = 0.;			h_B_dense[9] = 0.;			h_B_dense[13] = 0.;
	h_B_dense[2] = 0.;		h_B_dense[6] = 1.;			h_B_dense[10] = 0.;			h_B_dense[14] = 0.;
	h_B_dense[3] = 0.;		h_B_dense[7] = 0.;			h_B_dense[11] = 0.;			h_B_dense[15] = 1.;

	// --- Create device arrays and copy host arrays to them
	double *d_A_dense;	gpuErrchk(cudaMalloc(&d_A_dense, N * N * sizeof(*d_A_dense)));
	double *d_B_dense;	gpuErrchk(cudaMalloc(&d_B_dense, N * N * sizeof(*d_B_dense)));
	double *d_C_dense;	gpuErrchk(cudaMalloc(&d_C_dense, N * N * sizeof(*d_C_dense)));
	gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, N * N * sizeof(*d_A_dense), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_B_dense, h_B_dense, N * N * sizeof(*d_B_dense), cudaMemcpyHostToDevice));

	// --- Descriptor for sparse matrix A
	cusparseMatDescr_t descrA;		cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSafeCall(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));

	// --- Descriptor for sparse matrix B
	cusparseMatDescr_t descrB;		cusparseSafeCall(cusparseCreateMatDescr(&descrB));
	cusparseSafeCall(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ONE));

	// --- Descriptor for sparse matrix C
	cusparseMatDescr_t descrC;		cusparseSafeCall(cusparseCreateMatDescr(&descrC));
	cusparseSafeCall(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ONE));

	int nnzA = 0;							// --- Number of nonzero elements in dense matrix A
	int nnzB = 0;							// --- Number of nonzero elements in dense matrix B

	const int lda = N;						// --- Leading dimension of dense matrix

	// --- Device side number of nonzero elements per row of matrix A
	int *d_nnzPerVectorA; 	gpuErrchk(cudaMalloc(&d_nnzPerVectorA, N * sizeof(*d_nnzPerVectorA)));
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_COLUMN, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));

	// --- Device side number of nonzero elements per row of matrix B
	int *d_nnzPerVectorB; 	gpuErrchk(cudaMalloc(&d_nnzPerVectorB, N * sizeof(*d_nnzPerVectorB)));
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_COLUMN, N, N, descrB, d_B_dense, lda, d_nnzPerVectorB, &nnzB));

	// --- Host side number of nonzero elements per row of matrix A
	int *h_nnzPerVectorA = (int *)malloc(N * sizeof(*h_nnzPerVectorA));
	gpuErrchk(cudaMemcpy(h_nnzPerVectorA, d_nnzPerVectorA, N * sizeof(*h_nnzPerVectorA), cudaMemcpyDeviceToHost));

	// --- Host side number of nonzero elements per row of matrix B
	int *h_nnzPerVectorB = (int *)malloc(N * sizeof(*h_nnzPerVectorB));
	gpuErrchk(cudaMemcpy(h_nnzPerVectorB, d_nnzPerVectorB, N * sizeof(*h_nnzPerVectorB), cudaMemcpyDeviceToHost));

	printf("Number of nonzero elements in dense matrix A = %i\n\n", nnzA);
	for (int i = 0; i < N; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, h_nnzPerVectorA[i]);
	printf("\n");

	printf("Number of nonzero elements in dense matrix B = %i\n\n", nnzB);
	for (int i = 0; i < N; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, h_nnzPerVectorB[i]);
	printf("\n");

	// --- Device side sparse matrix
	double *d_A;			gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));
	double *d_B;			gpuErrchk(cudaMalloc(&d_B, nnzB * sizeof(*d_B)));

	int *d_A_RowIndices;	gpuErrchk(cudaMalloc(&d_A_RowIndices, (N + 1) * sizeof(*d_A_RowIndices)));
	int *d_B_RowIndices;	gpuErrchk(cudaMalloc(&d_B_RowIndices, (N + 1) * sizeof(*d_B_RowIndices)));
	int *d_C_RowIndices;	gpuErrchk(cudaMalloc(&d_C_RowIndices, (N + 1) * sizeof(*d_C_RowIndices)));
	int *d_A_ColIndices;	gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));
	int *d_B_ColIndices;	gpuErrchk(cudaMalloc(&d_B_ColIndices, nnzB * sizeof(*d_B_ColIndices)));

	cusparseSafeCall(cusparseDdense2csr(handle, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices));
	cusparseSafeCall(cusparseDdense2csr(handle, N, N, descrB, d_B_dense, lda, d_nnzPerVectorB, d_B, d_B_RowIndices, d_B_ColIndices));

	// --- Host side sparse matrices
	double *h_A = (double *)malloc(nnzA * sizeof(*h_A));
	double *h_B = (double *)malloc(nnzB * sizeof(*h_B));
	int *h_A_RowIndices = (int *)malloc((N + 1) * sizeof(*h_A_RowIndices));
	int *h_A_ColIndices = (int *)malloc(nnzA * sizeof(*h_A_ColIndices));
	int *h_B_RowIndices = (int *)malloc((N + 1) * sizeof(*h_B_RowIndices));
	int *h_B_ColIndices = (int *)malloc(nnzB * sizeof(*h_B_ColIndices));
	int *h_C_RowIndices = (int *)malloc((N + 1) * sizeof(*h_C_RowIndices));
	gpuErrchk(cudaMemcpy(h_A, d_A, nnzA * sizeof(*h_A), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (N + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnzA * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_B, d_B, nnzB * sizeof(*h_B), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_B_RowIndices, d_B_RowIndices, (N + 1) * sizeof(*h_B_RowIndices), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_B_ColIndices, d_B_ColIndices, nnzB * sizeof(*h_B_ColIndices), cudaMemcpyDeviceToHost));

	printf("\nOriginal matrix A in CSR format\n\n");
	for (int i = 0; i < nnzA; ++i) printf("A[%i] = %f ", i, h_A[i]); printf("\n");

	printf("\nOriginal matrix B in CSR format\n\n");
	for (int i = 0; i < nnzB; ++i) printf("B[%i] = %f ", i, h_B[i]); printf("\n");

	printf("\n");
	for (int i = 0; i < (N + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]); printf("\n");

	printf("\n");
	for (int i = 0; i < (N + 1); ++i) printf("h_B_RowIndices[%i] = %i \n", i, h_B_RowIndices[i]); printf("\n");

	printf("\n");
	for (int i = 0; i < nnzA; ++i) printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);

	printf("\n");
	for (int i = 0; i < nnzB; ++i) printf("h_B_ColIndices[%i] = %i \n", i, h_B_ColIndices[i]);

	// --- Performing the matrix - matrix multiplication
	int baseC, nnzC = 0;
	// nnzTotalDevHostPtr points to host memory
	int *nnzTotalDevHostPtr = &nnzC;

	cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

	cusparseSafeCall(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, N, descrB, nnzB,
		d_B_RowIndices, d_B_ColIndices, descrA, nnzA, d_A_RowIndices, d_A_ColIndices, descrC, d_C_RowIndices,
		nnzTotalDevHostPtr));
	if (NULL != nnzTotalDevHostPtr) nnzC = *nnzTotalDevHostPtr;
	else {
		gpuErrchk(cudaMemcpy(&nnzC, d_C_RowIndices + N, sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&baseC, d_C_RowIndices, sizeof(int), cudaMemcpyDeviceToHost));
		nnzC -= baseC;
	}
	int *d_C_ColIndices;	gpuErrchk(cudaMalloc(&d_C_ColIndices, nnzC * sizeof(int)));
	double *d_C;			gpuErrchk(cudaMalloc(&d_C, nnzC * sizeof(double)));
	double *h_C = (double *)malloc(nnzC * sizeof(*h_C));
	int *h_C_ColIndices = (int *)malloc(nnzC * sizeof(*h_C_ColIndices));
	cusparseSafeCall(cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, N, descrB, nnzB,
		d_B, d_B_RowIndices, d_B_ColIndices, descrA, nnzA, d_A, d_A_RowIndices, d_A_ColIndices, descrC,
		d_C, d_C_RowIndices, d_C_ColIndices));

	cusparseSafeCall(cusparseDcsr2dense(handle, N, N, descrC, d_C, d_C_RowIndices, d_C_ColIndices, d_C_dense, N));

	gpuErrchk(cudaMemcpy(h_C, d_C, nnzC * sizeof(*h_C), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_C_RowIndices, d_C_RowIndices, (N + 1) * sizeof(*h_C_RowIndices), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_C_ColIndices, d_C_ColIndices, nnzC * sizeof(*h_C_ColIndices), cudaMemcpyDeviceToHost));

	printf("\nResult matrix C in CSR format\n\n");
	for (int i = 0; i < nnzC; ++i) printf("C[%i] = %f ", i, h_C[i]); printf("\n");

	printf("\n");
	for (int i = 0; i < (N + 1); ++i) printf("h_C_RowIndices[%i] = %i \n", i, h_C_RowIndices[i]); printf("\n");

	printf("\n");
	for (int i = 0; i < nnzC; ++i) printf("h_C_ColIndices[%i] = %i \n", i, h_C_ColIndices[i]);

	gpuErrchk(cudaMemcpy(h_C_dense, d_C_dense, N * N * sizeof(double), cudaMemcpyDeviceToHost));

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
			printf("%f \t", h_C_dense[i * N + j]);
		printf("\n");
	}
}

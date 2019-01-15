#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

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
	double *h_A_dense = (double*)malloc(N * N * sizeof(double));
	double *h_x_dense = (double*)malloc(N *     sizeof(double));
	double *h_y_dense = (double*)malloc(N *     sizeof(double));

	// --- Column-major ordering
	h_A_dense[0] = 0.4612;  h_A_dense[4] = -0.0006;		h_A_dense[8] = 0.3566;		h_A_dense[12] = 0.0;
	h_A_dense[1] = -0.0006; h_A_dense[5] = 0.4640;		h_A_dense[9] = 0.0723;		h_A_dense[13] = 0.0;
	h_A_dense[2] = 0.3566;  h_A_dense[6] = 0.0723;		h_A_dense[10] = 0.7543;		h_A_dense[14] = 0.0;
	h_A_dense[3] = 0.;		h_A_dense[7] = 0.0;			h_A_dense[11] = 0.0;		h_A_dense[15] = 0.1;

	// --- Initializing the data and result vectors
	for (int k = 0; k < N; k++) {
		h_x_dense[k] = 1.;
		h_y_dense[k] = 0.;
	}

	// --- Create device arrays and copy host arrays to them
	double *d_A_dense;	gpuErrchk(cudaMalloc(&d_A_dense, N * N * sizeof(double)));
	double *d_x_dense;	gpuErrchk(cudaMalloc(&d_x_dense, N     * sizeof(double)));
	double *d_y_dense;	gpuErrchk(cudaMalloc(&d_y_dense, N     * sizeof(double)));
	gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, N * N * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_x_dense, h_x_dense, N     * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_y_dense, h_y_dense, N     * sizeof(double), cudaMemcpyHostToDevice));

	// --- Descriptor for sparse matrix A
	cusparseMatDescr_t descrA;		cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSafeCall(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));

	int nnzA = 0;							// --- Number of nonzero elements in dense matrix A

	const int lda = N;						// --- Leading dimension of dense matrix

	// --- Device side number of nonzero elements per row of matrix A
	int *d_nnzPerVectorA; 	gpuErrchk(cudaMalloc(&d_nnzPerVectorA, N * sizeof(*d_nnzPerVectorA)));
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));

	// --- Host side number of nonzero elements per row of matrix A
	int *h_nnzPerVectorA = (int *)malloc(N * sizeof(*h_nnzPerVectorA));
	gpuErrchk(cudaMemcpy(h_nnzPerVectorA, d_nnzPerVectorA, N * sizeof(*h_nnzPerVectorA), cudaMemcpyDeviceToHost));

	printf("Number of nonzero elements in dense matrix A = %i\n\n", nnzA);
	for (int i = 0; i < N; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, h_nnzPerVectorA[i]);
	printf("\n");

	// --- Device side sparse matrix
	double *d_A;			gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));

	int *d_A_RowIndices;	gpuErrchk(cudaMalloc(&d_A_RowIndices, (N + 1) * sizeof(*d_A_RowIndices)));
	int *d_A_ColIndices;	gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));

	cusparseSafeCall(cusparseDdense2csr(handle, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices));

	// --- Host side sparse matrices
	double *h_A = (double *)malloc(nnzA * sizeof(*h_A));
	int *h_A_RowIndices = (int *)malloc((N + 1) * sizeof(*h_A_RowIndices));
	int *h_A_ColIndices = (int *)malloc(nnzA * sizeof(*h_A_ColIndices));
	gpuErrchk(cudaMemcpy(h_A, d_A, nnzA * sizeof(*h_A), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (N + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnzA * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));

	printf("\nOriginal matrix A in CSR format\n\n");
	for (int i = 0; i < nnzA; ++i) printf("A[%i] = %f ", i, h_A[i]); printf("\n");

	printf("\n");
	for (int i = 0; i < (N + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]); printf("\n");

	printf("\n");
	for (int i = 0; i < nnzA; ++i) printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);

	printf("\n");
	for (int i = 0; i < N; ++i) printf("h_x[%i] = %f \n", i, h_x_dense[i]); printf("\n");

	const double alpha = 1.;
	const double beta = 0.;
	cusparseSafeCall(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nnzA, &alpha, descrA, d_A,
		d_A_RowIndices, d_A_ColIndices, d_x_dense, &beta, d_y_dense));

	gpuErrchk(cudaMemcpy(h_y_dense, d_y_dense, N * sizeof(double), cudaMemcpyDeviceToHost));

	printf("\nResult vector\n\n");
	for (int i = 0; i < N; ++i) printf("h_y[%i] = %f ", i, h_y_dense[i]); printf("\n");

}


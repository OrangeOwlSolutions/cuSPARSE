#include <stdio.h>
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
int main() {

	// --- Initialize cuSPARSE
	cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

	// --- Initialize matrix descriptors
	cusparseMatDescr_t descrA, descrB, descrC;
	cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSafeCall(cusparseCreateMatDescr(&descrB));
	cusparseSafeCall(cusparseCreateMatDescr(&descrC));

	const int M = 5;									// --- Number of rows
	const int N = 6;									// --- Number of columns

	const int nnz1 = 10;								// --- Number of non-zero blocks for matrix A
	const int nnz2 = 8;									// --- Number of non-zero blocks for matrix A

	// --- Host vectors defining the first block-sparse matrix
	float *h_csrValA = (float *)malloc(nnz1 * sizeof(float));
	int *h_csrRowPtrA = (int *)malloc((M + 1) * sizeof(int));
	int *h_csrColIndA = (int *)malloc(nnz1 * sizeof(int));

	// --- Host vectors defining the second block-sparse matrix
	float *h_csrValB = (float *)malloc(nnz1 * sizeof(float));
	int *h_csrRowPtrB = (int *)malloc((M + 1) * sizeof(int));
	int *h_csrColIndB = (int *)malloc(nnz1 * sizeof(int));

	h_csrValA[0] = 1.f;
	h_csrValA[1] = 7.f;
	h_csrValA[2] = 1.f;
	h_csrValA[3] = 3.f;
	h_csrValA[4] = -1.f;
	h_csrValA[5] = 10.f;
	h_csrValA[6] = 1.f;
	h_csrValA[7] = -4.f;
	h_csrValA[8] = 1.f;
	h_csrValA[9] = 3.f;

	h_csrRowPtrA[0] = 0;
	h_csrRowPtrA[1] = 3;
	h_csrRowPtrA[2] = 5;
	h_csrRowPtrA[3] = 6;
	h_csrRowPtrA[4] = 8;
	h_csrRowPtrA[5] = 10;

	h_csrColIndA[0] = 0;
	h_csrColIndA[1] = 3;
	h_csrColIndA[2] = 5;
	h_csrColIndA[3] = 2;
	h_csrColIndA[4] = 4;
	h_csrColIndA[5] = 1;
	h_csrColIndA[6] = 0;
	h_csrColIndA[7] = 3;
	h_csrColIndA[8] = 3;
	h_csrColIndA[9] = 5;

	h_csrValB[0] = 3.f;
	h_csrValB[1] = 1.f;
	h_csrValB[2] = -1.f;
	h_csrValB[3] = 1.f;
	h_csrValB[4] = -4.f;
	h_csrValB[5] = -3.f;
	h_csrValB[6] = -2.f;
	h_csrValB[7] = 10.f;

	h_csrRowPtrB[0] = 0;
	h_csrRowPtrB[1] = 2;
	h_csrRowPtrB[2] = 4;
	h_csrRowPtrB[3] = 5;
	h_csrRowPtrB[4] = 7;
	h_csrRowPtrB[5] = 8;

	h_csrColIndB[0] = 0;
	h_csrColIndB[1] = 4;
	h_csrColIndB[2] = 0;
	h_csrColIndB[3] = 1;
	h_csrColIndB[4] = 3;
	h_csrColIndB[5] = 0;
	h_csrColIndB[6] = 1;
	h_csrColIndB[7] = 3;

	// --- Device vectors defining the block-sparse matrices
	float *d_csrValA;		gpuErrchk(cudaMalloc(&d_csrValA, nnz1 * sizeof(float)));
	int *d_csrRowPtrA;		gpuErrchk(cudaMalloc(&d_csrRowPtrA, (M + 1) * sizeof(int)));
	int *d_csrColIndA;		gpuErrchk(cudaMalloc(&d_csrColIndA, nnz1 * sizeof(int)));

	float *d_csrValB;		gpuErrchk(cudaMalloc(&d_csrValB, nnz2 * sizeof(float)));
	int *d_csrRowPtrB;		gpuErrchk(cudaMalloc(&d_csrRowPtrB, (M + 1) * sizeof(int)));
	int *d_csrColIndB;		gpuErrchk(cudaMalloc(&d_csrColIndB, nnz2 * sizeof(int)));

	gpuErrchk(cudaMemcpy(d_csrValA, h_csrValA, nnz1 * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_csrColIndA, h_csrColIndA, nnz1 * sizeof(int), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(d_csrValB, h_csrValB, nnz2 * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_csrRowPtrB, h_csrRowPtrB, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_csrColIndB, h_csrColIndB, nnz2 * sizeof(int), cudaMemcpyHostToDevice));

	// --- Summing the two matrices
	int baseC, nnz3;
	// --- nnzTotalDevHostPtr points to host memory
	int *nnzTotalDevHostPtr = &nnz3;
	cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
	int *d_csrRowPtrC;	 gpuErrchk(cudaMalloc(&d_csrRowPtrC, (M + 1) * sizeof(int)));
	cusparseSafeCall(cusparseXcsrgeamNnz(handle, M, N, descrA, nnz1, d_csrRowPtrA, d_csrColIndA, descrB, nnz2, d_csrRowPtrB, d_csrColIndB, descrC, d_csrRowPtrC, nnzTotalDevHostPtr));
	if (NULL != nnzTotalDevHostPtr) {
		nnz3 = *nnzTotalDevHostPtr;
	}
	else{
		gpuErrchk(cudaMemcpy(&nnz3, d_csrRowPtrC + M, sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&baseC, d_csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost));
		nnz3 -= baseC;
	}
	int *d_csrColIndC;	 gpuErrchk(cudaMalloc(&d_csrColIndC, nnz3 * sizeof(int)));
	float *d_csrValC;	 gpuErrchk(cudaMalloc(&d_csrValC, nnz3 * sizeof(float)));
	float alpha = 1.f, beta = 1.f;
	cusparseSafeCall(cusparseScsrgeam(handle, M, N, &alpha, descrA, nnz1, d_csrValA, d_csrRowPtrA, d_csrColIndA, &beta,	descrB, nnz2, d_csrValB, d_csrRowPtrB, d_csrColIndB, descrC, d_csrValC, d_csrRowPtrC, d_csrColIndC));

	// --- Transforming csr to dense format
	float *d_C;				gpuErrchk(cudaMalloc(&d_C, M * N * sizeof(float)));
	cusparseSafeCall(cusparseScsr2dense(handle, M, N, descrC, d_csrValC, d_csrRowPtrC, d_csrColIndC, d_C, M));

	float *h_C = (float *)malloc(M * N * sizeof(float));
	gpuErrchk(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

	// --- m is row index, n column index
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			printf("%f ", h_C[m + n * M]);
		}
		printf("\n");
	}

	return 0;
}

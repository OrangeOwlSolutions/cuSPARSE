// --- Equivalent to Matlab's
//     m = 5;
//     I = speye(m);
//     e = ones(3, 1);
//     T = spdiags([e -4 * e e],[-1 0 1], 3, 3);
//     kron(I, T)

#include <stdio.h>
#include <assert.h>

#include <cusparse.h>

#define blockMatrixSize			3			// --- Each block of the sparse block matrix is blockMatrixSize x blockMatrixSize

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
	cusparseMatDescr_t descrA, descrC;
	cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSafeCall(cusparseCreateMatDescr(&descrC));

	const int Mb = 5;										// --- Number of blocks along rows
	const int Nb = 5;										// --- Number of blocks along columns

	const int M = Mb * blockMatrixSize;						// --- Number of rows
	const int N = Nb * blockMatrixSize;						// --- Number of columns

	const int nnzb = Mb;									// --- Number of non-zero blocks

	float h_block[blockMatrixSize * blockMatrixSize] = { 4.f, -1.f, 0.f, -1.f, 4.f, -1.f, 0.f, -1.f, 4.f };

	// --- Host vectors defining the block-sparse matrix
	float *h_bsrValA = (float *)malloc(blockMatrixSize * blockMatrixSize * nnzb * sizeof(float));
	int *h_bsrRowPtrA = (int *)malloc((Mb + 1) * sizeof(int));
	int *h_bsrColIndA = (int *)malloc(nnzb * sizeof(int));

	for (int k = 0; k < nnzb; k++) memcpy(h_bsrValA + k * blockMatrixSize * blockMatrixSize, h_block, blockMatrixSize * blockMatrixSize * sizeof(float));

	h_bsrRowPtrA[0] = 0;
	h_bsrRowPtrA[1] = 1;
	h_bsrRowPtrA[2] = 2;
	h_bsrRowPtrA[3] = 3;
	h_bsrRowPtrA[4] = 4;
	h_bsrRowPtrA[5] = 5;

	h_bsrColIndA[0] = 0;
	h_bsrColIndA[1] = 1;
	h_bsrColIndA[2] = 2;
	h_bsrColIndA[3] = 3;
	h_bsrColIndA[4] = 4;

	// --- Device vectors defining the block-sparse matrix
	float *d_bsrValA;		gpuErrchk(cudaMalloc(&d_bsrValA, blockMatrixSize * blockMatrixSize * nnzb * sizeof(float)));
	int *d_bsrRowPtrA;		gpuErrchk(cudaMalloc(&d_bsrRowPtrA, (Mb + 1) * sizeof(int)));
	int *d_bsrColIndA;		gpuErrchk(cudaMalloc(&d_bsrColIndA, nnzb * sizeof(int)));

	gpuErrchk(cudaMemcpy(d_bsrValA, h_bsrValA, blockMatrixSize * blockMatrixSize * nnzb * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_bsrRowPtrA, h_bsrRowPtrA, (Mb + 1) * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_bsrColIndA, h_bsrColIndA, nnzb * sizeof(int), cudaMemcpyHostToDevice));

	// --- Transforming bsr to csr format
	cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;
	const int nnz = nnzb * blockMatrixSize * blockMatrixSize; // --- Number of non-zero elements
	int *d_csrRowPtrC;		gpuErrchk(cudaMalloc(&d_csrRowPtrC, (M + 1) * sizeof(int)));
	int *d_csrColIndC;		gpuErrchk(cudaMalloc(&d_csrColIndC, nnz		* sizeof(int)));
	float *d_csrValC;		gpuErrchk(cudaMalloc(&d_csrValC, nnz		* sizeof(float)));
	cusparseSafeCall(cusparseSbsr2csr(handle, dir, Mb, Nb, descrA, d_bsrValA, d_bsrRowPtrA, d_bsrColIndA, blockMatrixSize, descrC, d_csrValC, d_csrRowPtrC, d_csrColIndC));

	// --- Transforming csr to dense format
	float *d_A;				gpuErrchk(cudaMalloc(&d_A, M * N * sizeof(float)));
	cusparseSafeCall(cusparseScsr2dense(handle, M, N, descrC, d_csrValC, d_csrRowPtrC, d_csrColIndC, d_A, M));

	float *h_A = (float *)malloc(M * N * sizeof(float));
	gpuErrchk(cudaMemcpy(h_A, d_A, M * N * sizeof(float), cudaMemcpyDeviceToHost));

	// --- m is row index, n column index
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			printf("%f ", h_A[m + n * M]);
		}
		printf("\n");
	}

	return 0;
}

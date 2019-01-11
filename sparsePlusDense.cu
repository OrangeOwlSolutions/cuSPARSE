#include <stdio.h>
#include <assert.h>

#include <cusparse.h>

#define		BLOCKSIZEX	16
#define		BLOCKSIZEY	16

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

/*****************************/
/* SETUP DESCRIPTOR FUNCTION */
/*****************************/
void setUpDescriptor(cusparseMatDescr_t &descrA, cusparseMatrixType_t matrixType, cusparseIndexBase_t indexBase) {
	cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSafeCall(cusparseSetMatType(descrA, matrixType));
	cusparseSafeCall(cusparseSetMatIndexBase(descrA, indexBase));
}

/********************************************************/
/* DENSE TO SPARSE CONVERSION FOR REAL DOUBLE PRECISION */
/********************************************************/
void dense2SparseD(const double * __restrict__ d_A_dense, int **d_nnzPerVector, double **d_A,
	int **d_A_RowIndices, int **d_A_ColIndices, int &nnz, cusparseMatDescr_t descrA,
	const cusparseHandle_t handle, const int M, const int N) {

	const int lda = M;                      // --- Leading dimension of dense matrix

	gpuErrchk(cudaMalloc(&d_nnzPerVector[0], M * sizeof(int)));

	// --- Compute the number of nonzero elements per row and the total number of nonzero elements 
	//     the dense d_A_dense
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, descrA, d_A_dense,
		lda, d_nnzPerVector[0], &nnz));

	// --- Device side sparse matrix
	gpuErrchk(cudaMalloc(&d_A[0], nnz * sizeof(double)));
	gpuErrchk(cudaMalloc(&d_A_RowIndices[0], (M + 1) * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_A_ColIndices[0], nnz * sizeof(int)));

	cusparseSafeCall(cusparseDdense2csr(handle, M, N, descrA, d_A_dense, lda, d_nnzPerVector[0],
		d_A[0], d_A_RowIndices[0], d_A_ColIndices[0]));
}

/********************************/
/* SPARSE + DENSE CUSTOM KERNEL */
/********************************/
__global__ void sparsePlusDense(const double * __restrict__ d_A, const int * __restrict__ d_A_RowIndices,
								const int * __restrict__ d_A_ColIndices, const double * __restrict__ d_B, 
								double * __restrict__ d_C, const int M, const int N) {

	const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	const int tidy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((tidx >= N) || (tidy >= M)) return;

	const int	row			= tidy;
	const int	nnzRow		= d_A_RowIndices[tidy + 1] - d_A_RowIndices[tidy];
	if (tidx >= nnzRow)	return;
	
	const int	col			= d_A_ColIndices[d_A_RowIndices[tidy] + tidx];

	//printf("%i %i %f\n", row, col, d_A[d_A_RowIndices[tidy] + tidx]);
	printf("%i %i %f\n", row, col, d_C[row * N + col]);
	d_C[row * N + col] = d_C[row * N + col] + d_A[d_A_RowIndices[tidy] + tidx];
}

/********/
/* MAIN */
/********/
int main() {

	cusparseHandle_t	handle;

	// --- Initialize cuSPARSE
	cusparseSafeCall(cusparseCreate(&handle));

	// --- Initialize matrix descriptors
	cusparseMatDescr_t descrA;
	setUpDescriptor(descrA, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO);

	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/
	const int M = 5;                        // --- Number of rows
	const int N = 4;                        // --- Number of columns

	// --- Host side dense matrix
	double *h_A_dense = (double*)malloc(M * N * sizeof(*h_A_dense));

	// --- Column-major storage
	h_A_dense[0] = 0.4612;  h_A_dense[5] = 0.0;       h_A_dense[10] = 1.3;     h_A_dense[15] = 0.0;
	h_A_dense[1] = 0.0;     h_A_dense[6] = 1.443;     h_A_dense[11] = 0.0;     h_A_dense[16] = 0.0;
	h_A_dense[2] = -0.0006; h_A_dense[7] = 0.4640;    h_A_dense[12] = 0.0723;  h_A_dense[17] = 0.0;
	h_A_dense[3] = 0.3566;  h_A_dense[8] = 0.0;       h_A_dense[13] = 0.7543;  h_A_dense[18] = 0.0;
	h_A_dense[4] = 0.;      h_A_dense[9] = 0.0;       h_A_dense[14] = 0.0;     h_A_dense[19] = 0.1;

	// --- Create device array and copy host array to it
	double *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, M * N * sizeof(double)));
	gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, M * N * sizeof(*d_A_dense), cudaMemcpyHostToDevice));

	/*******************************/
	/* FROM DENSE TO SPARSE MATRIX */
	/*******************************/
	int nnz = 0;            // --- Number of nonzero elements in dense matrix
	int *d_nnzPerVector;	// --- Device side number of nonzero elements per row

	double *d_A;		// --- Sparse matrix values - array of size nnz
	int *d_A_RowIndices;	// --- "Row indices"
	int *d_A_ColIndices;	// --- "Column indices"

	dense2SparseD(d_A_dense, &d_nnzPerVector, &d_A, &d_A_RowIndices, &d_A_ColIndices, nnz, descrA,
		handle, M, N);

	/*************************/
	/* DENSE MATRIX OPERANDS */
	/*************************/
	// --- Host side dense matrix
	double *h_B_dense = (double*)malloc(M * N * sizeof(*h_B_dense));

	// --- Column-major storage
	h_B_dense[0] = 1.5;		h_B_dense[5] = -0.2;    h_B_dense[10] = -0.9;		h_B_dense[15] = 1.1;
	h_B_dense[1] = 2.1;     h_B_dense[6] = 2.0;     h_B_dense[11] = 1.1;		h_B_dense[16] = -0.009;
	h_B_dense[2] = -2;		h_B_dense[7] = -0.82;   h_B_dense[12] = 1.2;		h_B_dense[17] = 1.21;
	h_B_dense[3] = -0.001;  h_B_dense[8] = -1.1;    h_B_dense[13] = 0.887;		h_B_dense[18] = 1.1143;
	h_B_dense[4] = 1.1;     h_B_dense[9] = 2.1;     h_B_dense[14] = -1.1213;    h_B_dense[19] = 5.4334;

	// --- Create device array and copy host array to it
	double *d_B_dense;  gpuErrchk(cudaMalloc(&d_B_dense, M * N * sizeof(double)));
	gpuErrchk(cudaMemcpy(d_B_dense, h_B_dense, M * N * sizeof(*d_B_dense), cudaMemcpyHostToDevice));

	// --- Allocate space for the result e initialize it
	double *d_C_dense;  gpuErrchk(cudaMalloc(&d_C_dense, M * N * sizeof(double)));
	gpuErrchk(cudaMemcpy(d_C_dense, d_B_dense, M * N * sizeof(double), cudaMemcpyDeviceToDevice));

	/*********************************/
	/* RUN THE SPARSE-DENSE ADDITION */
	/*********************************/
	dim3 GridDim(iDivUp(N, BLOCKSIZEX), iDivUp(M, BLOCKSIZEY));
	dim3 BlockDim(BLOCKSIZEX, BLOCKSIZEY);
	sparsePlusDense << <GridDim , BlockDim>> >(d_A, d_A_RowIndices, d_A_ColIndices, d_B_dense, d_C_dense, M, N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	/*******************************************************/
	/* CHECKING THE RESULTS FOR SPARSE TO DENSE CONVERSION */
	/*******************************************************/
	double *h_C_dense = (double *)malloc(M * N * sizeof(double));
	gpuErrchk(cudaMemcpy(h_C_dense, d_C_dense, M * N * sizeof(double), cudaMemcpyDeviceToHost));

	printf("\nFirst dense operand matrix (column-major storage) \n");
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++)
			printf("%f\t", h_A_dense[n * M + m]);
		printf("\n");
	}

	printf("\nSecond dense operand matrix (row-major storage) \n");
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++)
			printf("%f\t", h_B_dense[n + m * N]);
		printf("\n");
	}

	printf("\nReference dense matrix (the first has column-major storage, the second row-major\n");
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++)
			printf("%f\t", h_A_dense[n * M + m] + h_B_dense[n + m * N]);
		printf("\n");
	}

	printf("\nSecond dense operand matrix (row-major storage) \n");
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++)
			printf("%f\t", h_C_dense[n + m * N]);
		printf("\n");
	}

	return 0;
}

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
	const cusparseHandle_t handle, const int Nrows, const int Ncols) {

	const int lda = Nrows;                      // --- Leading dimension of dense matrix

	gpuErrchk(cudaMalloc(&d_nnzPerVector[0], Nrows * sizeof(int)));

	// --- Compute the number of nonzero elements per row and the total number of nonzero elements 
	//     the dense d_A_dense
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, d_A_dense,
		lda, d_nnzPerVector[0], &nnz));

	// --- Device side sparse matrix
	gpuErrchk(cudaMalloc(&d_A[0], nnz * sizeof(double)));
	gpuErrchk(cudaMalloc(&d_A_RowIndices[0], (Nrows + 1) * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_A_ColIndices[0], nnz * sizeof(int)));

	cusparseSafeCall(cusparseDdense2csr(handle, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector[0],
		d_A[0], d_A_RowIndices[0], d_A_ColIndices[0]));
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
	setUpDescriptor(descrA, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ONE);

	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/
	const int Nrows = 5;                        // --- Number of rows
	const int Ncols = 4;                        // --- Number of columns
	const int N = Nrows;

	// --- Host side dense matrix
	double *h_A_dense = (double*)malloc(Nrows * Ncols * sizeof(*h_A_dense));

	// --- Column-major storage
	h_A_dense[0] = 0.4612;  h_A_dense[5] = 0.0;       h_A_dense[10] = 1.3;     h_A_dense[15] = 0.0;
	h_A_dense[1] = 0.0;     h_A_dense[6] = 1.443;     h_A_dense[11] = 0.0;     h_A_dense[16] = 0.0;
	h_A_dense[2] = -0.0006; h_A_dense[7] = 0.4640;    h_A_dense[12] = 0.0723;  h_A_dense[17] = 0.0;
	h_A_dense[3] = 0.3566;  h_A_dense[8] = 0.0;       h_A_dense[13] = 0.7543;  h_A_dense[18] = 0.0;
	h_A_dense[4] = 0.;      h_A_dense[9] = 0.0;       h_A_dense[14] = 0.0;     h_A_dense[19] = 0.1;

	// --- Create device array and copy host array to it
	double *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, Nrows * Ncols * sizeof(double)));
	gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, Nrows * Ncols * sizeof(*d_A_dense), cudaMemcpyHostToDevice));

	/*******************************/
	/* FROM DENSE TO SPARSE MATRIX */
	/*******************************/
	int nnz = 0;            // --- Number of nonzero elements in dense matrix
	int *d_nnzPerVector;	// --- Device side number of nonzero elements per row

	double *d_A;		// --- Sparse matrix values - array of size nnz
	int *d_A_RowIndices;	// --- "Row indices"
	int *d_A_ColIndices;	// --- "Column indices"

	dense2SparseD(d_A_dense, &d_nnzPerVector, &d_A, &d_A_RowIndices, &d_A_ColIndices, nnz, descrA,
		handle, Nrows, Ncols);

	/*******************************************************/
	/* CHECKING THE RESULTS FOR DENSE TO SPARSE CONVERSION */
	/*******************************************************/
	// --- Host side number of nonzero elements per row
	int *h_nnzPerVector = (int *)malloc(Nrows * sizeof(int));
	gpuErrchk(cudaMemcpy(h_nnzPerVector, d_nnzPerVector, Nrows * sizeof(int), cudaMemcpyDeviceToHost));

	printf("Number of nonzero elements in dense matrix = %i\n\n", nnz);
	for (int i = 0; i < Nrows; ++i)
		printf("Number of nonzero elements in row %i = %i \n", i, h_nnzPerVector[i]);
	printf("\n");

	// --- Host side sparse matrix
	double *h_A = (double *)malloc(nnz * sizeof(double));
	int *h_A_RowIndices = (int *)malloc((Nrows + 1) * sizeof(int));
	int *h_A_ColIndices = (int *)malloc(nnz * sizeof(int));
	gpuErrchk(cudaMemcpy(h_A, d_A, nnz * sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (Nrows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnz * sizeof(int), cudaMemcpyDeviceToHost));

	printf("\nOriginal matrix in CSR format\n\n");
	for (int i = 0; i < nnz; ++i) printf("A[%i] = %f\n", i, h_A[i]);
	printf("\n\n");

	for (int i = 0; i < (Nrows + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]);
	printf("\n");
	for (int i = 0; i < nnz; ++i) printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);

	/*******************************/
	/* FROM SPARSE TO DENSE MATRIX */
	/*******************************/
	double *d_A_denseReconstructed; gpuErrchk(cudaMalloc(&d_A_denseReconstructed,
		Nrows * Ncols * sizeof(double)));
	cusparseSafeCall(cusparseDcsr2dense(handle, Nrows, Ncols, descrA, d_A, d_A_RowIndices, d_A_ColIndices,
		d_A_denseReconstructed, Nrows));

	/*******************************************************/
	/* CHECKING THE RESULTS FOR SPARSE TO DENSE CONVERSION */
	/*******************************************************/
	double *h_A_denseReconstructed = (double *)malloc(Nrows * Ncols * sizeof(double));
	gpuErrchk(cudaMemcpy(h_A_denseReconstructed, d_A_denseReconstructed, Nrows * Ncols * sizeof(double),
		cudaMemcpyDeviceToHost));

	printf("\nReconstructed dense matrix \n");
	for (int m = 0; m < Nrows; m++) {
		for (int n = 0; n < Ncols; n++)
			printf("%f\t", h_A_denseReconstructed[n * Nrows + m]);
		printf("\n");
	}

	return 0;
}

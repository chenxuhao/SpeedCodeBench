#include "graph.h"

typedef float T;

bool readColMajorMatrixFile(std::string fn, int &nr_row, int &nr_col, std::vector<T>&v);
bool randomInitMatrix(int nrow, int ncol, std::vector<T>&v);

extern "C" {
//void SpmDm(GraphF &g, const T *B, int matBcol, T *C);

// Inputs: sparse matrix A, dense matrix B
// Output: dense matrix C
void SpmDm(char transa, char transb, 
           vidType m, eidType nnz, int k, 
           T alpha, const eidType *A_rowptr,
           const vidType *A_colidx, const T *A_values, 
           int lda, const T *B, int ldb, 
           T beta, T *C, int ldc);
}

int main(int argc, char *argv[]) {
  printf("Sparse Matrix Dense Matrix Multiplication\n");
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph-prefix>\n";
    std::cout << "Example: " << argv[0] << " inputs/citeseer\n";
    exit(1);
  }
  GraphF g(argv[1], 0, 0, 0, 1, 1, 0, 0);
  g.print_meta_data();
  vidType m = g.V();
  eidType nnz = g.E();

  int matBrow, matBcol;
  std::vector<T> matBT;
  std::string matBFile = "";
  if (argc > 2) matBFile = argv[2];
  // load B^T
  if (matBFile != "")
    readColMajorMatrixFile(matBFile, matBcol, matBrow, matBT);
  else {
    matBrow = m;
    matBcol = 128;
    matBT.resize(matBrow*matBcol);
    randomInitMatrix(matBcol, matBrow, matBT);
  }
  assert(matBrow == m);
 
  auto Ap = g.in_rowptr();
  auto Aj = g.in_colidx();
  auto Ax = g.get_elabel_ptr();
  for (eidType i = 0; i < g.E(); i++) assert(Aj[i] < m);
  std::vector<T> matC(m*matBcol);
  // A: m x m
  // B: m x matBcol
  // C: m x matBcol
  SpmDm('N', 'T', m, nnz, matBcol, 1.0f, Ap, Aj, Ax, m,
        &matBT.front(), matBcol, 0.0f, &matC.front(), m);
  return 0;
}

bool readColMajorMatrixFile(std::string fn, int &nr_row, int &nr_col, std::vector<T>&v) {
  std::cerr << "Opening file:"<< fn << std::endl;
  std::fstream f(fn, std::fstream::in);
  if ( !f.good() ) {
    return false;
  }
  // Read # of rows and cols
  f >> nr_row;
  f >> nr_col;
  float data;
  std::cerr << "Matrix dimension: "<<nr_row<<"x"<<nr_col<<std::endl;
  while (f.good() ) {
    f >> data;
    v.push_back(data);
  }
  v.pop_back(); // remove the duplicated last element
  return true;
}

bool randomInitMatrix(int nrow, int ncol, std::vector<T>&v) {
  srand(13);
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      //v[i*ncol+j] = rand() / (RAND_MAX + 1.0);
      v[i*ncol+j] = 0.3;
    }
  }
  return true;
}


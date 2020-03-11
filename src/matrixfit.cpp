// [[Rcpp::depends(RcppEigen)]]
#ifndef WIN_BUILD
// [[Rcpp::plugins(cpp14)]]
#else
// [[Rcpp::plugins(cpp11)]]
#endif
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <vector>
#include <RcppEigen.h>
#include <SymEigs.h>
#include <Rcpp.h>
#include "matops.h"
#include <R.h>
#include <Rinternals.h>

using namespace std;
using namespace Rcpp;
using namespace Eigen;
using namespace Spectra;
typedef Eigen::Map<const Eigen::VectorXd> MapConstVec;
#ifndef WIN_BUILD
#include <rsvd/Constants.hpp>
#include <rsvd/ErrorEstimators.hpp>
#include <rsvd/RandomizedSvd.hpp>
using namespace Rsvd;
#endif


struct value_index {
  double value;
  int index;
};

bool smaller(const value_index& x, const value_index& y) {
  return x.value < y.value;
}

bool bigger(const value_index& x, const value_index& y) {
  return x.value > y.value;
}

#ifndef WIN_BUILD
Eigen::MatrixXd random_trun_svd(Eigen::MatrixXd X, int k) {
  mt19937_64 randomEngine{};
  randomEngine.seed(1029);
  RandomizedSvd<MatrixXd, mt19937_64, SubspaceIterationConditioner::Lu> rsvd(randomEngine);
  rsvd.compute(X, k);
  return rsvd.matrixU()*rsvd.singularValues().asDiagonal()*rsvd.matrixV().adjoint();
}
#endif




// [[Rcpp::export]]
Eigen::MatrixXd trun_svd(Eigen::MatrixXd X, int k) {
  int m = X.rows();
  int n = X.cols();
  MatrixXd Y(m, n);
  int K = k;

  Rcpp::NumericVector ctr_vec = 0;
  Rcpp::NumericVector scl_vec = 0;
  MapConstVec ctr_map(ctr_vec.begin(), m);
  MapConstVec scl_map(scl_vec.begin(), m);
  bool center = false;
  bool scale = false;
  SEXP A_mat = PROTECT(Rf_allocMatrix(REALSXP, m, n));
  double *rans = REAL(A_mat);

  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++)
      rans[i + m*j] = X(i, j);
  }
  // Operation for original matrix
  // MatProd* op_orig = get_mat_prod_op(A_mat, m, n, A_mat, 1);
  MatProd* op_orig = new MatProd_matrix(A_mat, m, n);
  // Operation for SVD
  MatProd* op;

  if (m > n) {
    op = new SVDTallOp(op_orig, center, scale, ctr_map, scl_map);
    SymEigsSolver<double, LARGEST_ALGE, MatProd> eig_r(op, K, 2 * K + 1 > n ? n : 2 * K + 1);
    // MatrixXd R = X.transpose() * X;
    // DenseSymMatProd<double> op_r(R);
    // SymEigsSolver< double, LARGEST_ALGE, DenseSymMatProd<double> > eig_r(&op_r, K, 2 * K > m ? m : 2 * K);
    eig_r.init();
    UNPROTECT(1);
    int nconv = eig_r.compute();
    VectorXd evalues;
    if (eig_r.info() == SUCCESSFUL) {
      evalues = eig_r.eigenvalues();
      if(nconv < K){
        Rcpp::warning("only %d singular values converged, less than K = %d", nconv, K);
        K = nconv;
      }
      VectorXd d = evalues.head(K);
      d = d.array().sqrt();

      MatrixXd v = eig_r.eigenvectors(K);
      MatrixXd u = X * v;
      MatrixXd D;
      D.setIdentity(K, K);

      for(int i = 0; i < K; i++)
      {
        u.col(i).array() /= d(i);
        D(i, i) = d(i);

      }
      Y = u * D * v.transpose();
    }
  } else {
    //MatProd* L;
    // L = new SVDWideOp(op_orig);
    op = new SVDWideOp(op_orig, center, scale, ctr_map, scl_map);
    SymEigsSolver<double, LARGEST_ALGE, MatProd> eig_l(op, K, 2 * K + 1 > n ? n : 2 * K + 1);
    // MatrixXd L = X * X.transpose();
    // DenseSymMatProd<double> op_l(L);
    // SymEigsSolver< double, LARGEST_ALGE, DenseSymMatProd<double> > eig_l(&op_l, K, 2 * K > m ? m : 2 * K);
    eig_l.init();
    UNPROTECT(1);
    int nconv = eig_l.compute();
    VectorXd evalues;
    if (eig_l.info() == SUCCESSFUL) {
      evalues = eig_l.eigenvalues();
      if(nconv < K){
        Rcpp::warning("only %d singular values converged, less than K = %d", nconv, K);
        K = nconv;
      }

      VectorXd d = evalues.head(K);
      d = d.array().sqrt();

      MatrixXd u = eig_l.eigenvectors(K);
      MatrixXd v = X.transpose() * u;
      MatrixXd D;
      D.setIdentity(K, K);

      for(int i = 0; i < K; i++)
      {
        v.col(i).array() /= d(i);
        D(i, i) = d(i);
      }

      Y = u * D * v.transpose();
      // MatrixXd vec_l = eig_l.eigenvectors(K);
      // Y = vec_l * vec_l.transpose() * X;
    }
  }
  return Y;
}

MatrixXd DS(MatrixXd M, MatrixXd L, vector<value_index> imp, int s) {
  int m = M.rows();
  int n = M.cols();
  MatrixXd S = MatrixXd::Zero(m, n);
  MatrixXd S_t = M - L;
  for (int k = 0; k < s; ++k) {
    int i = int(imp[k].index / n);
    int j = imp[k].index % n;
    S(i, j) = S_t(i, j);
  }
  return S;
}

//' @noRd
//' @param omega The matrix index of the observed value
//' @param X The obeserved value of the matrix
//' @param m, n The dimension of the matrix
//' @param rank The rank of matrix
//' @param max_it	 maximum number of iterations.
//' @param tol convergence threshold, measured as the relative change in the Frobenius norm between two successive estimates.
//' @param type computing singular value decomposition, 1 is truncated singular value decomposition, 2 is randomized singular value decomposition
//' @description Use Rcpp to fit a low-rank matrix approximation to a matrix with two method computing singular value decomposition.
// [[Rcpp::export]]
List kkt_fix(Eigen::MatrixXi &omega, Eigen::VectorXd &X, int m, int n, int rank, int max_it, double tol, int type)
{
  // when rho = 1, it is equivalent to Hard Impute
  int l = omega.rows();
  double temp = X.mean();
  Eigen::MatrixXd Z_old = MatrixXd::Constant(m, n, temp);
  Eigen::MatrixXd Z_new(m, n);
  Eigen::MatrixXd lambda = MatrixXd::Zero(m, n);
  double eps = 1;
  int count = 0;

  Eigen::MatrixXd (*svd_method) (Eigen::MatrixXd, int);
#ifndef WIN_BUILD
  if(type == 1) {
    svd_method = &trun_svd;
  } else {
    svd_method = &random_trun_svd;
  }
#else
  svd_method = &trun_svd;
#endif

 while (eps > tol && count < max_it)
 {
    for (int i = 0; i < l; i++)
    {
      int r = omega(i, 0);
      int c = omega(i, 1);
      lambda(r, c) = X(i) - Z_old(r, c);
    }
    Z_new = svd_method(Z_old + lambda, rank);
    eps = (Z_new - Z_old).squaredNorm() / Z_old.squaredNorm();
    count++;
    Z_old = Z_new;
  }

  // cout << count << ',' << eps << ',' << rank << endl;
  return Rcpp::List::create(Z_new, count);
}

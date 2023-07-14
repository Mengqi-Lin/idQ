// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
using namespace Rcpp;
using namespace arma;

//' get ideal response for DINA
//'
//' replace Z by 1-Z and 1- the resultant matrix gives the
//' ideal response matrix for DINO
//'
//' @param Q Q matrix of size J by K
//' @param Z latent attribute profiles of individuals; N by K
//'
//' @return binary ideal response matrix of size N by J
//'
//'
//' @examples
//' Z <- rbind(c(1,1,1),c(0,0,1))
//' Q <- matrix(c(0,0,0, 0,0,1, 0,1,0,
//'               0,1,1, 1,0,0, 1,0,1,
//'               1,1,0, 1,1,1),ncol=3,byrow=TRUE)
//' get_ideal_resp(Q,Z)
//'
//' get_ideal_resp(Q=matrix(c(1,1,0,
//'                           0,0,1,
//'                           0,1,0,
//'                           1,0,1),nrow=4,byrow=TRUE),
//'                Z = rbind(c(1,1,1),c(0,0,0)))
//' @useDynLib slamR
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
arma::mat get_ideal_resp(arma::mat Q, arma::mat Z){
  int J = Q.n_rows, K = Q.n_cols, A = Z.n_rows;
  arma::mat xi(A,J);xi.ones();
  for (int a=0;a<A;a++){
    for (int j=0;j<J; j++){
      for (int k=0;k<K; k++){
        xi(a,j) *= pow(Z(a,k),Q(j,k));
      }
    }
  }

  // for  (int j=0;j<J; j++){
  //   for (int k=0;k<K; k++){
  //     xi.col(j) %= pow(Z.col(k),Q(j,k));
  //   }
  // }
  return(xi);
}



// //' 7 shaped binary array operation by bsxfun from matlab
// //'
// //' this function also multiple along the rows of A and columns of B
// //'
// //' @param A matrix of size d1 by d2
// //' @param B matrix of size d3 by d1
// //' @param collapse
// //'
// //' @return a matrix of dimension d3 by d2
// //'
// //' @examples
// //' bsxfun_7_pow(matrix(c(1,2,3,4,5,6),nrow=3,byrow=TRUE),
// //' matrix(c(1:18),nrow=6,byrow=TRUE))
// //' @useDynLib slamR
// //' @importFrom Rcpp sourceCpp
// //' @export
// // [[Rcpp::export]]
// arma::mat bsxfun_7_pow(arma::mat A, arma::mat B){
//   int d1 = A.n_rows, d2 = A.n_cols, d3 = B.n_rows;
//   arma::mat xi(d3,d2);xi.ones();
//   for (int i=0;i<d3;i++){
//     for (int j=0;j<d2; j++){
//       for (int k=0;k<d1; k++){
//         xi(i,j) *= pow(A(k,j),B(i,k));
//       }
//     }
//   }
//   return(xi);
// }

//' 7 shaped binary array operation by bsxfun from matlab
//'
//' this function also multiple along the rows of A and columns of B
//'
//' @param A matrix of size d1 by d2
//' @param B matrix of size d3 by d1
//'
//' @return an array of dimension d3 by d1 by d2
//'
//' @examples
//' bsxfun_7_pow(matrix(c(1,2,3,4,5,6),nrow=3,byrow=TRUE),
//' matrix(c(1:18),nrow=6,byrow=TRUE))
//' @useDynLib slamR
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
arma::cube bsxfun_7_pow(arma::mat A, arma::mat B){
  int d1 = A.n_rows, d2 = A.n_cols, d3 = B.n_rows;
  arma::cube xi(d3,d1,d2);xi.ones();
  for (int i=0;i<d3;i++){
    for (int k=0;k<d1; k++){
      for (int j=0;j<d2; j++){
        xi(i,k,j) = pow(A(k,j),B(i,k));
      }
    }
  }
  return(xi);
}

//' 7 shaped binary array operation by bsxfun from matlab but
//' with "hinge" shape
//'
//'
//' @param A matrix of size d1 by d2
//' @param B matrix of size d3 by d2
//'
//' @return a matrix of dimension d3 by d1 by d2
//'
//' @examples
//' bsxfun_7_hinge_pow(matrix(c(1,2,3,4,5,6),nrow=3,byrow=TRUE),
//' matrix(c(1:18),nrow=9,byrow=TRUE))
//' @useDynLib slamR
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
arma::cube bsxfun_7_hinge_pow(arma::mat A, arma::mat B){
  int d1 = A.n_rows, d2 = A.n_cols, d3 = B.n_rows;
  arma::cube xi(d3,d1,d2);xi.ones();
  for (int i=0;i<d3;i++){
    for (int k=0;k<d1; k++){
      for (int j=0;j<d2; j++){
        xi(i,k,j) = pow(A(k,j),B(i,j));
      }
    }
  }
  return(xi);
}


//' 7 shaped binary array operation by bsxfun from matlab but
//' with "hinge" shape; multiple along dimension d2
//'
//'
//' @param A matrix of size d1 by d2
//' @param B matrix of size d3 by d2
//'
//' @return a matrix of dimension d3 by d1 by d2
//'
//' @examples
//' bsxfun_7_hinge_pow_prod(matrix(c(1,2,3,4,5,6),nrow=3,byrow=TRUE),
//' matrix(c(1:18),nrow=9,byrow=TRUE))
//' @useDynLib slamR
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
arma::mat bsxfun_7_hinge_pow_prod(arma::mat A, arma::mat B){
  int d1 = A.n_rows, d2 = A.n_cols, d3 = B.n_rows;
  arma::mat xi(d3,d1);xi.ones();
  for (int i=0;i<d3;i++){
    for (int k=0;k<d1; k++){
      for (int j=0;j<d2; j++){
        xi(i,k) *= pow(A(k,j),B(i,j));
      }
    }
  }
  return(xi);
}


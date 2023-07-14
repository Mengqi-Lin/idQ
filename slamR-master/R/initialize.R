# helper functions

#' get initial Q and Z
#'
#' @param N sample size
#' @param Q0 J by K initial Q
#' @param Z0 N by K initial Z
#'
#' @return
#' \itemize{
#' \item Q_ini
#' \item Z_ini
#' }
#' @export

get_initial_s <- function(N,Q0,Z0){ # deal with all zero rows of Z
  #starts screening:

  # INIT: 1. set initial value for Q
  J <- nrow(Q0)
  K <- ncol(Q0)
  Q_ini <- Q0
  len_pert <- floor(J*K/10)
  rowind_pert <- sample(1:J,len_pert,replace = TRUE)
  colind_pert <- sample(1:K,len_pert,replace = TRUE)

  for (ii in 1:len_pert){
    Q_ini[rowind_pert[ii],colind_pert[ii]] <-
      1-Q_ini[rowind_pert[ii],colind_pert[ii]]
  }
  ind_all0row <- which(bin2ind(Q_ini)==0)
  if (length(ind_all0row)>0){
    for (jj in 1:length(ind_all0row)){
      Q_ini[ind_all0row[jj],sample(1:K,floor(K/2))] <- rep(1,floor(K/2))
    }
  }
  cat("==[slamR] number of entries in Q: ", J*K, ", number of difference: ",sum(sum(abs(Q_ini-Q0))),"==\n")

  # INIT: 2.set initial value for Z; randomly perturb half entries of the X[,1:K]
  Z_ini <- Z0
  len_pert <- floor(N*K/5)
  rowind_pert <- sample(1:N,len_pert,replace = TRUE)
  colind_pert <- sample(1:K,len_pert,replace = TRUE)
  for (ii in 1:len_pert){
    Z_ini[rowind_pert[ii],colind_pert[ii]] <-
      1-Z_ini[rowind_pert[ii],colind_pert[ii]]
  }
  ind_all0row <- which(bin2ind(Z_ini)==0)
  if (length(ind_all0row)>0){
    for (jj in 1:length(ind_all0row)){
      Z_ini[ind_all0row[jj],sample(1:K,floor(K/2))] <- rep(1,floor(K/2))
    }
  }
  cat("==[slamR] number of entries in Z: ", N*K, ", number of difference: ",sum(sum(abs(Z_ini-Z0))),"==\n")

  make_list(Q_ini,Z_ini)
}



# helper functions

#' get initial Q and Z
#'
#' @param N sample size
#' @param Q0 J by K initial Q
#' @param Z0 N by K initial Z
#'
#' @return
#' \itemize{
#' \item Q_ini
#' \item Z_ini
#' }
#' @export

get_initial_n <- function(N,Q0,Z0){ # deal with all zero rows of Z
  #starts screening:

  # INIT: 1. set initial value for Q
  J <- nrow(Q0)
  K <- ncol(Q0)
  Q_ini <- Q0
  len_pert <- floor(J*K/10)
  rowind_pert <- sample(1:J,len_pert,replace = TRUE)
  colind_pert <- sample(1:K,len_pert,replace = TRUE)

  for (ii in 1:len_pert){
    Q_ini[rowind_pert[ii],colind_pert[ii]] <-
      1-Q_ini[rowind_pert[ii],colind_pert[ii]]
  }
  ind_all0row <- which(bin2ind(Q_ini)==0)
  if (length(ind_all0row)>0){
    for (jj in 1:length(ind_all0row)){
      Q_ini[ind_all0row[jj],sample(1:K,floor(K/2))] <- rep(1,floor(K/2))
    }
  }
  cat("==[slamR] number of entries in Q: ", J*K, ", number of difference: ",sum(sum(abs(Q_ini-Q0))),"==\n")

  # INIT: 2.set initial value for Z; randomly perturb half entries of the X[,1:K]
  Z_ini <- Z0
  len_pert <- floor(N*K/5)
  rowind_pert <- sample(1:N,len_pert,replace = TRUE)
  colind_pert <- sample(1:K,len_pert,replace = TRUE)
  for (ii in 1:len_pert){
    Z_ini[rowind_pert[ii],colind_pert[ii]] <-
      1-Z_ini[rowind_pert[ii],colind_pert[ii]]
  }


  cat("==[slamR] number of entries in Z: ", N*K, ", number of difference: ",sum(sum(abs(Z_ini-Z0))),"==\n")

  make_list(Q_ini,Z_ini)
}



























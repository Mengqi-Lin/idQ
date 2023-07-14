# screening

## screen Q using Gibbs (Alternating Gibbs)
##
## @param X N by J binary data matrix
## @param Z_ini N by K initial latent attributes
## @param Q_ini J by K initial Q matrix
## @param max_iter maximum iterations (e.g., 50)
## @param err_prob noise level
##
## @return
## \itemize{
## \item Z_est Estimated latent attributes for all people
## \item Z_candi candidate latent attribute patterns (unique)
##\item Q_arr a list of Q matrices obtained from the algorithm
##\item c J dimensional 1-slipping parameter
## \item g J dimensional guessing parameter
## }
##
## @export
#screen_Q_gibbs_large <- function(X,Z_ini,Q_ini,max_iter,err_prob){
#  J <- nrow(Q_ini)
#  K <- ncol(Q_ini)
#
#  Q <- Q_ini
#  range_gen <- (1-2*err_prob)/4 #?
#  g_new <- err_prob + (-range_gen/2+range_gen*stats::runif(J))
#  c_new <- 1-err_prob + (-range_gen/2+range_gen*stats::runif(J))
#
#  iter <- 0
#  err  <- 1
#  G <- 6 # 2*m_burnin, for example.
#  m_burnin <- 3
#
#  Z <- Z_ini
#  Q_arr <- vector("list",max_iter)
#
#  # the following two arrays store probabilities used to sample Q:
#  while (err > 1e-3 & iter < max_iter){
#    g <- g_new
#    c <- c_new
#    Q_old <- Q
#
#    # E-step:
#    Rcg <- sweep(X,2,(log_prac(c)-log_prac(g)),"*")+
#      sweep(1-X,2,log_prac(1-c)-log_prac(1-g),"*")
#    iter <- iter +1
#
#    ZR <- 0 # ideal response matrix.
#    ZZ <- 0
#    num <- 0
#
#    # start Gibbs sampling of attribute profiles for G times:
#    for (mm in 1:G){
#      for (k in 1:K){ #iterate over attributes:
#        ZR_k <- get_ideal_resp(Q[,-k,drop=FALSE],
#                               Z[,-k,drop=FALSE])
#        P <- (ZR_k*Rcg)%*%Q[,k,drop=FALSE]#?
#        #p_nume <- exp(c(P)) # potential numerical overflow!
#        #Z[,k] <- rvbern(p_nume/(p_nume+1))
#        Z[,k] <- rvbern(exp(c(P)-
#                              c(apply(cbind(c(P),0),1,matrixStats::logSumExp))))
#      }
#      if (mm > m_burnin | iter >200){
#        ZZ <- ZZ+Z
#        ZR <- ZR+get_ideal_resp(Q,Z)
#        num <- num+1
#      }
#    }
#    # end of Gibbs "sampling" of latent attributes.
#    ave_ZR_new <- ZR/num
#    ave_Z_new  <- ZZ/num
#
#
#    # new step for updating Q:
#    QQ  <- 0
#    num <- 0
#    # start Gibbs sampling of Q-matrix for G time:
#    for (mm in 1:G){
#      for (k in 1:K){
#        ZR_kq <- get_ideal_resp(Q[,-k,drop=FALSE],ave_Z_new[,-k,drop=FALSE])
#        inside_kq <- matrix((1-ave_Z_new[,k]),nrow=1)%*%(ZR_kq*Rcg)
#        Q[,k] <- rvbern(1/(1+exp(c(inside_kq))))
#      }
#
#      if (mm>m_burnin | iter >200){
#        QQ <- QQ+Q
#        num <- num+1
#      }
#    }
#    # end of Gibb sampling for Q.
#
#    ave_Q_new <- QQ/num
#    Q <- 0+(ave_Q_new > 1/2)
#    Q_arr[[iter]] <- Q
#    # end of updating Q.
#
#    ## new M step:
#    if (iter ==1){
#      ave_ZR <- ave_ZR_new
#      ave_Z  <- ave_Z_new
#      ave_Q  <- ave_Q_new
#    } else{
#      step <- 1/iter
#      ave_ZR <- step * ave_ZR_new + (1-step) * ave_ZR
#      ave_Z  <- step * ave_Z_new + (1-step) * ave_Z
#      ave_Q  <- step * ave_Q_new + (1-step) * ave_Q
#    }
#
#    c_new <- colSums(X * ave_ZR) / colSums(ave_ZR)
#    g_new <- colSums(X * (1-ave_ZR)) / colSums(1-ave_ZR)
#    c_new <- c_new
#    g_new <- g_new
#
#    c_new[is.nan(c_new)] <- 1
#
#    err <- (sum(abs(g-g_new)) + sum(abs(c-c_new)))/(2*J) # parameter differences
#    cat("==[slamR] Iteration: ",iter,", Err: ",round(err,6),", number of Q-entries changed: ",sum(sum(abs(Q-Q_old))),". ==\n")
#    # after the algorithm has converged, examine the equivalent class of Q.
#  }
#
#  Z_est <- 0+(ave_Z >1/2)
#  Z_candi <- unique_sort_binmat(Z_est)
#
#  Q_arr <- Q_arr[1:iter]
#
#  make_list(Z_est,Z_candi,Q_arr,c,g)
#}


#' Estimate Q, screening latent attributes (Alternating Gibbs)
#'
#' This function implements the Alternating Direction Gibbs EM (ADG-EM)
#' algorithm in the scenario of responses observed over many taxonomies
#' (trees)
#'
#' @param X N by J2 binary data matrix - level 2
#' @param Z_ini N by K initial latent attributes
#' @param Q_ini J by K inital Q matrix
#' @param max_iter maximum iterations (e.g., 50)
#' @param err_prob noise level
#' @param must_maxiter 1 to force maxiter; default is \code{0}
#' @param D_mat J1 by J2 binary matrix to indicate children in two-level trees.
#'  \code{D_mat} is the \code{J1 * J2} binary adjacency matrix specifying how the
#'          trees are grown in the second layer, i.e., which second-level
#'          responses are "children" of each first-level response. Default is \code{NULL}
#' @param X1 N by J1 binary data matrix - level 1; default is \code{NULL}
#' @param model "DINA" (default) or "DINO"
#'
#' @return
#' \itemize{
#' \item Z_est Estimated latent attributes for all people
#' \item Z_candi candidate latent attribute patterns (unique)
#' \item Q_arr a list of Q matrices obtained from the algorithm
#' \item c J dimensional 1-slipping parameter
#' \item g J dimensional guessing parameter
#' }
#' @importFrom matrixStats logSumExp
#' @export
adg_em <- function(X,Z_ini,Q_ini,max_iter,err_prob,must_maxiter=0,
                   D_mat=NULL,X1=NULL,model="DINA"){
  # obtain a binary matrix specifying which subjets are linked to
  # which second-level responses

  # for DINO, do the following transformation:
  if(model=="DINO"){
    X <- 1-X
    Z_ini <- 1-Z_ini
  }

  indmat_im <- NULL
  if (!is.null(D_mat) & !is.null(X1)){indmat_im <- X1%*%D_mat}

  J <- nrow(Q_ini)
  K <- ncol(Q_ini)

  Q <- Q_ini
  range_gen <- (1-2*err_prob)/4 #?
  g_new <- err_prob + (-range_gen/2+range_gen*stats::runif(J))
  c_new <- 1-err_prob + (-range_gen/2+range_gen*stats::runif(J))

  iter <- 0
  err  <- 1
  G    <- 6 # 2*m_burnin, for example.
  m_burnin <- 3

  Z <- Z_ini
  Q_arr <- vector("list",max_iter)

  if (must_maxiter==1){
    iter_proceed <- (iter < max_iter)
  } else{
    iter_proceed <- (err > 1e-3 && iter < max_iter)
  }

  # the following two arrays store probabilities used to sample Q:
  while (iter_proceed){
    g <- g_new
    c <- c_new
    Q_old <- Q

    # E-step: N by J
    Rcg <- sweep(X,2,(log_prac(c)-log_prac(g)),"*")+
      sweep(1-X,2,log_prac(1-c)-log_prac(1-g),"*")
    iter <- iter +1

    ZR <- 0 # ideal response matrix.
    ZZ <- 0
    num <- 0

    # start Gibbs sampling of attribute profiles for G times:
    for (mm in 1:G){
      for (k in 1:K){ #iterate over attributes:
        ZR_k <- get_ideal_resp(Q[,-k,drop=FALSE],
                               Z[,-k,drop=FALSE])
        if(!is.null(indmat_im)) {
          P <- (ZR_k*Rcg*indmat_im)%*%Q[,k,drop=FALSE]
        } else{
          P <- (ZR_k*Rcg)%*%Q[,k,drop=FALSE]#?
        }
        #p_nume <- exp(c(P)) # potential numerical overflow!
        #Z[,k] <- rvbern(p_nume/(p_nume+1))
        Z[,k] <- rvbern(exp(c(P)-
                              c(apply(cbind(c(P),0),1,matrixStats::logSumExp))))
      }
      if (mm > m_burnin | iter >200){
        ZZ <- ZZ+Z
        ZR <- ZR+get_ideal_resp(Q,Z)
        num <- num+1
      }
    }
    # end of Gibbs "sampling" of latent attributes.
    ave_ZR_new <- ZR/num
    ave_Z_new  <- ZZ/num


    # new step for updating Q:
    QQ  <- 0
    num <- 0
    # start Gibbs sampling of Q-matrix for G time:
    for (mm in 1:G){
      for (k in 1:K){
        ZR_kq <- get_ideal_resp(Q[,-k,drop=FALSE],ave_Z_new[,-k,drop=FALSE])
        if(!is.null(indmat_im)) {
          inside_kq <- matrix((1-ave_Z_new[,k]),nrow=1)%*%(ZR_kq*Rcg*indmat_im)
        } else{
          inside_kq <- matrix((1-ave_Z_new[,k]),nrow=1)%*%(ZR_kq*Rcg)
        }
        Q[,k] <- rvbern(exp(0 -
                              c(apply(cbind(c(inside_kq),0),1,matrixStats::logSumExp))))
      }

      if (mm>m_burnin | iter >200){
        QQ <- QQ+Q
        num <- num+1
      }
    }
    # end of Gibb sampling for Q.

    ave_Q_new <- QQ/num
    Q <- 0+(ave_Q_new > 1/2)
    Q_arr[[iter]] <- Q
    # end of updating Q.

    ## new M step:
    if (iter ==1){
      ave_ZR <- ave_ZR_new
      ave_Z  <- ave_Z_new
      ave_Q  <- ave_Q_new
    } else{
      step <- 1/iter
      ave_ZR <- step * ave_ZR_new + (1-step) * ave_ZR
      ave_Z  <- step * ave_Z_new + (1-step) * ave_Z
      ave_Q  <- step * ave_Q_new + (1-step) * ave_Q
    }

    c_new <- colSums(X * ave_ZR) / colSums(ave_ZR)
    g_new <- colSums(X * (1-ave_ZR)) / colSums(1-ave_ZR)
    c_new <- c_new
    g_new <- g_new

    c_new[is.nan(c_new)] <- 1

    err <- (sum(abs(g-g_new)) + sum(abs(c-c_new)))/(2*J) # parameter differences
    cat("==[slamR] Iteration: ",iter,", Err: ",round(err,6),", number of Q-entries changed: ",sum(sum(abs(Q-Q_old))),". ==\n")

    if (must_maxiter==1){
      iter_proceed <- (iter < max_iter)
    } else{
      iter_proceed <- (err > 1e-3 && iter < max_iter)
    }
    # after the algorithm has converged, examine the equivalent class of Q.
  }

  Z_est <- 0+(ave_Z >1/2)
  Z_candi <- unique_sort_binmat(Z_est)

  Q_arr <- Q_arr[1:iter]

  make_list(Z_est,Z_candi,Q_arr,c,g)
}





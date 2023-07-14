# estimation step (penalized likelihood or fractional power formulation with proper
# prior on prevalence parameters)



#' performs shrinkage estimation and use Extended BIC (EBIC) to select the
#' best tuning parameter in the penalized likelihood from an entire solution path
#'
#' @param X N by J data matrix
#' @param Q_est J by K Q after screening by \code{\link{adg_em}} (one-level version)
#' @param Z_candi K-column candidate latent attributes (candidates;
#' \code{\link{adg_em}} (one-level version) )
#' @param lambda_vec a decreasing sequence of negative penalty parameters
#' @param c_ini J initial values of 1-slipping parameters
#' @param g_ini J initial values of guessing parameters
#' @param is_sub use subsample (1) or not (0; default) - NB: not implemented.
#' @param model "DINA" (default) or "DINO"
#'
#' @return
#' \itemize{
#' \item A_final final estimated set of latent attribute patterns that made the cut
#' \item EBIC_vec vector of EBIC values
#' \item size_vec for each tunning parameter, the size of latent attribute patterns
#' that made the cut.
#' }
#' @export

perform_shrink <- function(X, Q_est, Z_candi, lambda_vec, c_ini, g_ini,
                           is_sub=0,model="DINA"){

  # for DINO, do the following transformation:
  if(model=="DINO"){
    X <- 1-X
    Z_candi <- 1-Z_candi
  }

  num_candi <- nrow(Z_candi)
  K <- ncol(Z_candi)
  N <- nrow(X)
  thres <- 0.5/N
  thres_c <- 0.01 # for modified E step in algorithm 1 (PEM)
  err_prob <- 0.2

  tune_len <- length(lambda_vec)
  mat_nu   <- matrix(0,nrow=tune_len,ncol=num_candi) # each pattern needs a Delta.
  EBIC_vec <- rep(0,tune_len)

  nu_ini1 <- rep(1,num_candi)/num_candi # the Delta's in the algorithm 1 - PEM
  nu <- nu_ini1
  c <- c_ini
  g <- g_ini

  Q <- Q_est

  for (i in 1:tune_len){
    if (is_sub==0){
      res <- get_PEM(X,Q,Z_candi,lambda_vec[i],c,g,nu,thres_c) # <--------------------- work.
      nu <- res$nu
      c  <- res$c
      g  <- res$g
      rm(res)
    } else{
      stop("==[slamR] check 'get_pem_subsample.m'; not done yet :-( ==\n")
    }
    mat_nu[i,] <- nu

    A_est <- Z_candi[nu>thres,,drop=FALSE] # get rid of insignificant patterns.
    #res <- get_EM(X,Q,A_est,err_prob,c,g)
    res <- get_EM(X,Q,A_est,err_prob) # <---- not uising initial c,g from PEM.
    loglik_EM <- res$loglik # for calculating EBIC.

    EBIC_vec[i] <- -2*loglik_EM+2^K*log(N)+2*log(nchoosek_prac(2^K,nrow(A_est)))
    cat("==[slamR] PEM: lambda ", lambda_vec[i]," completed <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

  }

  EBIC_vec[EBIC_vec==0] <- max(EBIC_vec)+1000
  Index <- min(which.min(EBIC_vec))
  A_final <- Z_candi[mat_nu[Index,]>thres,,drop=FALSE]
  size_vec <- rowSums(mat_nu > thres) # total number of latent attribute patterns
  # that made the cut for each lambda.
  make_list(A_final,EBIC_vec,size_vec)

}

#' PEM FOR SHRINKAGE - this is to take a candidate latent attribute set
#'
#' @param X N by J data matrix
#' @param Q J by K structural matrix
#' @param A K-column latent attributes; candidates to be selected!
#' @param lambda negative penalty parameter
#' @param c_ini initial 1-slipping parameters
#' @param g_ini initial guessing parameters
#' @param nu_ini initial potentials
#' @param thres_c threshold for making sure M step does not generate negative
#'  probabilities (algorithm 1 of Xu and Gu JMLR 2019)
#' @param model "DINA" (default) or "DINO"
#' @return
#' \itemize{
#' \item nu vector of potential ("Delta")
#' \item c 1-slipping parameter
#' \item g guessing parameter
#' }
#' @export
get_PEM <- function(X,Q,A,lambda,c_ini,g_ini,nu_ini,thres_c,model="DINA"){
  J <- nrow(Q)
  K <- ncol(Q)
  N <- nrow(X)
  prop_resp <- rep(1/N,N)

  c <- c_ini
  g <- g_ini

  err <- 1
  itera <- 0

  n_in <- nrow(A)
  if (model=="DINA"){ideal_resp <- t(get_ideal_resp(Q,A))}
  if (model=="DINO"){ideal_resp <- 1-t(get_ideal_resp(Q,1-A))}

  delta <- nu_ini*n_in

  obj_func <- 0
  obj_vec  <- NULL

  while (abs(err) > 1e-1 & itera <1000){
    old_obj_func <- obj_func
    theta_mat <- sweep(ideal_resp,1,c,"*") + sweep(1-ideal_resp,1,g,"*")

    # prob for each response pattern in X and each attribute profile, under
    # current values of g and c
    # posi_part <- matrix(0,nrow=n_in,ncol=N)
    # nega_part <- matrix(0,nrow=n_in,ncol=N)
    # for (ii in 1:N){ # can improve using log....
    #   posi_part[,ii] <- apply(sweep(t(theta_mat),2,X[ii,],"^"),1,prod)
    #   nega_part[,ii] <- apply(sweep(t(1-theta_mat),2,1-X[ii,],"^"),1,prod)
    # }
    posi_part <- t(bsxfun_7_hinge_pow_prod(t(theta_mat),X))
    nega_part <- t(bsxfun_7_hinge_pow_prod(t(1-theta_mat),1-X))
    # posi_part < apply(bsxfun_7_hinge_pow(t(theta_mat),X),c(1,2),prod)
    # nega_part < apply(bsxfun_7_hinge_pow(t(1-theta_mat),1-X),c(1,2),prod)

    alpha_li0 <- sweep(t(posi_part)*t(nega_part),2,delta,"*")
    alpha_li <- alpha_li0/rowSums(alpha_li0) # N by n_in. phi_i_alpha


    numer <- colSums(N*prop_resp*alpha_li)

    delta <- pmax(thres_c,lambda+numer)

    prob_know_ri <- alpha_li%*%t(ideal_resp)

    # closed-form M step for updating theta plus and theta negative.
    c_denom <- colSums(prob_know_ri)
    c_nume  <- colSums(X*prob_know_ri)

    c[c_denom!=0] <- c_nume[c_denom!=0]/c_denom[c_denom!=0]
    c[c_denom==0] <- 1 # no info.

    g_denom <- colSums(1-prob_know_ri)
    g_nume  <- colSums(X*(1-prob_know_ri))

    g[g_denom!=0] <- g_nume[g_denom!=0]/g_denom[g_denom!=0]
    g[g_denom==0] <- 0 # no info.

    itera <- itera +1

    nu <- delta/sum(delta)
    nu <- nu/sum(nu)

    # NEXT compute the log likelihood at the current iteration

    prod_d2_arr <- t(posi_part*nega_part)

    obj_func <- sum(log_prac(prod_d2_arr %*%matrix(nu,ncol=1)))+lambda*sum(log_prac(nu))
    err <- obj_func - old_obj_func

    obj_vec <- c(obj_vec,obj_func)

    cat("==[slamR] Penalized EM iter (choose A): ", itera, ", size above threshold: ", sum(nu>0.5/N),", Err: ", err, "==\n")
    # <------- be careful about the implicit threshold for significant
    # latent attribute patterns.
  }
  make_list(nu,c,g)
}




#' function to actually perform EM for DINA (fixed A)
#'
#' This is primarily for plugging in final A (e.g., from PEM) and redo EM to calculate log lik
#' for EBIC calculations. Response probabilities estimated within the algorithm
#' with A and Q fixed.
#'
#' @param X N by J data matrix
#' @param Q J by K structural matrix
#' @param A K-column latent attribute matrix
#' @param err_prob error probability; for initializing c g
#' @param c,g default to NULL, but can use values obtained from PEM.
#' @param model "DINA" (default) or "DINO"
#'
#' @return
#' \itemize{
#' \item nu vector of potential ("Delta")
#' \item c 1-slipping parameter
#' \item g guessing parameter
#' \item loglik log likelihood
#' }
#' @export
get_EM <- function(X,Q,A,err_prob,c=NULL,g=NULL,model="DINA"){
  J <- nrow(Q)
  K <- ncol(Q)
  N <- nrow(X)
  err <- 1
  itera <- 0

  n_in <- nrow(A)
  if (model=="DINA"){ideal_resp <- t(get_ideal_resp(Q,A))}
  if (model=="DINO"){ideal_resp <- 1-t(get_ideal_resp(Q,1-A))}

  obj_func <- 0
  obj_vec  <- NULL

  if(is.null(c) | is.null(g)){
    range_gen <- (1-2*err_prob)/8
    g <- err_prob + (-range_gen/2+range_gen*stats::runif(J))
    c <- 1-err_prob + (-range_gen/2+range_gen*stats::runif(J))
  }

  while (abs(err) > 1e-1 & itera <1000){
    old_obj_func <- obj_func
    theta_mat <- sweep(ideal_resp,1,c,"*") + sweep(1-ideal_resp,1,g,"*")

    # probability for each response pattern in X and each attribute profile
    # under current values of g and c:

    # posi_part <- matrix(0,nrow=n_in,ncol=N)
    # nega_part <- matrix(0,nrow=n_in,ncol=N)
    # for (ii in 1:N){ # can improve using log....
    #   posi_part[,ii] <- apply(sweep(t(theta_mat),2,X[ii,],"^"),1,prod)
    #   nega_part[,ii] <- apply(sweep(t(1-theta_mat),2,1-X[ii,],"^"),1,prod)
    # }
    posi_part <- t(bsxfun_7_hinge_pow_prod(t(theta_mat),X))
    nega_part <- t(bsxfun_7_hinge_pow_prod(t(1-theta_mat),1-X))

    alpha_li0 <- t(posi_part)*t(nega_part)
    alpha_li <- alpha_li0/rowSums(alpha_li0) # N by n_in. phi_i_alpha

    numer <- colSums(alpha_li) # get Delta.

    prob_know_ri <- alpha_li%*%t(ideal_resp)

    # closed-form M step for updating theta plus and theta negative.
    c_denom <- colSums(prob_know_ri)
    c_nume  <- colSums(X*prob_know_ri)

    c[c_denom!=0] <- c_nume[c_denom!=0]/c_denom[c_denom!=0]
    c[c_denom==0] <- 1 # no info.

    g_denom <- colSums(1-prob_know_ri)
    g_nume  <- colSums(X*(1-prob_know_ri))

    g[g_denom!=0] <- g_nume[g_denom!=0]/g_denom[g_denom!=0]
    g[g_denom==0] <- 0 # no info.

    itera <- itera +1

    nu <- numer/sum(numer)
    nu <- nu/sum(nu)

    # NEXT compute the log likelihood at the current iteration:

    prod_d2_arr <- t(posi_part*nega_part)

    obj_func <- sum(log_prac(prod_d2_arr %*%matrix(nu,ncol=1)))
    err <- obj_func - old_obj_func

    obj_vec <- c(obj_vec,obj_func)

    cat("==[slamR] Plain EM iter (for EBIC): ", itera, ", size above threshold: ", sum(nu>0.5/N),", Err: ", err, "==\n")
    # <------- be careful about the implicit threshold for significant
    # latent attribute patterns.
  }

  loglik <- obj_func
  make_list(nu,c,g,loglik)
}



#' EM algorithm for estimating c, g and mixing parameter pi
#'
#' @param X N by J data matrix
#' @param Q J by K structural matrix
#' @param A_in C by K latent attributes
#' @param err_prob error probabilities 0.2
#' @param model "DINA" (default) or "DINO"
#' @return
#' \itemize{
#' \item nu
#' \item c
#' \item g
#' \item loglik
#' \item Z_shrink
#' }
#' @export
get_em_classify <- function(X,Q,A_in,err_prob,model="DINA"){
  J <- nrow(Q)
  K <- ncol(Q)

  N <- nrow(X)
  resp_vecs <- X
  prop_resp <- rep(1/N,N)

  N0 <- N

  n_in <- nrow(A_in)
  if (model=="DINA"){ideal_resp <- t(get_ideal_resp(Q,A_in))}
  if (model=="DINO"){ideal_resp <- 1-t(get_ideal_resp(Q,1-A_in))}

  nu <- rep(1/n_in,n_in)

  range_gen <- (1-2*err_prob)/4 # arbitrary choice - supposedly not to matter too much
  g <- err_prob + (-range_gen/2+range_gen*stats::runif(J))
  c <- 1-err_prob + (-range_gen/2+range_gen*stats::runif(J))

  err <- 1
  itera <- 0

  while (err > 1e-4 & itera < 1000){
    # J by C prob matrix for positive responses:
    theta_mat <- sweep(ideal_resp,1,c,"*") + sweep(1-ideal_resp,1,g,"*")

    posi_part <- matrix(0,nrow=n_in,ncol=N)
    nega_part <- matrix(0,nrow=n_in,ncol=N)
    # for (ii in 1:N){ # can improve using log....
    #   posi_part[,ii] <- apply(sweep(t(theta_mat),2,X[ii,],"^"),1,prod)
    #   nega_part[,ii] <- apply(sweep(t(1-theta_mat),2,1-X[ii,],"^"),1,prod)
    # }
    posi_part <- t(bsxfun_7_hinge_pow_prod(t(theta_mat),X))
    nega_part <- t(bsxfun_7_hinge_pow_prod(t(1-theta_mat),1-X))

    alpha_li0 <- t(posi_part)*t(nega_part)
    alpha_li <- alpha_li0/rowSums(alpha_li0) # N by C. phi_i_alpha
    if (any(is.nan(alpha_li))){
      ind_notnan <- which(!is.nan(rowSums(alpha_li)))
      alpha_li <- alpha_li[ind_notnan,,drop=FALSE]
      X <- X[ind_notnan,,drop=FALSE]
      N <- nrow(X); resp_vecs <- X; prop_resp <- rep(1/N,N)
    }

    nu <- colSums(alpha_li*prop_resp)

    # N by J marginal probability for each response pattern:
    prob_know_ri <- alpha_li%*%t(ideal_resp)

    old_c <- c
    old_g <- g

    # closed-form M step for updating theta plus and theta negative.
    c_denom <- colSums(prob_know_ri)
    c_nume  <- colSums(resp_vecs*prob_know_ri)

    c[c_denom!=0] <- c_nume[c_denom!=0]/c_denom[c_denom!=0]
    c[c_denom==0] <- 1 # no info.

    g_denom <- colSums(1-prob_know_ri)
    g_nume  <- colSums(resp_vecs*(1-prob_know_ri))

    g[g_denom!=0] <- g_nume[g_denom!=0]/g_denom[g_denom!=0]
    g[g_denom==0] <- 0 # no info.

    err0 <- abs(old_c-c)+abs(old_g-g)

    ind_item <- (g_denom!=0)*(c_denom!=0)

    err <- max(err0[ind_item])
    if (length(err)<1){
      stop("==[slamR] no response probabilities has information from data==\n")
    }

    itera <- itera+1
    cat("==[slamR] EM iteration: ",itera,", Err: ",err,"==\n")
  }

  # J by n_in:
  theta_mat <- sweep(ideal_resp,1,c,"*") + sweep(1-ideal_resp,1,g,"*")
  alpha_rc  <- sweep(apply(bsxfun_7_pow(theta_mat,resp_vecs)*
                             bsxfun_7_pow(1-theta_mat,1-resp_vecs),c(1,3),prod),
                     2,nu,"*")

  alpha_rc <- sweep(alpha_rc,1,rowSums(alpha_rc),"/")

  nu <- colSums(sweep(alpha_rc,1,prop_resp,"*")) # length n_in vector

  loglik <- N*sum(alpha_rc)

  nu <- nu/sum(nu)

  alpha_mat <- alpha_rc

  Z_shrink <- matrix(0,N0,K)
  idx <- apply(alpha_mat,1,which.max)
  Z_shrink <- A_in[idx,]

  make_list(nu,c,g,loglik,Z_shrink)
}











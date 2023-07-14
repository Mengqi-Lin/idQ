#' Sample a vector of Bernoulli variables.
#'
#' Sample a vector of Bernoulli variables with higher speed
#' (same length with \code{"p"}).
#' The Bernoulli random variables can have different means.
#'
#' @param p A vector of probabilities, each being the head probability
#' of an independent coin toss
#'
#' @return A vector of 1s (head) and 0s (tail)
#' @export
rvbern <-
  function(p) {
    U  <- stats::runif(length(p),0,1)
    res <- (U < p) + 0
    res
  }


#' get ideal response for DINA R
#'
#' replace Z by 1-Z and 1- the resultant matrix gives the
#' ideal response matrix for DINO
#'
#' @param Q Q matrix of size J by K
#' @param Z latent attribute profiles of individuals; N by K
#'
#' @return binary ideal response matrix of size N by J
#'
#' @export
#' @examples
#' Z <- rbind(c(1,1,1),c(0,0,1))
#' Q <- matrix(c(0,0,0, 0,0,1, 0,1,0,
#'               0,1,1, 1,0,0, 1,0,1,
#'               1,1,0, 1,1,1),ncol=3,byrow=TRUE)
#' get_ideal_resp(Q,Z)
#'
#' get_ideal_resp_R(Q=matrix(c(1,1,0,
#'                           0,0,1,
#'                           0,1,0,
#'                           1,0,1),nrow=4,byrow=TRUE),
#'                Z = rbind(c(1,1,1),c(0,0,0)))

get_ideal_resp_R  <- function(Q,Z){
  J <- nrow(Q)
  K <- ncol(Q)
  # ideal response matrix, size N by J:
  xi <- matrix(0,nrow=nrow(Z),ncol=J)
  for (j in 1:J){
    xi[,j] <- apply(sweep(Z,2,Q[j,],"^"),1,prod)
  }
  xi
  # res <- array(0,c(J,nrow(Z),K))
  # aperm(replicate(nrow(Q),Z),c(3,1,2))^
  #   aperm(replicate(nrow(Z),Q),c(1,3,2)) # replicate to the right dimension.
  # apply(res,c(2,1),prod)
} # slow.

# pp <- rep(0,4)
# for (i in 1:1000){
# pp <- pp + rvbern(matrix(c(0.3,0.9,0.1,0.5),nrow=2))
# }
# pp/1000

#' simulate data according to DINA
#'
#' may be redundant; this can be done by \link{generate_X_fromA}
#'
#' @param N sample size
#' @param p 2^K by 1 vector of prevalence probabilities
#' @param Q M by K item-attribute relationship
#' @param c M by 1 one minus slipping parameters
#' @param g M by 1 guessing parameters
#'
#' @return
#' \itemize{
#' \item X N by M matrix of responses
#' \item A N by K matrix of attributes for N individuals
#' }
#'
#'
#' @export
#'
#' @examples
#' N <- 1000
#' #p <- c(1,rep(0,7))
#' p <- c(rep(0,7),1)
#' #p <- c(0.1,0.1,0.1,0.1,0.1,0.2,0.3,0)
#' Q <- matrix(c(0,0,0, 0,0,1, 0,1,0,
#'               0,1,1, 1,0,0, 1,0,1,
#'               1,1,0, 1,1,1),ncol=3,byrow=TRUE)
#'
#' M <- nrow(Q)
#' c <- rep(0.8,M)
#' g <- rep(0.2,M)
#'
#' generate_X_DINA(N,p,Q,c,g)

generate_X_DINA <- function(N,p,Q,c,g){
  M <- nrow(Q)
  K <- ncol(Q)
  counts <- colSums(t(sapply(sample(1:length(p),N,replace=TRUE,prob=p),
                             function(x) {match(1:length(p),x,nomatch=0)})))
  A <- rep(0,N)
  n <- 1
  # A stores empirical CDF for categories: 1, 2. ..., 2^K.
  for (a in 1:length(p)){
    A[n:(n+counts[a]-1)] <- a-1
    n <- n+counts[a]
  }

  #permute A, size N by 1
  A <- A[sample(N)]
  #convert each number in A (ranging from 0 to 2^K-1) to binary form,
  # size N by K; each row of A stores the attribute profile for each individual:
  A <- binary(A,K)

  #ideal response matrix, size N by M:
  xi <- matrix(0,nrow=N,ncol=M)
  for (j in 1:M){
    xi[,j] <- apply(sweep(A,2,Q[j,],"^"),1,prod)
  }

  # conditional probability for each individual correctly
  # answering each question, size N by K:
  p_correct <- t(replicate(N,g))+sweep(xi,2,c-g,"*")

  X <- rvbern(p_correct)
  make_list(X,A)
}



#' generate simulated data A
#'
#' Default to DINA
#'
#' shorter p compared to \code{\link{generate_X_DINA}}
#'
#' @param N sample size
#' @param A K column latent state attributes (unique rows)
#' @param p 2^K by 1 vector of prevalence probabilities
#' @param Q M by K item-attribute relationship
#' @param c M by 1 one minus slipping parameters
#' @param g M by 1 guessing parameters
#' @param model "DINA"(default) or "DINO"
#'
#' @return
#' \itemize{
#' \item X: N by M matrix of responses
#' \item Z: N by K matrix of attributes for N individuals
#' \item X_true: N by M ideal response matrix
#' }
#'
#' @export
#' @examples
#' N <- 1000
#' A <- rbind(c(1,1,1),c(0,0,1))
#' p <- c(0.9,0.1)
#' Q <- matrix(c(0,0,0, 0,0,1, 0,1,0,
#'               0,1,1, 1,0,0, 1,0,1,
#'               1,1,0, 1,1,1),ncol=3,byrow=TRUE)
#' M <- nrow(Q)
#' c <- rep(0.8,M)
#' g <- rep(0.2,M)
#' generate_X_fromA(N,A,p,Q,c,g)
generate_X_fromA <- function(N,A,p,Q,c,g,model="DINA"){
  ind_len <- nrow(A)
  ind_sample <- sample(1:ind_len,N,replace = TRUE,p)
  Z <- A[ind_sample,] # latent attribut profiles for all subjects.
  ## the ideal_resp encodes the ground truth of X:
  if (!is.null(model) & model=="DINA"){
    ideal_resp <- get_ideal_resp(Q,Z)
  }

  if (!is.null(model) & model=="DINO"){ #duality.
    ideal_resp <- 1-get_ideal_resp(Q,1-Z)
  }
  p_correct <-  t(replicate(N,g))+sweep(ideal_resp,2,c-g,"*")
  X <- rvbern(p_correct)
  X_true <- ideal_resp
  make_list(X,Z,X_true)
}

#' generate second-level responses with taxonomy under DINA
#'
#' @param X1_sub The subset of individuals who has one particular level 1 latent attributes
#' @param A_set20 The true leve 2 set of latent attributes
#' @param D_mat J1 by J2 matrix for indicating the tree structure
#' @param p_short level 2 sparse population level prevalences
#' @param Q2 level 2 Q in a subset of level 1 latent attributes
#' @param c 1-slipping parameters
#' @param g guess parameters
#' @param model "DINA" (default); "DINO"
#'
#' @return
#' \itemize{
#'
#' \item X2 level 2 observed responses
#' \item Z latent attribute profiles for all individuals at level 2
#' \item X_true2 Ideal response matrix
#' }
#'
#' @export
#'
generate_X_tax2 <- function(X1_sub,A_set20,D_mat,p_short,
                            Q2,c,g,model="DINA"){
  N_sub <- nrow(X1_sub)
  indmat_im <- X1_sub %*% D_mat

  ind_len <- nrow(A_set20) # total number of level 1 responses.
  ind_sample <- sample(1:ind_len,N_sub,replace = TRUE,p_short)
  Z <- A_set20[ind_sample,] # latent attribute profiles of all subjects.

  #the ideal_resp encodes the "ground truth" of X:
  if (!is.null(model) & model=="DINA"){
    ideal_resp <- get_ideal_resp(Q2,Z)
  }
  if (!is.null(model) & model=="DINO"){
    ideal_resp <- 1-get_ideal_resp(Q2,1-Z)
  }
  p_correct <- t(replicate(N_sub,g))+sweep(ideal_resp,2,c-g,"*")
  X2_temp <- rvbern(p_correct)
  X2 <- X2_temp*indmat_im
  X_true2 <- ideal_resp*indmat_im

  #par(mfrow=c(1,2))
  #image(X2_temp,main=paste0("2nd-level orig.: ",mean(X2_temp)))
  #image(X2,main=paste0("2nd level observ.: ",mean(X2)))

  make_list(X2,Z,X_true2)
}











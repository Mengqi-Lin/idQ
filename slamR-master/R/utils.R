# helper functions


#' flip the matrix to the right form for \code{image()}
#'
#' This function makes the result of \code{image()} to be plotted as shown in a matrix
#'
#' @param m a matrix
#' @return a matrix after image() will look the same as the matrix
#' @export
f <- function(m) {t(m)[,nrow(m):1]}

#' Takes any number of R objects as arguments and returns a list whose names are
#' derived from the names of the R objects.
#'
#' Roger Peng's listlabeling challenge from
#' \url{http://simplystatistics.tumblr.com/post/11988685443/computing-on-the-language}.
#' Code copied from \url{https://gist.github.com/ajdamico/1329117/0134148987859856fcecbe4446cfd37e500e4272}
#'
#' @param ... any R objects
#'
#' @return a list as described above
#'
#' @examples
#' #create three example variables for a list
#' x <- 1
#' y <- 2
#' z <- "hello"
#' #display the results
#' make_list( x , y , z )
#' @export
make_list <- function(...) {
  #put all values into a list
  argument_values <- list(...)

  #save all argument names into another list
  argument_names <- as.list(sys.call())

  #cycle through the first list and label with the second, ignoring the function itself
  for (i in 2:length(argument_names)) {
    names(argument_values)[i - 1] <- argument_names[i]
  }
  #return the newly-labeled function
  argument_values
}

#' convert binary codes to decimal integers
#'
#' @param mat A binary matrix; Each row is a binary representation to be converted.
#' @param LOG Take the logarithm of the decimal integer (default is \code{TRUE});
#' otherwise, set to \code{FALSE}.
#'
#' @return A vector of numbers after conversion (not a matrix)
#' @export
#'
#' @examples
#' bin2ind(c(1,1,1))
#' bin2ind(rbind(c(1,0,0),c(1,1,0),c(0,0,1),c(1,0,1),c(0,0,0)))
#' bin2ind(rbind(c(1,0,0),c(1,1,0),c(0,0,1),c(1,0,1),c(0,0,0)),LOG=FALSE)
bin2ind <- function(mat,LOG=FALSE){# This is the log version:
  if(!is.matrix(mat)){mat <- matrix(c(mat),nrow=1)} # convert to a matrix.
  L <- ncol(mat)
  M <- nrow(mat)
  v <- log(as.matrix(2^{(L-1):0},ncol=1))
  diag_v <- matrix(0,nrow=L,ncol=L)
  diag(diag_v) <- v
  tmp_prod <- mat%*%diag_v
  permute_M_vec <- sapply(1:M,function(i){matrixStats::logSumExp(tmp_prod[i,mat[i,]!=0])})
  if (!LOG){
    permute_M_vec <- exp(permute_M_vec)
  }
  #permute_M_vec[rev(order(permute_M_vec))]
  permute_M_vec
}

#' convert decimal integers to binary codes
#'
#' @param x a vector of base 10 integers
#' @param k length of binary codes (if no specified, this will be the
#' length of the binary representation for the maximum in \code{x}).
#' Default to \code{NULL}.
#' @return A matrix of binary forms of x (by row)
#' @export
#'
#' @examples
#' decimals <- c(10,3,5)
#' binary(decimals)
#' binary(decimals,6)
binary <- function(x,k=NULL){
  if(!is.null(k)){ # if specified length for binary codes.
    base <- 2
    kmax <- max(floor(log2(max(x)))+1,1)
    if (k < kmax) {cat ("==[slamR] 'k is too small,
        ignoring the specification and setting it to: '", kmax, "! Or
                                     specify k >=", kmax," ==");
      k <- kmax
    }
  } else{ # if length is not specified.
    base <- 2
    k <- max(floor(log2(x))+1,1)
  }
  divs <- t(floor(sapply(x,function(s){s/base^(seq(k-1,0,by=-1))})))
  divs - cbind(rep(0,length(x)), base*divs[,1:k-1,drop=FALSE])
}

#' check if matrix Q is complete
#'
#' Check if there is an item solely requiring attribute k
#' for each attribute k = 1, 2, ..., K
#'
#' @param Q binary matrix
#'
#' @return a list of information
#' \itemize{
#' \item is_complete: 0 for not complete
#' \item ind_I_K: the single item index for each attribute;
#' if an attribute has multiple single-attribute items, just pick the one with
#' the lowest item index.
#' \item sum_single: number of items requiring each attribute
#' }
#'
#'
#' @examples
#' check_complete(matrix(c(1,0,0, 0, 1,0, 0, 0, 1, 1, 1, 0, 0, 0,1),
#' ncol=3,byrow=TRUE))
#' @export
check_complete <- function(Q){
  J <- nrow(Q)
  K <- ncol(Q)
  is_k_single <- rep(0,K)
  ind_I_K <- rep(0,K)
  sum_single <- rep(0,K)
  for (k in 1:K){
    ee <- rep(0,K)
    ee[k] <- 1 # one-hot representation.

    # absolute difference between rows of Q and ee_k:
    absdiff <- abs(Q-t(replicate(J,ee)))

    # find the indices of the single-attribute items requring k:
    find_single <- which(rowSums(absdiff) == 0)


    sum_single[k] <- length(find_single) # number of items just requring k.

    is_k_single[k] <- 0+(length(find_single)>0)

    if (is_k_single[k]>0){
      ind_I_K[k] <- find_single[1]
    }
  }
  is_complete <- prod(is_k_single)
  make_list(is_complete,ind_I_K,sum_single)
}


#' check if patterns in A1 match those in A2
#'
#' @param A1 a binary matrix
#' @param A2 a binary matrix with the same dimensions as \code{A1}
#'
#'
#' @return a list of three variables
#' \itemize{
#' \item among1 a binary vector of each row of A1 identical to some row(s) of A2
#' \item among2 same as among1 but switch A1 and A2.
#' \item overlap a binary vector
#' }
#' @examples
#'
#' check_overlap(matrix(c(1,1,0, 1,0,0),nrow=2,byrow=TRUE),
#' matrix(c(1,0,1, 1,1,0),nrow=2,byrow=TRUE))
#' @export
#'
#'
check_overlap <- function(A1,A2){
  among1 <- rep(0,nrow(A1))
  among2 <- rep(0,nrow(A2))
  for (i in 1:nrow(A1)){
    among1[i] <- bin2ind(A1[i,])%in%bin2ind(A2)
  }
  for (i in 1:nrow(A2)){
    among2[i] <- bin2ind(A2[i,])%in%bin2ind(A1)
  }
  overlap <- among1*among2
  make_list(among1, among2, overlap)
}

#' log with log(0) being 0
#'
#' @param x a nonnegative value
#'
#' @export
log_prac <- Vectorize(function(x){
  if (x>0){ res <- log(x)
  } else {
    res <- 0
  }
  res
})


#' n choose k - dealing with large values.
#'
#' @param n,k integers (n>=k)
#'
#' @export
nchoosek_prac <- function(n,k){
  exp(sum(log(((n-k+1):n)))-sum(log(1:k)))
}


#' get none, singleton, doubleton, ... until full binary patterns
#'
#' Within each k-ton, we order from the largest to smallest
#'
#' @param D sum of binary codes (if zero, forced to be 1)
#' @param M dimension of binary codes
#'
#' @return a binary matrix of size (M + M choose 2 + ...+ M choose D)by M
#' @export
#' @examples
#' get_I(3,3)
#' get_I(2,3)
#' get_I(1,3)
#' get_I(0,3)
get_I <- function(D,M){
  index0 <- 2^seq(M-1,0,by=-1) # length M.
  indices <- index0
  if (D>=2){
    for (d in 2:D){
      newindices <- outer(index0[d:M],indices,"+") # (M-d+1) by M
      indices <- c(indices,newindices)
      ordertemp <- sapply(sort(unique(indices)),
                          function(s) {min(match(s,indices))}) # first match.
      indices <- indices[sort(ordertemp)]
    }
  }
  binary(indices,M)
}



#' calculates marginal probabilities of all the 2^J possible
#' response patterns
#'
#' @param A  K-column, latent attribute profile matrix
#' @param Q  M by K Q matrix
#' @param c  M by 1: 1-slipping
#' @param g  M by 1 guessing parameter
#' @param I_full J-column matrix; Each row is one of 2^J patterns under consideration
#' can be generated by \code{rbind(rep(0,J),get_I(J,J))}
#'
#' @export
#'
#' @examples
#' A <- rbind(c(1,1,1),c(0,0,1))
#' Q <- rbind(c(1,0,0),c(1,0,0),c(0,1,0),c(1,1,1))
#' c <- c(0.8,0.8,0.8,0.8)
#' g <- 1-c(0.8,0.8,0.8,0.8)
#' I_full <- rbind(rep(0,4), get_I(4,4))
#' get_rp(A,Q,c,g,I_full)
get_rp <- function(A,Q,c,g,I_full){
  M <- nrow(Q)
  K <- ncol(Q)
  ideal_resp <- get_ideal_resp(Q,A) # |A| by M
  # M by |A| prob of positive response for the capable
  prob_posi <- t(sweep(sweep(ideal_resp,2,c-g,"*"),2,g,"+"))
  # M by |A| prob of negative reponse for the capable
  prob_zero <- 1-prob_posi
  Rp <- matrix(0,nrow=2^M,ncol=nrow(A))
  for (i in 1:nrow(I_full)){ # for every pattern:
    Rp[i,] <- apply(prob_zero[which(I_full[i,]==0),,drop=FALSE],2,prod)*
      apply(prob_posi[which(I_full[i,]==1),,drop=FALSE],2,prod)
  }
  Rp
}

#' order binary matrix by row
#'
#' @param mat binary matrix
#' @param ... arguments in \code{order}
#'
#' @return binary matrix (ordered by row)
#' @export
#' @examples
#' sort_binmat(matrix(c(1, 1, 1, 0 , 1, 0, 1, 1,0),nrow=3,byrow=TRUE))
sort_binmat <- function(mat,...){
  mat[order(bin2ind(mat),...),,drop=FALSE]
}


#' unique and ordered binary matrix by row
#'
#' @param mat binary matrix
#' @param ...  arguments in \code{order}
#'
#' @return binary matrix (ordered by row)
#' @export
#' @examples
#' unique_sort_binmat(matrix(c(1, 1, 1, 1,1,1, 0 , 1, 0, 1, 1,0),nrow=4,byrow=TRUE))
unique_sort_binmat <- function(mat,...){
  unique(mat[order(bin2ind(mat),...),,drop=FALSE])
}

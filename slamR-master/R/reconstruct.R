#' reconstructs N by J data matrix using the estimated item parameters c, g
#' and other quantities
#'
#' To reconstruct without c or g, use \code{get_ideal_resp(Q_est,Z_est)}
#'
#' @param Z_est N by K of estimated attributes
#' @param Q_est J by K of estimated Q
#' @param c J vector of 1-slipping parameters
#' @param g J vector of guessing parameters
#' @param model "DINA" (default), or "DINO"
#'
#' @seealso \code{\link{get_ideal_resp}}
#'
#' @export
#' @return N by J matrix; use p_correct >0.5 as reconstruction
#'
#' @examples
#' set.seed(60)
#' Z_est <- matrix((rnorm(200)>0)+0,nrow=50,ncol=4)
#' Q_est <- matrix(c(1,1,1,1,0,0,1,0,1,0,1,0,1,1,1,0,0,0,0,1,1,0,0,1),ncol=4,
#' byrow=TRUE)
#' c <- rep(0.8,nrow(Q_est))
#' g <- 1-c
#' reconstruct(Z_est,Q_est,c,g)
reconstruct <- function(Z_est,Q_est,c,g,model="DINA"){
  if (!is.null(model) & model=="DINA"){
    ideal_resp <- get_ideal_resp(Q_est,Z_est)
  }
  if (!is.null(model) & model=="DINO"){
    ideal_resp <- 1-get_ideal_resp(Q_est,1-Z_est)
  }
  p_correct <- t(replicate(nrow(Z_est),g))+sweep(ideal_resp,2,c-g,"*")
  0+(p_correct > 1/2)
}

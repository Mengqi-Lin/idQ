#' slamR: \strong{s}tructured \strong{l}atent \strong{a}ttribute
#'\strong{m}odels in\strong{R}
#'
#'
#' \code{slamR} implements scalable algorithms to fit structured latent attribute
#' models (SLAM) for high-dimensional binary data observed over a single or multiple
#'  levels of specificity. It works for
#' \itemize{
#' \item 1) unknown structural matrix Q,
#' \item 2) unknown latent
#' attribute set,
#' \item 3) one ore more levels of binary data (observed over multiple binary
#' or non-binary trees)
#' \item 4) two-parameter or multi-parameter SLAM models
#' (NB: multi-parameter model under development)
#' }
#'
#' @seealso
#' \itemize{
#' \item \url{https://github.com/zhenkewu/slamR} for the source code
#' and system/software requirements to use \code{slamR} for your data.
#' }
#'
#' @example
#' inst/example/compare_flat.R
#'
#' @useDynLib slamR
#' @importFrom Rcpp sourceCpp
#' @docType package
#' @name slamR
#' @references
#' \itemize{
#' \item This package partly adapts Matlab programs originally written by \href{https://sites.google.com/umich.edu/yuqigu/home}{Yuqi Gu}. Please
#' refer to \url{https://github.com/zhenkewu/slamR} for the papers about the model
#' formulation and algorithm
#' }
NULL
#> NULL


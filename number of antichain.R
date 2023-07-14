# Number of antichains of length k on power set 2^n. A(n, k).

CountAntichains <- function(n) {
  # compute A(n, 1)
  a1 <- 2^n
  
  # compute A(n, 2)
  a2 <- (1/2)*4^n - 3^n + (1/2)*2^n
  
  # compute A(n, 3)
  a3 <- (1/6)*8^n - 6^n + 5^n + (1/2)*4^n - 3^n + (1/3)*2^n
  
  # compute A(n, 4)
  a4 <- (1/24)*16^n - (1/2)*12^n + 10^n + (1/6)*9^n - (3/4)*8^n + (1/4)*7^n - (3/2)*6^n + (3/2)*5^n + (11/24)*4^n - (11/12)*3^n + (1/4)*2^n
  
  return(c(a1, a2, a3, a4))
}
sum(CountAntichains(4))


secondlargest <- function(n) {
  choose(n, floor(n/2)) - floor(n/2)
}
plot(secondlargest

     secondlargest(5)
     
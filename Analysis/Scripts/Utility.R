
psychometric = function(x,df){
  df$lapse + (1-2*df$lapse)*(0.5+0.5*erf((x-df$alpha)/(sqrt(2)*df$beta)))
}

entropy = function(p){
  -p * log(p) - (1-p) * log(1-p)
}

RT_mean = function(p,df){

  entropy_t = entropy(p)
  df$rt_int + entropy_t * df$rt_slope

}

conf_mean = function(p,ACC,df){

  entropy_t = entropy(p)
  df$conf_int + entropy_t * df$conf_slope + df$conf_ACC * ACC + df$conf_slope_ACC * ACC * entropy_t

}

is_pos_def <- function(r12, r13, r23) {
  Sigma <- matrix(c(
    1,    r12, r13,
    r12,  1,   r23,
    r13,  r23, 1
  ), 3, 3)
  all(eigen(Sigma, only.values = TRUE)$values > 0)
}




get_copula_vec = function(df,n){

  set.seed(123)

  Sigma <- matrix(c(
    1,       df$rho12, df$rho13,
    df$rho12,  1,      df$rho23,
    df$rho13,   df$rho23, 1
  ), ncol = 3)

  df = data.frame(mnormt::rmnorm(n = n, mean = c(0,0,0),
                                 varcov = Sigma)) %>% rename(Bin = X1,
                                                             RT = X2,
                                                             VAS = X3)

  us_1 = pnorm(df$Bin)
  us_2 = pnorm(df$RT)
  us_3 = pnorm(df$VAS)

  return(data.frame(u_bin = us_1,
                    u_rt = us_2,
                    u_vas = us_3))
}




qordbeta <- function(p, mu, phi, cutzero, cutone) {
  set.seed(123)

  # ensure p and mu are same length
  n <- max(length(p), length(mu))
  p  <- rep(p,  length.out = n)
  mu <- rep(mu, length.out = n)

  # Beta parameters
  alpha <- mu * phi
  beta  <- (1 - mu) * phi

  # mixture weights (logistic cutpoints)
  p0    <- plogis(cutzero)            # mass at 0
  p1    <- 1 - plogis(cutone)         # mass at 1
  p_mid <- 1 - p0 - p1                # mass in (0,1)

  # initialize output
  y <- numeric(n)

  # regions
  y[p < p0] <- 0
  y[p > (1 - p1)] <- 1

  # continuous region: rescale p to (0,1)
  idx <- p >= p0 & p <= (1 - p1)
  p_rescaled <- (p[idx] - p0) / p_mid
  y[idx] <- qbeta(p_rescaled, alpha[idx], beta[idx])

  return(y)
}

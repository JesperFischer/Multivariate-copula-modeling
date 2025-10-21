
qord_logit <- function(u, cutpoints) {
  cum_p <- plogis(cutpoints)

  y <- numeric(length(u))

  for (i in seq_along(u)) {
    # Find which category this u belongs to
    category <- 1
    for (k in seq_along(cum_p)) {
      if (u[i] >= cum_p[k]) {
        category <- k + 1
      } else {
        break
      }
    }
    y[i] <- category
  }

  return(y)
}






sim = function(N,K){

  means = seq(-3,3,length.out = K-1)
  sds = rep(K/K^2,K-1)
  repeat {
    cutpoints <- rnorm(K - 1, mean = means, sd = sds)
    if (is.unsorted(cutpoints, strictly = TRUE))next
    break
  }


  rho = ggdist::rlkjcorr_marginal(1,2,12)

  df = data.frame(mnormt::rmnorm(n = N, mean = c(0,0),
                                 varcov = cbind(c(1,rho),c(rho,1)))) %>% rename(Bin = X1, RT = X2)

  uni_bin = pnorm(df$Bin)
  uni_RT = pnorm(df$RT)

  mean_RT = rnorm(1,2,2)
  sd_RT = abs(rnorm(1,5,1))

  RT = qnorm(uni_RT,mean_RT,sd_RT)
  Bin = qord_logit(uni_bin,cutpoints = cutpoints)



  list(
    variables = list(
      cutpoints = cutpoints,
      mean_RT = mean_RT,
      sd_RT = sd_RT,
      rho = rho
    ),
    generated = list(
      N = N,
      binom_y = Bin,
      Y_con = RT,
      K = K
    )
  )

}




fit = function(N,K){

  data = sim(N,K)

  mod = cmdstan_model(here::here("tests","tester.stan"))

  fit = mod$sample(data = data$generated,
                             iter_warmup = 1000,
                             iter_sampling = 1000,
                             chains = 4,
                             refresh = 0,
                             max_treedepth = 10,
                             parallel_chains = 4,
                             adapt_delta = 0.95)

  divs = fit$diagnostic_summary()

  df = fit$summary(names(data$variables)) %>% mutate(simulated = as.numeric(unlist(data$variables))) %>%
    mutate(div = max(divs$num_divergent),
           tree = max(divs$num_max_treedepth),
           N = N,
           K = K)


  return(df)

}

fit(50,10)

packages = c("brms","tidyverse","bayesplot","pracma","here",
             "patchwork","posterior","HDInterval","loo", "furrr", "SBC","future")

do.call(pacman::p_load, as.list(packages))


source(here::here("Simulations","Learning","model_recovery_estimation","sim_learn.R"))

cores = 20
n_sim = 1000
plan(multisession, workers = cores)

qq = fitter_ddm(500)

possfit_model = possibly(.f = fitter_ddm, otherwise = "Error")

results <- future_map(rep(1000,n_sim), ~possfit_model(.x), .options = furrr_options(seed = TRUE), .progress = T)

save.image(here::here("Simulations","Learning","model_recovery_estimation","results","1000_rw_fitter_ddm_1000_samples1.RData"))




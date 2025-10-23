
packages = c("brms","tidyverse","bayesplot","pracma","here","cmdstanr",
             "patchwork","posterior","HDInterval","loo", "furrr", "SBC","future")

do.call(pacman::p_load, as.list(packages))


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

cores = 10
n_sim = 100
plan(multisession, workers = cores)

trials_vec <- seq(50, 300, by = 50)
K_vec <- 3:10

# Create all combinations of trials and K
param_grid <- expand.grid(trials = trials_vec,
                          K = K_vec,
                          sim = 1:n_sim)


possfit_model = possibly(.f = fit, otherwise = "Error")

results <- future_map2(
  param_grid$trials, param_grid$K,
  ~ possfit_model(.x, .y),
  .options = furrr_options(seed = TRUE),
  .progress = T
)


save.image(here::here("SBC_ordered_conf","results.RData"))


bind_rows(results) %>% ggplot(aes(x = simulated, y = mean, ymin = q5, ymax = q95))+geom_pointrange()+facet_grid(K~N, scales = "free")


n_sims <- 1000 # Number of SBC iterations to run

generator <- SBC_generator_function(sim, N = 100, K = 5)

dataset <- generate_datasets(
  generator,
  n_sims)

backend <- SBC_backend_cmdstan_sample(
  cmdstan_model(here::here("tests","tester.stan")),
  iter_warmup = 1000,
  iter_sampling = 2000,
  chains = 4,
  max_treedepth = 10,
  parallel_chains = 4,
  adapt_delta = 0.95)


library(future)
cores = 10
plan(multisession, workers = cores)

results <- compute_SBC(dataset,
                       ensure_num_ranks_divisor = 4,
                       keep_fits = F,
                       thin_ranks = 20,
                       #cores_per_fit = 10,
                       backend)
save.image(here::here("SBC_ordered_conf","SBC_results_k5.RData"))



n_sims <- 1000 # Number of SBC iterations to run

generator <- SBC_generator_function(sim, N = 100, K = 10)

dataset <- generate_datasets(
  generator,
  n_sims)

backend <- SBC_backend_cmdstan_sample(
  cmdstan_model(here::here("tests","tester.stan")),
  iter_warmup = 1000,
  iter_sampling = 2000,
  chains = 4,
  max_treedepth = 10,
  parallel_chains = 4,
  adapt_delta = 0.95)


library(future)
cores = 10
plan(multisession, workers = cores)

results <- compute_SBC(dataset,
                       ensure_num_ranks_divisor = 4,
                       keep_fits = F,
                       thin_ranks = 20,
                       #cores_per_fit = 10,
                       backend)
save.image(here::here("SBC_ordered_conf","SBC_results_k10.RData"))




# remove divergence and max tree depth:
sim_ids_to_keep <- results$backend_diagnostics %>%
  dplyr::filter(n_divergent == 0 & n_max_treedepth == 0) %>%
  dplyr::pull(sim_id)


sim_ids_to_exclude <- results$backend_diagnostics %>%
  dplyr::filter(n_divergent != 0 | n_max_treedepth != 0) %>%
  dplyr::pull(sim_id)


results_subset <- results[sim_ids_to_keep]
results_excluded <- results[sim_ids_to_exclude]

# also less than 400 effective samples

sim_ids_to_keep <- results_subset$default_diagnostics %>%
  dplyr::filter(min_ess_bulk  > 400)%>%
  dplyr::filter(min_ess_tail  > 400) %>%
  dplyr::pull(sim_id)


results_subset <- results_subset[sim_ids_to_keep]


plot_rank_hist(results_subset)+facet_wrap(~variable, nrow = 2)+theme_minimal()


plot_sim_estimated(results_subset, alpha = 0.5)+
  facet_wrap(~variable, nrow = 1, scales = "free")+
  theme_minimal()+
  theme(strip.text = element_blank())



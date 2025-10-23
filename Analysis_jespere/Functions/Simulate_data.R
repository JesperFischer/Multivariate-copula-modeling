
simulate_data = function(S = 20, N = 100){

  # Extract posterior means
  gm_draws <- fit$draws("gm") %>% as_draws_matrix() %>% colMeans()
  tau_draws <- fit$draws("tau_u") %>% as_draws_matrix() %>% colMeans()
  cor_draws <- fit$draws("correlation_matrix") %>% as_draws_matrix() %>% colMeans()

  cor_mat = matrix(ncol = 6, nrow = 6)
  k = 0
  for(i in 1:6){
    for(j in 1:6){
      k = k+1
      cor_mat[i,j] = cor_draws[k]
    }
  }


  # Scale by tau
  Sigma <- diag(tau_draws) %*% cor_mat %*% diag(tau_draws)  # covariance matrix

  # Simulate P parameters for S subjects
  indi_params <- MASS::mvrnorm(n = S, mu = gm_draws, Sigma = Sigma)

  colnames(indi_params) <- c("alpha","beta","lapse","rt_int","rt_slope","rt_prec")

  copulacor = data.frame(cop_cor = fit$draws("rho_p_rt")%>% as_draws_matrix() %>% colMeans()) %>% summarize(mean = mean(cop_cor),
                                                                                                            sd = sd(cop_cor),
                                                                                                            q5 = quantile(cop_cor,0.05),
                                                                                                            q95 = quantile(cop_cor,0.95))


  indi_params = as.data.frame(indi_params) %>% mutate(beta = exp(beta),
                                                      lapse = brms::inv_logit_scaled(lapse) / 2,
                                                      rt_prec = exp(rt_prec)) %>%
    mutate(rt_ndt = rnorm(n(),0.35,0.05),
           cop_cor = rnorm(n(),copulacor$mean,copulacor$sd)) %>% mutate(subject = 1:n())





  generate_trials = function(params,N){

    x = rnorm(N,params$alpha,params$beta*5)

    p = psychometric(x,params)
    rt_mu = RT_mean(p,params)

    us = get_copula_vec(params, length(x))

    bin = qbinom(us$u_bin,1,p)


    rts = qlnorm(us$u_rt,rt_mu, params$rt_prec) + params$rt_ndt


    predictions = data.frame(bin = bin, rts = rts,x = x, prob = p, rt_mu = exp(rt_mu) + params$rt_ndt)

  }


  df = indi_params %>% rowwise() %>% mutate(resps = list(generate_trials(cur_data(), N = N)))

  df %>%
    unnest(resps) %>%
    mutate(bin = ifelse(bin == 0, 0.5, 1)) %>%
    pivot_longer(cols = c("bin","rts"), names_to = "name", values_to = "value") %>%
    pivot_longer(cols = c("prob","rt_mu"), names_to = "means", values_to = "mean_value") %>%
    mutate(means = ifelse(means == "prob","bin","rts")) %>%
    filter(name == means) %>%
    ggplot() +
    geom_point(aes(x = x, y = value), alpha = 0.6) +
    geom_line(aes(x = x, y = mean_value), col = "red")+
    facet_grid(name ~ subject, scales = "free") +   # <-- rows = response type, cols = subjects
    theme_classic(base_size = 16) +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(size = 10),
      axis.text = element_text(size = 8)
    ) +
    labs(x = "Stimulus (x)", y = "Response")

  }


simulate_data()



psychometric = function(x,df){
  0.5+0.5*(1-2*df$lapse)*brms::inv_logit_scaled(df$beta*(x-df$alpha))

}

entropy = function(p){
  -p * log(p) - (1-p) * log(1-p)
}

RT_mean = function(p,df){

  entropy_t = entropy(p)
  df$rt_int + entropy_t * df$rt_slope

}



get_copula_vec = function(df,n){

  set.seed(123)

  Sigma <- matrix(c(
    1,       df$cop_cor,
    df$cop_cor,  1
  ), ncol = 2)

  df = data.frame(mnormt::rmnorm(n = n, mean = c(0,0),
                                 varcov = Sigma)) %>% rename(Bin = X1,
                                                             RT = X2)

  us_1 = pnorm(df$Bin)
  us_2 = pnorm(df$RT)
  return(data.frame(u_bin = us_1,
                    u_rt = us_2))
}


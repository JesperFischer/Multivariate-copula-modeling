
fit_data_copula_rt = function(df,ACC, outputname){


  df$subject = as.numeric(as.factor(df$subject))

  t_p_s = df %>% group_by(subject) %>% summarize(n = n())


  ends <- cumsum(t_p_s$n)

  # Calculate the start points
  starts <- c(1, head(ends, -1) + 1)

  if(ACC){
  mod = cmdstan_model(here::here("Stanmodels","Bin_RT_ACC.stan"))
  }else{
  mod = cmdstan_model(here::here("Stanmodels","Bin_RT_Standard.stan"))
  }

  datastan = list(N = nrow(df),
                  S = length(unique(df$subject)),
                  starts = starts,
                  minRT = df %>% group_by(subject) %>% summarize(minRT = min(RT)) %>% .$minRT,
                  ends = ends,
                  t_p_s = t_p_s$n,
                  X = df$X,
                  S_id = df$subject,
                  RT = df$RT,
                  binom_y = df$Y)


  cor <-mod$sample(
    data = datastan,
    refresh = 10,
    iter_sampling = 500,
    iter_warmup = 500,
    adapt_delta = 0.95,
    max_treedepth = 12,
    init  = 0,
    parallel_chains = 4)


  cor$save_object(here::here("Saved models",paste0(outputname,".rds")))

  return(cor)

}

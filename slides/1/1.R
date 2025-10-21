pacman::p_load(tidyverse, gganimate)
n = 10000
mu = 5
sd = 1
fig1 = function(n,mu,sd){
  
  x = seq(0,10,length.out = n)
  dens = dnorm(x,mu,sd)
  dens_df <- data.frame(x = x, dens = dens)
  
  # Generate random draws

  
  # Create reps vector
  reps <- c(seq(1, 101, by = 10), seq(201, n, by = 100))
  draw_ids <- seq_along(reps)
  
  # Generate one long vector of random draws
  max_n <- max(reps)
  base_draws <- rnorm(max_n, mu, sd)
  
  set.seed(123)
  # Generate one long vector of random draws (the maximum number needed)
  max_n <- max(reps)
  base_draws <- rnorm(max_n, mu, sd)
  
  # Build dataframe where each draw_id contains the first `reps[i]` draws
  draws_df <- tibble(draw_id = draw_ids, reps = reps) %>%
    rowwise() %>%
    mutate(xr = list(base_draws[1:reps])) %>%
    unnest(xr)
  
  
  # Make the plot
  p <- ggplot() +
    geom_histogram(data = draws_df, aes(x = xr, y = after_stat(density)),
                   bins = 50, fill = "lightblue", color = "black") +
    geom_line(data = dens_df, aes(x = x, y = dens), col = "red", size = 1) +
    scale_y_continuous(limits = c(0,1))+
    theme_minimal()+
    transition_states(draw_id, transition_length = 1, state_length = 0.1) +
    ease_aes('exponential-in')

  p
  }



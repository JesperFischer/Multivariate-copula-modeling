

Group_plot = function(df, bin = F){
  if(bin == F){
    plot = bind_rows(
      df %>%
        group_by(X) %>%
        summarize(name = "Type-1",
                  mean = mean(Y),
                  se = mean(Y)*(1-mean(Y)) / sqrt(n())),

      df %>%
        group_by(X) %>%
        summarize(name = "RT",
                  mean = mean(RT),
                  se = sd(RT) / sqrt(n())),

      df %>%
        group_by(X, Correct) %>%   # <-- group by correctness only here
        summarize(name = "Confidence",
                  mean = mean(Confidence),
                  se = sd(Confidence) / sqrt(n()))
    ) %>%
      ggplot(aes(x = X, y = mean, ymin = mean - 2*se, ymax = mean + 2*se)) +
      geom_pointrange(aes(color = as.factor(Correct))) +   # only affects responseConf
      facet_wrap(~name, scales = "free", ncol = 1) +
      theme_classic(base_size = 16) +
      labs(color = "Correct")
  }else{
    plot = bind_rows(
      df %>%
        mutate(X_bin = cut(X, bin)) %>%
        group_by(X_bin) %>%
        summarize(name = "Type-1",
                  mean = mean(Y),
                  se = mean(Y)*(1-mean(Y)) / sqrt(n())),

      df %>%
        mutate(X_bin = cut(X, bin)) %>%
        group_by(X_bin) %>%
        summarize(name = "RT",
                  mean = mean(RT),
                  se = sd(RT) / sqrt(n())),

      df %>%
        mutate(X_bin = cut(X, bin)) %>%
        group_by(X_bin,Correct) %>%
        summarize(name = "Confidence",
                  mean = mean(Confidence),
                  se = sd(Confidence) / sqrt(n()))
    ) %>%
      ggplot(aes(x = X_bin, y = mean, ymin = mean - 2*se, ymax = mean + 2*se)) +
      geom_pointrange(aes(color = as.factor(Correct))) +   # only affects responseConf
      facet_wrap(~name, scales = "free", ncol = 1) +
      theme_classic(base_size = 16) +
      labs(color = "Correct")
  }

  return(plot)

}



plot_subjects <- function(df, subject_ids, bin = F) {

  if(bin == F){

    df_summary <- bind_rows(
      # accuracy
      df %>%
        group_by(X, subject) %>%
        summarize(name = "correct",
                  mean = mean(Y),
                  se = sd(Y) / sqrt(n()),
                  .groups = "drop"),

      # RT
      df %>%
        group_by(X, subject) %>%
        summarize(name = "RT",
                  mean = mean(RT),
                  se = sd(RT) / sqrt(n()),
                  .groups = "drop"),

      # response confidence
      df %>%
        group_by(X, subject,Correct) %>%
        summarize(name = "Confidence",
                  mean = mean(Confidence),
                  se = sd(Confidence) / sqrt(n()),
                  .groups = "drop")
    )

    plot = df_summary %>%
      filter(subject %in% subject_ids) %>%
      ggplot(aes(x = X, y = mean,
                 ymin = mean - 2*se, ymax = mean + 2*se)) +
      geom_pointrange(aes(color = as.factor(Correct))) +
      facet_grid(name ~ subject, scales = "free") +
      theme_classic(base_size = 16)

  }else{
    df_summary <- bind_rows(
      # accuracy
      df %>%
        mutate(X_bin = cut(X, bin)) %>%
        group_by(X_bin, subject) %>%
        summarize(name = "correct",
                  mean = mean(Y),
                  se = sd(Y) / sqrt(n()),
                  .groups = "drop"),

      # RT
      df %>%
        mutate(X_bin = cut(X, bin)) %>%
        group_by(X_bin, subject) %>%
        summarize(name = "RT",
                  mean = mean(RT),
                  se = sd(RT) / sqrt(n()),
                  .groups = "drop"),

      # response confidence
      df %>%
        mutate(X_bin = cut(X, bin)) %>%
        group_by(X_bin, subject,Correct) %>%
        summarize(name = "Confidence",
                  mean = mean(Confidence),
                  se = sd(Confidence) / sqrt(n()),
                  .groups = "drop")
    )

    plot = df_summary %>%
      filter(subject %in% subject_ids) %>%
      ggplot(aes(x = X_bin, y = mean,
                 ymin = mean - 2*se, ymax = mean + 2*se)) +
      geom_pointrange(aes(color = as.factor(Correct))) +
      facet_grid(name ~ subject, scales = "free") +
      theme_classic(base_size = 16)
  }

  return(plot)

}

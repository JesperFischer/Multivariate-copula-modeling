functions {
  real gauss_copula_cholesky_lpdf(matrix u, matrix L) {
    array[rows(u)] row_vector[cols(u)] q;
    for (n in 1:rows(u)) {
      q[n] = inv_Phi(u[n]);
    }

    return multi_normal_cholesky_lpdf(q | rep_row_vector(0, cols(L)), L)
            - std_normal_lpdf(to_vector(to_matrix(q)));
  }


matrix uvar_bounds(array[] int binom_y, vector cutpoints,array[] int correct,real beta_correct, int is_upper) {
  int N = size(binom_y);
  int K = size(cutpoints) + 1;  // Number of categories
  matrix[N, 1] u_bounds;



  for (n in 1:N) {
    int y = binom_y[n];  // Current category (1, 2, or 3)

    if (is_upper == 0) {  // Lower bound
      if (y == 1) {
        u_bounds[n, 1] = 0.0;
      } else {
        u_bounds[n, 1] = inv_logit(cutpoints[y - 1] - correct[n] * beta_correct);
      }
    } else {  // Upper bound
      if (y == K) {
        u_bounds[n, 1] = 1.0;
      } else {
        u_bounds[n, 1] = inv_logit(cutpoints[y]- correct[n] * beta_correct);
      }
    }
  }

  return u_bounds;
}


}

data {
  int<lower=0> N;
  vector[N] Y_con;
  array[N] int binom_y;
  array[N] int correct;
  int K;

}

transformed data{

  vector[K-1] cutmeans = linspaced_vector(K-1,-3,3);
  real cutsd = 1.0 / K;
}

parameters {
  ordered[K-1] cutpoints;
  real mean_RT;
  real <lower=0> sd_RT;
  real beta_correct;

  matrix<
    lower=uvar_bounds(binom_y, cutpoints,correct,beta_correct, 0),
    upper=uvar_bounds(binom_y, cutpoints,correct,beta_correct, 1)
    >[N, 1] u;

  cholesky_factor_corr[2] rho_chol;



}

model {
  matrix[N, 2] u_mix;

  for (n in 1:N) {
    u_mix[n, 1] = u[n,1];
    u_mix[n, 2] = normal_cdf(Y_con[n] | mean_RT,sd_RT);
  }

  Y_con ~ normal(mean_RT,sd_RT);

  u_mix ~ gauss_copula_cholesky(rho_chol);

  mean_RT ~ normal(2,2);
  sd_RT ~ normal(5,1);

  cutpoints ~ normal(cutmeans,cutsd);

  rho_chol ~ lkj_corr_cholesky(12);           // LKJ prior on correlation matrix


}

generated quantities {
  real rho = multiply_lower_tri_self_transpose(rho_chol)[1, 2];
}

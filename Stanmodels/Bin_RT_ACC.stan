functions {


  real psycho_ACC(real x, real alpha, real beta, real lapse){
   return (0.5 + (0.5 * (1-2*lapse)) * (tanh(beta*(x-alpha))  / 2 + 0.5));
  }

  real entropy(real p){
    return(-p * log(p) - (1-p) * log(1-p));
  }

  real gauss_copula_cholesky_lpdf(matrix u, matrix L) {
    array[rows(u)] row_vector[cols(u)] q;
    for (n in 1:rows(u)) {
      q[n] = inv_Phi(u[n]);
    }

    return multi_normal_cholesky_lpdf(q | rep_row_vector(0, cols(L)), L)
            - std_normal_lpdf(to_vector(to_matrix(q)));
  }

   vector gauss_copula_cholesky_per_row(matrix u, matrix L) {
    int N = rows(u);
    int D = cols(u);
    array[N] row_vector[D] q;
    vector[N] loglik;

    for (n in 1:N) {
        q[n,] = inv_Phi(u[n,]);
        loglik[n] = multi_normal_cholesky_lpdf(to_row_vector(q[n,]) |
                                                 rep_row_vector(0, D), L) - std_normal_lpdf(to_vector(to_matrix(q[n,])));
    }

    return loglik;
  }





  matrix uvar_bounds(array[] int binom_y, vector gm, vector tau_u,matrix L_u, matrix z_expo, array[] int S_id, vector X,
                     int is_upper) {

    int N = size(binom_y);
    matrix[N, 1] u_bounds;

    int S = cols(z_expo);
    int P = rows(z_expo);

    matrix[S, P] indi_dif = (diag_pre_multiply(tau_u, L_u) * z_expo)';

    matrix[S, P] param;

    for(p in 1:P){
      param[,p]= gm[p] + indi_dif[,p];
    }


    vector[S] alpha = (param[,1]);
    vector[S] beta = exp(param[,2]);
    vector[S] lapse = inv_logit(param[,3]) / 2;




    for (n in 1:N) {
      real theta = psycho_ACC(X[n], alpha[S_id[n]], beta[S_id[n]], lapse[S_id[n]]);
      if (is_upper == 0) {
        u_bounds[n, 1] = binom_y[n] == 0.0
                          ? 0.0 : binomial_cdf(binom_y[n] - 1 | 1, theta);
      } else {
        u_bounds[n, 1] = binomial_cdf(binom_y[n] | 1, theta);
      }
    }

    return u_bounds;
  }


  real ord_beta_reg_cdf(real y, real mu, real phi, real cutzero, real cutone) {

    vector[2] thresh;
    thresh[1] = cutzero;
    thresh[2] = cutzero + exp(cutone);

    real p0 = 1-inv_logit(mu - thresh[1]);

    real p_m = (inv_logit(mu - thresh[1])-inv_logit(mu - thresh[2]))  * beta_cdf(y | exp(log_inv_logit(mu) + log(phi)), exp(log1m_inv_logit(mu) + log(phi)));



    if (y < 0) {
      return 0;
    } else if (y == 0) {
      return p0;
    } else if (y == 1) {
      return 1-(1e-12);
    } else {
      return (p0 + p_m);
    }
  }

  // ordered beta function
  real ord_beta_reg_lpdf(real y, real mu, real phi, real cutzero, real cutone) {

    vector[2] thresh;
    thresh[1] = cutzero;
    thresh[2] = cutzero + exp(cutone);

  if(y==0) {
      return log1m_inv_logit(mu - thresh[1]);
    } else if(y==1) {
      return log_inv_logit(mu  - thresh[2]);
    } else {
      return log_diff_exp(log_inv_logit(mu - thresh[1]), log_inv_logit(mu - thresh[2])) +
                beta_lpdf(y|exp(log_inv_logit(mu) + log(phi)),exp(log1m_inv_logit(mu) + log(phi)));
    }
  }

  real induced_dirichlet_lpdf(real nocut, vector alpha, real phi, int cutnum, real cut1, real cut2) {
    int K = num_elements(alpha);
    vector[K-1] c = [cut1, cut1 + exp(cut2)]';
    vector[K - 1] sigma = inv_logit(phi - c);
    vector[K] p;
    matrix[K, K] J = rep_matrix(0, K, K);

    if(cutnum==1) {

    // Induced ordinal probabilities
    p[1] = 1 - sigma[1];
    for (k in 2:(K - 1))
      p[k] = sigma[k - 1] - sigma[k];
    p[K] = sigma[K - 1];

    // Baseline column of Jacobian
    for (k in 1:K) J[k, 1] = 1;

    // Diagonal entries of Jacobian
    for (k in 2:K) {
      real rho = sigma[k - 1] * (1 - sigma[k - 1]);
      J[k, k] = - rho;
      J[k - 1, k] = rho;
    }

    // divide in half for the two cutpoints

    // don't forget the ordered transformation

      return   dirichlet_lpdf(p | alpha)
           + log_determinant(J) + cut2;
    } else {
      return(0);
    }
  }
}



data {
  int<lower=0> N;
  int<lower=0> S;
  array[N] int S_id;

  array[S] int starts;
  array[S] int ends;

  array[N] int binom_y;
  vector[N] RT;

  vector[N] X;

  vector[S] minRT;

  array[S] int t_p_s;


}
transformed data{
  int P = 6;
}

parameters {
  vector[P] gm;
  vector<lower=0>[P] tau_u;
  cholesky_factor_corr[P] L_u;    // Between participant cholesky decomposition
  matrix[P, S] z_expo;    // Participant deviation from the group means


  matrix<
    lower=uvar_bounds(binom_y, gm, tau_u,L_u,z_expo, S_id,X, 0),
    upper=uvar_bounds(binom_y, gm, tau_u,L_u,z_expo, S_id,X, 1)
  >[N, 1] u;

  // cholesky_factor_corr[2] rho_chol;

  array[S] cholesky_factor_corr[3] rho_chol;

  vector[S] c0;
  vector[S] c11;
  vector<lower=0, upper = minRT>[S] rt_ndt;

}

transformed parameters{

   // Extracting individual deviations for each subject for each parameter
  matrix[S, P] indi_dif = (diag_pre_multiply(tau_u, L_u) * z_expo)';

  matrix[S, P] param;

  for(p in 1:P){
    param[,p]= gm[p] + indi_dif[,p];
  }

  vector[S] alpha = (param[,1]);
  vector[S] beta = exp(param[,2]);
  vector[S] lapse = inv_logit(param[,3]) / 2;

  vector[S] rt_int = param[,4];
  vector[S] rt_slope = param[,5];
  vector[S] rt_prec = exp(param[,6]);


  vector[N] entropy_t;

  vector[N] theta;

  profile("likelihood") {
  for (n in 1:N) {
  theta[n] = psycho_ACC(X[n], alpha[S_id[n]], beta[S_id[n]], lapse[S_id[n]]);

  entropy_t[n] = entropy(theta[n]);

  }

  }
}

model {

  gm[1] ~ normal(5,5); //global mean of beta
  gm[2] ~ normal(0,2); //global mean of beta
  gm[3] ~ normal(-4,1); //global mean of beta
  gm[4:6] ~ normal(0,3); //global mean of beta

  to_vector(z_expo) ~ std_normal();

  tau_u[1] ~ normal(3 , 3);
  tau_u[2] ~ normal(0 , 3);
  tau_u[3] ~ normal(0 , 3);
  tau_u[4:6] ~ normal(0 , 3);

  L_u ~ lkj_corr_cholesky(2);

  rt_ndt ~ normal(0.3,0.05);



  matrix[N, 2] u_mix;
  for (n in 1:N) {
    u_mix[n, 1] = u[n,1];

    u_mix[n, 2] = lognormal_cdf(RT[n] - rt_ndt[S_id[n]] | rt_int[S_id[n]] + rt_slope[S_id[n]] * entropy_t[n] , rt_prec[S_id[n]]);

    target += lognormal_lpdf(RT[n] - rt_ndt[S_id[n]] | rt_int[S_id[n]] + rt_slope[S_id[n]] * entropy_t[n], rt_prec[S_id[n]]);



  }


  for(s in 1:S){
    rho_chol[s] ~ lkj_corr_cholesky(2);

    u_mix[starts[s]:ends[s],] ~ gauss_copula_cholesky(rho_chol[s]);
  }
}

generated quantities {
  vector[S] rho_p_rt;

  matrix[P,P] correlation_matrix = L_u * L_u';

  vector[N] log_lik_bin = rep_vector(0,N);
  vector[N] log_lik_rt = rep_vector(0,N);
  vector[N] log_lik = rep_vector(0,N);


  matrix[N, 2] u_mixx;
  for (n in 1:N) {
    u_mixx[n, 1] = u[n,1];

    u_mixx[n, 2] = lognormal_cdf(RT[n] - rt_ndt[S_id[n]] | rt_int[S_id[n]] + rt_slope[S_id[n]] * entropy_t[n] , rt_prec[S_id[n]]);

  }

  vector[N] log_lik_cop;
  int pos;
  pos = 1;
  for (s in 1:S) {
    int n_s = t_p_s[s];
    vector[n_s] log_lik_s;

    log_lik_s = gauss_copula_cholesky_per_row(u_mixx[1:n_s, ], rho_chol[s]);

    // store results in the big vector
    log_lik_cop[pos:(pos + n_s - 1)] = log_lik_s;

    pos += n_s;
  }

  for(s in 1:S){

    rho_p_rt[s] = multiply_lower_tri_self_transpose(rho_chol[s])[1, 2];

  }
  for (n in 1:N) {
    log_lik_bin[n] = binomial_lpmf(binom_y[n] | 1, theta[n]);
    log_lik_rt[n] = lognormal_lpdf(RT[n] - rt_ndt[S_id[n]] | rt_int[S_id[n]] + rt_slope[S_id[n]] * entropy_t[n], rt_prec[S_id[n]]);
    log_lik[n] = log_lik_bin[n] + log_lik_rt[n] + log_lik_cop[n];
  }


}

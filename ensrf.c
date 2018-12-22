#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cblas-openblas.h>


/*  Calculate the mean across ensembles for a state vector.
 *
 *  Ignores nans.
 * 
 *  Parameters
 *  ----------
 *  m : int
 *      State vector length.
 *  n : int
 *      Ensemble length.
 *  x : double*
 *      State vector ensemble. A pointer to double row-major (m, n) array.
 * 
 *  Returns
 *  -------
 *  out : double*
 *      State vector mean. Pointer to (m) array.
 */
double* ensemble_mean(int m, int n, double *x)
{
    double *out = (double *)malloc(m * sizeof(double));
    int i;

    #pragma pragma omp parallel for
    for (i = 0; i < m; i++) {

        double sum = 0.0;
        double count = 0.0;

        int j;
        for (j = 0; j < n; j++) {

            if (isnan(x[i * n + j])) {
                continue;
            }
            else {
                sum += x[i * n + j];
                count += 1.0;
            }

        }

        if (count > 0.0) {
            out[i] = sum / count;
        }
        else {
            /* If all nans just put float nan */
            out[i] = NAN;
        }
    }

    return out;
}


/*  Calculate the deviation across ensembles for a state vector.
 *
 *  Ignores nans.
 *
 *  Parameters
 *  ----------
 *  m : int
 *      State vector length.
 *  n : int
 *      Ensemble length.
 *  x : double*
 *      State vector ensemble. Row-major (m, n) array.
 * xbar : double*
 *      State vector mean. Row-major (m) array.
 * 
 *  Returns
 *  -------
 *  out : double*
 *      State vector ensemble deviations. Pointer to (m) array.
 */
double* ensemble_dev(int m, int n, double *x, double *xbar) 
{
    double *x_prime = (double *)malloc(m * n * sizeof(double));
    int i, j;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if ((~isnan(x[i * n + j])) & (~isnan(x[i * n + j]))) {
                x_prime[i * n + j] = x[i * n + j] - xbar[i];
            }
            else {
                x_prime[i * n + j] = NAN;
            }
        }
    }
    return x_prime;
}


/* Variance of 1d array. Ignores nans.
 *
 * Parameters
 * ----------
 * m : int
 *     Array length.
 * x : double*
 *     m-length array to calculate variance of.
 * ddof : int
 *     Degrees of freedom for variance calc. Usually 0 for population 
 *     variance. Or 1 for unbiased sample variance.
 * 
 * Returns
 * -------
 * out : double
 *     Variance of array x.
 */
double sample_var_1d(int m, double *x, int ddof)
{
    double sumsq = 0.0;
    double count = 0.0;
    double out = NAN;
    int i;

    for (i = 0; i < m; i++) {

        if (isnan(x[i])) {
            continue;
        }
        else {
            count += 1;
            sumsq += pow(x[i], 2.0);
        }

    }

    if (count > 0) {
        out = sumsq / (count - ddof);
    }

    return out;
}


/* Covariance given arrays of deviations and degrees of freedom. Ingores nans
 *
 * Parameters
 * ----------
 * m : int
 *     State vector length.
 * n : int
 *     Ensemble size.
 * xprime : double*
 *     Ensemble state vector deviations. (m x n) column-major array.
 * yprime : double*
 *     Ensemble observation estimate deviations. (n) array.
 * ddof : double
 *     Degrees of freedom.
 * 
 * Returns
 * -------
 * out : double*
 *     Array (m) of covariances.
 */
double* sample_cov_2d1d(int m, int n, double *xprime, double *yprime, double ddof)
{
    double *out = (double *)malloc(m * sizeof(double));
    int i, j;
    double count = 0;
    double prodsum = 0.0;
    
    for (i = 0; i < m; i++) {

        count = 0;
        prodsum = 0.0;

        for (j = 0; j < n; j++) {
            if (isnan(xprime[i * n + j]) | isnan(yprime[j])) {
                continue;
            }
            else {
                prodsum += xprime[i * n + j] * yprime[j];
                count += 1;
            }
        }

        if (count > 0) {
            out[i] = prodsum / (count - ddof);
        }
        else {
            out[i] = NAN;
        }
    }

    return out;
}


/*
 * Inflate ensemble state vector variance.
 * 
 * Parameters
 * ----------
 * m : int
 *     State vector length.
 * n : int
 *     Ensemble size.
 * x : double*
 *     Ensemble state vector. (m x n) Row-major array.
 * infl : double
 *     Inflation factor to apply to state vector.
 * 
 * Returns
 * -------
 * out : double*
 *     Inflated ensemble state vector, an (m x n) array.
 */
double* inflate_state_variance(int m, int n, double *x, double infl)
{
    double *out = (double *)malloc(m * n * sizeof(double));
    double *xbar = (double *)malloc(m * sizeof(double));
    int i, j;

    xbar = ensemble_mean(m, n, x);

    for (i = 0; i < m; i++) {
        for (j = 0; i < n; j++) {

            if (isnan(x[i * n + j])) {
                out[i * n + j] = NAN;
            }
            else {
                out[i * n + j] = xbar[i] + (x[i * n + j] - xbar[i]) * infl;
            }

        }
    }

    return out;
}


/* Kalman gain (K) for sequential ensemble square root filter.
 * 
 * Parameters
 * ----------
 * m : int
 *     State vector length.
 * x : double*
 *     (m) array sample covariance between the background state deviations (x) 
 *     and observation estimate deviations (y). Often noted as PbHt.
 * y : double
 *      Sample variance of observation estimate deviations (y). Often noted as HPbHt.
 * r : double
 *      Observation error variance.
 * 
 * Returns
 * -------
 * k : double*
 *      (m) array Kalman Gain.
 */
double* kalman_gain(int m, double *x, double y, double r) 
{
    double *out = (double *)malloc(m * sizeof(double));
    double denom = y + r;
    int i;

    for (i = 0; i < m; i++) {
        if (isnan(x[i])) {
            out[i] = NAN;
        }
        else {
            out[i] = x[i] / denom;
        }
    }
    return out;
}


/* Modified kalman gain (~K) for sequential ensemble square root filter.
 *
 * Paramters
 * ---------
 * m : int
 *      State vector length.
 * k : double*
 *      Kalman gain, (m) array.
 * y : double
 *      Sample variance of observation estimate deviations. Often noted as HPbHt.
 * r : double
 *      Observation error variance.
 * 
 * Returns
 * -------
 * out : double*
 *      Modified Kalman Gain, (m) array.
 */
double* modified_kalman_gain(int m, double *k, double y, double r) 
{
    double *out = (double *)malloc(m * sizeof(double));
    double a;
    int i;

    a = 1 + sqrt(r / (y + r));

    for (i = 0; i < m; i++) {
        out[i] = 1 / (a * k[i]);
    }

    return out;
}


/* Update step for analysis mean (xabar).
 * 
 * Parameters
 * ----------
 * m : int
 *      State vector length.
 * x : double*
 *      Background state ensemble mean, (m) array.
 * k : double*
 *      Kalman gain, (m) array.
 * y0 : double
 *      Observation to be assimilated.
 * y : double
 *      Ensemble mean estimate for observation.
 */
double* analysis_mean(int m, double *x, double *k, double y0, double y)
{
    double* out = (double *)malloc(m * sizeof(double));
    int i, a;

    a = y0 - y;

    for (i = 0; i < m; i++) {
        if (isnan(x[i]) | isnan(k[i])) {
            out[i] = NAN;
        }
        else {
        out[i] = x[i] + k[i] * a;
        }
    }

    return out;
}


/* Update step for analysis deviation.
 *
 * Parameters
 * ----------
 * m : int
 *      State vector length.
 * n : int
 *      Ensemble length.
 * x : double*
 *      Background sate ensemble deviation, (m) array.
 * k : double*
 *      Modified Kalman Gain, (m) array.
 * y : double*
 *      Ensemble deviation for observation estimate, (n) array.
 * 
 * Returns
 * -------
 * out : double*
 *      Ensemble deviation for the updated ensemble state vector, (m x n) array.
 */
double* analysis_deviation(int m, int n, double *x, double *k, double *y) 
{
    double *out = (double *)malloc(m * n * sizeof(double));
    int i, j;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {

            if (isnan(x[i * n + j]) | isnan(k[i]) | isnan(y[j])) {
                out[i * n + j] = NAN;
            }
            else {
                out[i * n + j] = x[i * n + j] - k[i] * y[j];
            }

        }
    }

    return out;
}

double nanmean(int m, double *x) 
{
    double out = NAN;
    double sum = 0.0;
    double count = 0.0;
    int i;

    for (i = 0; i < m; i++) {
        
        if (isnan(x[i])) {
            continue;
        }
        else {
            sum += x[i];
            count += 1.0;
        }

    }

    if (count > 0.0) {
        out = sum / count;
    }

    return out;
}


double* ensrf_update(int m, int n, double *xb, double *yb, double y, double r)
{
    double *xb_bar = (double *)malloc(m * sizeof(double));
    double *xb_prime = (double *)malloc(m * n * sizeof(double));
    double yb_bar;
    double *yb_prime = (double *)malloc(n * sizeof(double));
    double yb_prime_var;
    double *xb_prime_yb_prime_cov = (double *)malloc(m * sizeof(double));
    double *kalman = (double *)malloc(m * sizeof(double));
    double *kalman_tilde = (double *)malloc(m * sizeof(double));
    double *xa_prime = (double *)malloc(m * n * sizeof(double));
    double *xa_bar = (double *)malloc(m * sizeof(double));
    double *xa = (double *)malloc(m * n * sizeof(double));
    int i, j, k;

    /* Background state mean and deviations. */
    xb_bar = ensemble_mean(m, n, xb);
    xb_prime = ensemble_dev(m, n, xb, xb_bar);

    /* Obs estimate mean and deviations. */
    yb_bar = nanmean(n, yb);
    for (i = 0; i < n; i++) {
        yb_prime[i] = yb[i] - yb_bar;
    }

    /* Obs estimate sample variance. */
    yb_prime_var = cblas_ddot(n, yb_prime, 1, yb_prime, 1) / (n - 1);

    /* 
     * Obs estimate sample covariance with background state
     * (i.e. background error covariance).
     */
    cblas_dgemv(CblasRowMajor, 'n', m, n, 1.0, xb_prime, m * n, yb_prime, 1, 1.0, xb_prime_yb_prime_cov, 1);
    for (i = 0; i < m; i++) {
        xb_prime_yb_prime_cov[i] /= (n - 1);
    }

    /* TODO: This is where we would apply covariance localization weights. */

    /* Assemble Kalman Gains. */
    kalman = kalman_gain(m, xb_prime_yb_prime_cov, yb_prime_var, r);
    kalman_tilde = modified_kalman_gain(m, kalman, yb_prime_var, r);

    /* Analysis state mean and deviations. */
    xa_prime = analysis_deviation(m, n, xb_prime, kalman_tilde, yb_prime);
    xa_bar = analysis_mean(m, xb_bar, kalman, y, yb_bar);

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            xa[i * n + j] = xa_bar[i] + xa_prime[i * n + j];
        }
    }

    return xa;
}


double* sensrf(int m, int n, int p, double *xb, int *yb_idx, double *y, double *r, double infl) 
{
    double *out = (double *)malloc(m * n * sizeof(double));
    int i, j;
    
    out = inflate_state_variance(m, n, xb, infl);

    /* TODO: Recursion might be nice here. */

    for (i = 0; i < p; i++) {
        out = ensrf_update(m, n, out, &out[yb_idx[i]], y[i], r[i]);
    }
    
    return out;
}


int main(int argc, char **argv)
{
    // double a[4 * 5] = {1.0, 2, 3, 4, 5,
    //                   6, 7, 8, 9,10,
    //                   11,12,13,14,15,
    //                   16,17,18,19,20
    //                    };
    double *out = (double *)malloc(5 * sizeof(double));
    double *a = (double *)malloc(4 * 5 * sizeof(double));
    int i, j;

    for (i=0; i < 4; i++) {
        for (j = 0; j < 5; j++) {

            a[i * 5 + j] = i * 5 + j;
            if (i * 5 + j == 10) {
                a[10] = NAN;  /* Add nan to test things */
            }

            printf("%f \n", a[i * 5 + j]);
        }
        printf("\n");
    }
    printf("\n\n");

    out = ensemble_mean(4, 5, (double *)a);

    // int i;
    for (i = 0; i < 4; i++) {
        printf("%f \n", out[i]);
    }
    printf("\n\n");
    return 0;
}

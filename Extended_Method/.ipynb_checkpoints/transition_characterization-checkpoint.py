'''
This file provides a function for the straightfoward
application of the Bayesian ramp-fit algorithm, which was
originally published in conjunction with the article 'Erhardt, T.
et al. Decadal-scale progression of the onset of
Dansgaard-Oeschger warming events. Clim. Past 15, 811–825 (2019)'
and which is available from
https://github.com/terhardt/DO-progression

This extended version is published with the article Slattery,
J. et al. The Temporal Phasing of Rapid Dansgaard–Oeschger
Warming Events Cannot Be Reliably Determined, and the code is 
available from https://github.com/johatt11/DO_Temporal_Phasing

While the original code is tailored to the analysis of specific
time series, here we provide an interface which allows to use the
algorithm on any time series which comprises an abrupt change of
the level of values.

Theory: The Bayesian ramp-fit is based on the assumption that
the time-series under investigation can be modelled as an AR1
process that fluctuates around a time dependent mean value. The
time dependent mean is assumed to have an initial, slowly
changing state, an abrupt transition, and a final slowly changing
state. We mostly want to apply this to time series with a clear,
sudden transition between two gradual changing states,
such that the gradient in the transition is much higher than that
before or after. However, there is no restriction on the respective
gradients, and so ultimately we are simply fitting three linear 
segments to our time series. Having said this, the initial conditions
and priors should tend to favour the sorts of transitions that
we expect to see.

          y0 + GS_slope * (t - t0)              if t_i<t0
y(t_i) =  y0 + dy * (t - t0) / dt               if t0<t_i<t0+dt
          y0 + dy + GIS_slope * (t - t0 - dt)   if t_i>t0+dt

The fluctuations around those mean values are mathematically 
described as an AR1 process: 

f(t_i) = alpha * f_i-1 + sqrt(sigma² * (1-alpha²)) * epsilon_i, 

with epsilon_i being a standard Gaussian random variable, alpha 
being the AR(1) coefficient and sigma being the variance of the 
AR(1) process (not of the gaussian noise). The algorithm further 
works with the autocorrelation time tau of the AR1 process which
is give by tau = - log(alpha) / delta t, alpha = exp(-1/tau *delta t)
where delta t denotes the sampling time step of the time series. 

The overall model thus reads: 
ypred(t_i) = y(t_i) + f(t_i)

and is uniquely defined by the set of 8 parameters: 
(t0, dt, y0, dy, GS_slope, GIS_slope, alpha, sigma)

The LIKELIHOOD function p(D|M) indicates the probability that the
model prediction exactly matches the observations: 

ypred = yobs

and is thus given by: 

p(D|M) = product_i=1^N 1/sqrt(2 * pi * sigma_eff²) 
     exp(-(1/2) * (delta_i-delta_i-1 * alpha)² / sigma_eff²)
      * p(ypred_0 = yobs_0), 

with sigma_eff = sqrt(sigma² * (1-alpha²)), delta_i = yobs_i - y_i 
and p(ypred_0 = yobs_0) = 1/sqrt(2 * pi * sigma_eff) 
                               exp(-(1/2) * (yobs)² / sigma_eff²)


the PRIORS for the parameters are defined in the file model.py as
part of the function lnpost

-----------------------------------------------------------------
note that all probabilities are ln transformed for computational
reasons                              
-----------------------------------------------------------------

# start time t0
lnp = normal_like(t0, 0.0, 50.0)                
Gaussian distribution with std of 50 and mean 0 

# transition length dt
lnp += gamma_like(dt, 2.0, 0.02) + np.log(dt)   
This is the same as gamma.logpdf(x, alpha = 2.0, scale=1/0.02) 
from scipy.stats

# start height y0 
lnp += 1.0        
uniform prior 

# step height dy                              
lnp += 1.0        
uniform prior

# initial gradient GS_slope
lnp += normal_like(GS_slope, 0.0, 0.1*abs(dy)/dt)

# initial gradient GIS_slope
lnp += normal_like(GIS_slope, 0.0, 0.1*abs(dy)/dt)

# autocorrelation time tau
lnp += gamma_like(tau, 1.5, 0.05) + np.log(tau) 
gamma distribution weighted with a linear function 

# variance sigma
no prior defined, this is equivalent to a uniform prior. 

'''

from model import fit_mcmc, linear_ramp, fit_rmse
import numpy as np
import pandas as pd
import sys


def combined_transition(time, traces):
    '''
    creates a transition for each sampled ptrans and subsequently 
    computes the p5, p50 and p95 percentiles for each point in time
    across all samples transition models.

    input
    -----
    time := the time axis 
    traces [pd.DataFrame] := the samples from the posterior dist. 

    output
    ------
    p5, p50, p95 := the percentiles across all transition models 
                    as defined by the sampled parameters for each 
                    point in time.

    '''
    m = traces.shape[0]
    n = len(time)
    ramps = np.zeros((m, n))
    for i, p in enumerate(traces.values):
        ramps[i] = linear_ramp(time, p[0], p[1], p[2], p[3], p[4], p[5])
    p5, p50, p95 = np.percentile(ramps, [5, 50, 95], axis=0)
    return p5, p50, p95


def find_trans_time(time, obs, rw=None):
    '''
    finding an initial guess for the transition onset time, based 
    on comparing means of two adjacent running windows. The point 
    in time with the biggest difference between the means computed 
    over the windows before and after this point is chosen as the 
    initial guess for t0. 

    input
    -----
    time := a time axis for the data
    obs := observational data
    rw := length of the running window in data points. 
          if None, then 1/10 of the total length is chosen. 

    output
    ------
    the initial guess for t0
    the difference score between adjacent running windows
    '''
    
    n = len(time)
    if rw is None:
        rw = int(n/10)

    score = np.zeros(int(n-2*rw))
    for i in range(n-2*rw):
        score[i] = np.abs(np.mean(obs[i:i+rw])
                          - np.mean(obs[i+rw:i+2*rw]))

    return time[np.argmax(score)+rw], score


def estimate_transition(time, obs, ptrans=None, pnoise=None, nwalkers=60, nsamples=60000, nthin=600, nburnin=2000):
    '''Create an MCMC sample from the posterior probability joint
    distribution of the model parameters.

    input
    -----
    time := time axis of the time series under study
    obs := observational data

    ptrans := initial guess for the parameters that define the transition 
    pnoise := initial guess for the parameters that define the 
              the AR1 process around the time dependent mean state
    
    nwalkers := number of walkers used for the MCMC sampler
    nsamples := number of accepted samples required by each walker
    nthin := number of samples that will be omitted between two
             samples that will be used for further analysis. 
             Omitting samples between two 'valid' samples 
             guarantees that only truly uncorrelated samples 
             survive.
    nburnin := number of initial samples before samples start being stored.

    output
    ------ 

    out [pandas DataFrame] := the output pandas DataFrame 
    comprises 8 columns ['t0', 'dt', 'y0', 'dy', 'GS_slope',
    'GIS_slope', 'sigma', 'tau'] 
    each of which represent one of the transition parameters.
    Each row of the DataFrame contains one sample from the 
    joint posterior distribution of the eight model parameters. 
    There are ntotal = nwalkers * nsamples / nthin 
    rows / samples comprised in the output.

    int_tau := Integrated autocorrelation time of the sampler. 
    For the retained samples to be truly uncorrelated we require
    nthin to be at least half int_tau. To avoid dependence on the
    initial guess we require nburnin to be at least a few
    int_tau.

    accept_frac := Mean acceptance fraction of the MCMC sampler
    
    Caution: with the default settings, the MCMC sampler may take 
    up an hour to run on a personal computer.

    '''

    if ptrans is None:
        # initial t0 is computed by maximizing the difference
        # between means of two neighbouring sliding windows
        t0, score = find_trans_time(time, obs)

        # no easy way to estimate dt - taking 1/10 of the total
        # time of the time series should be of the right magnitude

        dt = 1/10 * (time[-1] - time[0])

        # initial y0 is estimated by computing the mean before t0

        y0 = np.mean(obs[time < t0 - dt])
        
        # initial dy is estimated by subtracting means after
        # and before t0
        
        dy = (np.mean(obs[time > t0 + dt])
              - np.mean(obs[time < t0 - dt]))

        #initial GS_slope taken as 0.0

        GS_slope = 0.0
        
        #initial GIS_slope taken as 0.0

        GIS_slope = 0.0
        

    else:
        # unpack the initial guess for the parameter
        t0 = ptrans[0]
        dt = ptrans[1]
        y0 = ptrans[2]
        dy = ptrans[3]
        GS_slope = ptrans[4]
        GIS_slope = ptrans[5]
        

    if pnoise is None:
        mask1 = time < t0 - 0.5*dt
        mask2 = time > t0 + 0.5*dt
        # sigma is computed as the mean of obs std before and
        # after the estimated transition. Note that this sigma
        # denotes the variance of the AR(1) process, not the
        # amplitude of the noise. The amplitude of the gaussian
        # noise which is added in every time step of the AR(1)
        # process is given by: 
        # sigma_eff = np.sqrt(sigma ** 2 * (1 - alpha ** 2))

        sigma = (np.std(obs[mask1])/2 + 
                 np.std(obs[mask2])/2)

        # the autocorrelation time tau is defined as the inverse
        # restoring force of an OU process: tau = 1/gamma.
        # alpha = exp(-1/tau dt) of an AR1 process can be
        # estimated before and after the transition by
        # linearly regressing the increments dx = x_i+1 - x_i
        # against the values of the time series x_i.
        # This must be done on either side of the transition

        dobs1 = np.diff(obs[mask1])
        a1, b1 = np.polyfit(obs[mask1][:-1], dobs1, 1)

        dobs2 = np.diff(obs[mask2])
        a2, b2 = np.polyfit(obs[mask2][:-1], dobs2, 1)

        alpha = 1 + (a1+a2)/2

    else:
        alpha = pnoise[0]
        sigma = pnoise[1]

    # Occasionally, the method for estimating alpha fails
    # and produces a negative value. This is not allowed
    # and so we instead use a naive estimate of 
    # alpha = 0.1
    if alpha > 0:
        tau = -(time[1] - time[0])/np.log(alpha)
    else:
        tau = -(time[1] - time[0])/np.log(0.1)

    # shift time to center the transition
    stime = time - t0

    # take ln of the dt, tau and sigma

    lndt = np.log(dt)
    lntau = np.log(tau)
    lnsigma = np.log(sigma)

    # define thet0
    theta0 = (0, lndt, y0, dy, GS_slope, GIS_slope, lntau, lnsigma)

    # run mcmc machinery

    sampler = fit_mcmc(stime, obs, theta0, nwalkers=nwalkers, nsample=nsamples, nburnin=nburnin)

    # load every nthin'th sample from the walkers and reshape to
    # final dimensions
    trace = sampler.chain[:, ::nthin, :].reshape(-1, sampler.ndim).copy()

    # convert from sample space to meaningful space
    trace[:, [1, 6, 7]] = np.exp(trace[:, [1, 6, 7]])

    trace[:,0] = trace[:,0] + t0

    int_tau = np.array(sampler.get_autocorr_time())

    accept_frac = np.mean(sampler.acceptance_fraction)

    out = pd.DataFrame(trace, columns = ['t0', 'dt', 'y0', 'dy', 'GS_slope', 'GIS_slope',
                                         'tau', 'sigma'])
                            
    return out, int_tau, accept_frac
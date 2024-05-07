'''This script contains examples of how the synthetic data used
in the publication Slattery, J. et al. "The Temporal Phasing of
Rapid Dansgaard –Oeschger Warming Events Cannot Be Reliably
Determined", 2024 are generated.

We do not actually generate all of the >200,000 synthetic
transitions used for the analysis in that publication
in this file, but instead we give an
overview of how this was done.'''

import numpy as np
sys.path.append(os.path.join(sys.path[0],'Extended_Method'))
from model import linear_ramp
from distributions import sample_ar1

'''Annual-resolution transitions for the systematic testing
shown in Figure 3 (a-d & i-l):'''

delta = 1.0
t0 = 400.0
dt = 50.0
time = np.arange(0, 800, delta, dtype='float')

sigma = np.empty(10000)
tau = np.empty(10000)
GS_slope = np.empty(10000)
GIS_slope = np.empty(10000)
duration = np.empty(10000)

#Setting parameter ranges
for i in range(10000):
    sigma[i] = 0.05 + 0.05 * (i%10)
    tau[i] = 0.5 + 0.5 * int((i%100)/10)
    GS_slope[i] = -1.8e-3 + 4e-4 * int((i%100)/10)
    GIS_slope[i] = 0.0 - 3e-4 * int((i%100)/10)
    duration[i] = 10.0 + 10.0 * int((i%100)/10)

#Varying Noise / Signal and Autocorrelation Time
for i in range(10000):
    trans = linear_ramp(time, t0=400.0, dt=50.0, y0=0.0, dy = 1.0, GS_slope = 0.0, GIS_slope = -1e-3)
    noise = sample_ar1(len(time), alpha=np.exp(-1.0/tau[i]), sigma=sigma[i], x0=0.0)
    synt_trans = trans + noise

#Varying Noise / Signal and Greenland Stadial Slope
for i in range(10000):
    trans = linear_ramp(time, t0=400.0, dt=50.0, y0=0.0, dy = 1.0, GS_slope = GS_slope[i], GIS_slope = -1e-3)
    noise = sample_ar1(len(time), alpha=np.exp(-1), sigma=sigma[i], x0=0.0)
    synt_trans = trans + noise

#Varying Noise / Signal and Greenland Interstadial Slope
for i in range(10000):
    trans = linear_ramp(time, t0=400.0, dt=50.0, y0=0.0, dy = 1.0, GS_slope = 0.0, GIS_slope = GIS_slope[i])
    noise = sample_ar1(len(time), alpha=np.exp(-1), sigma=sigma[i], x0=0.0)
    synt_trans = trans + noise

#Varying Noise / Signal and Duration
for i in range(10000):
    trans = linear_ramp(time, t0=400.0, dt=duration[i], y0=0.0, dy = 1.0, GS_slope = 0.0, GIS_slope = -1e-3)
    noise = sample_ar1(len(time), alpha=np.exp(-1), sigma=sigma[i], x0=0.0)
    synt_trans = trans + noise

'''Decadal-resolution transitions for the systematic testing
shown in Figure 3 (e-h & m-p):'''

delta = 10.0
t0 = 400.0
dt = 50.0
time = np.arange(0, 800, delta, dtype='float')

sigma = np.empty(10000)
tau = np.empty(10000)
GS_slope = np.empty(10000)
GIS_slope = np.empty(10000)
duration = np.empty(10000)

for i in range(10000):
    sigma[i] = 0.02 + 0.02 * (i%10)
    tau[i] = 3.0 + 3.0 * int((i%100)/10)
    GS_slope[i] = -1.8e-3 + 4e-4 * int((i%100)/10)
    GIS_slope[i] = 0.0 - 3e-4 * int((i%100)/10)
    duration[i] = 10.0 + 10.0 * int((i%100)/10)

#Varying Noise / Signal and Autocorrelation Time
for i in range(10000):
    trans = linear_ramp(time, t0=400.0, dt=50.0, y0=0.0, dy = 1.0, GS_slope = 0.0, GIS_slope = -1e-3)
    noise = sample_ar1(len(time), alpha=np.exp(-delta/tau[i]), sigma=sigma[i], x0=0.0)
    synt_trans = trans + noise

#Varying Noise / Signal and Greenland Stadial Slope
for i in range(10000):
    trans = linear_ramp(time, t0=400.0, dt=50.0, y0=0.0, dy = 1.0, GS_slope = GS_slope[i], GIS_slope = -1e-3)
    noise = sample_ar1(len(time), alpha=np.exp(-1), sigma=sigma[i], x0=0.0)
    synt_trans = trans + noise

#Varying Noise / Signal and Greenland Interstadial Slope
for i in range(10000):
    trans = linear_ramp(time, t0=400.0, dt=50.0, y0=0.0, dy = 1.0, GS_slope = 0.0, GIS_slope = GIS_slope[i])
    noise = sample_ar1(len(time), alpha=np.exp(-1), sigma=sigma[i], x0=0.0)
    synt_trans = trans + noise

#Varying Noise / Signal and Duration
for i in range(10000):
    trans = linear_ramp(time, t0=400.0, dt=duration[i], y0=0.0, dy = 1.0, GS_slope = 0.0, GIS_slope = -1e-3)
    noise = sample_ar1(len(time), alpha=np.exp(-1), sigma=sigma[i], x0=0.0)
    synt_trans = trans + noise


'''"Analogous" transitions to CCSM4 model variables, used for
the bias estimates in Table 2.'''

delta = 10.0
time = np.arange(0, 800, delta, dtype='float')
t0 = 400.0
dt = 50.0

'''Create transitions with temperature-like parameters'''
for i in range(1000):
    alpha = np.exp(-delta/13.65)
    sigma = 0.066	
    trans = linear_ramp(time, t0=400.0, dt=50, y0=0.0, dy = 1.0, GS_slope = 1.64e-4, GIS_slope = -9.40e-4)
    noise = sample_ar1(len(time), alpha=alpha, sigma=sigma, x0=0)
    synt_trans = trans + noise

'''Create transitions with precipitation-like parameters,
assuming constant noise across the transition'''
for i in range(1000):
    alpha = np.exp(-delta/7.61)
    sigma = 0.154
    trans = linear_ramp(time, t0=400.0, dt=50, y0=0.0, dy = 1.0, GS_slope = 3.35e-4, GIS_slope = -1.273e-3)
    noise = sample_ar1(len(time), alpha=alpha, sigma=sigma, x0=0)
    synt_trans = trans + noise

'''Create transitions with precipitation-like parameters,
treating noise before and after the transition separately'''
for i in range(1000):
    alpha1 = np.exp(-delta/8.56)
    alpha2 = np.exp(-delta/19.35)
    sigma1 = 0.1807
    sigma2 = 0.1521
    trans = linear_ramp(time, t0=400.0, dt=50, y0=0.0, dy = 1.0, GS_slope = 3.35e-4, GIS_slope = -1.273e-3)
    noise1 = sample_ar1(int(t0/delta), alpha=alpha1, sigma=sigma1, x0=0)
    noise2 = sample_ar1(int(dt/delta), alpha=(alpha1+alpha2)/2, sigma=(sigma1+sigma2)/2, x0=noise1[-1])
    noise3 = sample_ar1(len(time)-int((t0+dt)/delta), alpha=alpha2, sigma=sigma2, x0=noise2[-1])
    noise = np.append(noise1, noise2)
    noise = np.append(noise, noise3)
    synt_trans = trans + noise

'''Create transitions with sea-ice -like parameters'''
for i in range(1000):
    alpha = np.exp(-delta/13.52)
    sigma = 0.068
    trans = linear_ramp(time, t0=400.0, dt=50, y0=1.0, dy = -1.0, GS_slope = -1.68e-4, GIS_slope = 9.24e-4)
    noise = sample_ar1(len(time), alpha=alpha, sigma=sigma, x0=0)
    synt_trans = trans + noise

'''Create transitions with AMOC-like parameters'''
for i in range(1000):
    alpha = np.exp(-delta/20.32)
    sigma = 0.093
    trans = linear_ramp(time, t0=400.0, dt=50, y0=0.0, dy = 1.0, GS_slope = 2.89e-4, GIS_slope = -1.697e-3)
    noise = sample_ar1(len(time), alpha=alpha, sigma=sigma, x0=0)
    synt_trans = trans + noise

'''Create transitions with NAO-like parameters'''
for i in range(1000):
    alpha = np.exp(-delta/6.77)
    sigma = 0.076
    trans = linear_ramp(time, t0=400.0, dt=50, y0=0.0, dy = 1.0, GS_slope = 1.12e-4, GIS_slope = -7.89e-4)
    noise = sample_ar1(len(time), alpha=alpha, sigma=sigma, x0=0)
    synt_trans = trans + noise


'''"Analogous transitions to NGRIP proxies, used for the 
bias estimates in Table 2.'''

'''Create transitions with dO18-like parameters'''
dO18_time = np.arange(500, step=5.0, dtype='float')
for i in range(1000):
    alpha = 0.318
    sigma = 0.355
    trans = linear_ramp(dO18_time, t0=250, dt=50, y0=0.0, dy = 1.0, GS_slope = 4.16e-4, GIS_slope = -5.98e-4)
    noise = sample_ar1(len(dO18_time), alpha=alpha, sigma=sigma, x0=0)
    synt_trans = trans + noise

'''Create transitions with Annual Layer Thickness-like parameters'''
time = np.arange(500, step=2.0, dtype='float')
for i in range(1000):
    alpha = 0.084
    sigma = 0.369
    trans = linear_ramp(time, t0=250, dt=50, y0=0.0, dy = 1.0, GS_slope = 8.22e-4, GIS_slope = -7.62e-4)
    noise = sample_ar1(len(time), alpha=alpha, sigma=sigma, x0=0)
    synt_trans = trans + noise

'''Create transitions with Na+ -like parameters'''
time = np.arange(500, step=2.0, dtype='float')
for i in range(1000):
    alpha = 0.287
    sigma = 0.407
    trans = linear_ramp(time, t0=250, dt=50, y0=1.0, dy = -1.0, GS_slope = -4.59e-4, GIS_slope = 2.12e-4)
    noise = sample_ar1(len(time), alpha=alpha, sigma=sigma, x0=0)
    synt_trans = trans + noise

'''Create transitions with Ca2+ -like parameters'''
time = np.arange(500, step=2.0, dtype='float')
for i in range(1000):
    alpha = 0.529
    sigma = 0.179
    trans = linear_ramp(time, t0=250, dt=50, y0=1.0, dy = -1.0, GS_slope = -8.83e-4, GIS_slope = 4.04e-4)
    noise = sample_ar1(len(time), alpha=alpha, sigma=sigma, x0=0)
    synt_trans = trans + noise
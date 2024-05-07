import numpy as np
import pandas as pd
import proplot as pplt
import os
from scipy.stats import gaussian_kde
import sys

'''Create Fig. A1, showing how the uncertainty
depends on noise and slope parameters for both
implementations.'''

gs = pplt.GridSpec(nrows=2, ncols=2, hpad=4)
fig = pplt.figure(refheight=4, share=False, fontsize=18)
levels = np.arange(10,100,step=10)
cmap1 = pplt.Colormap('Reds')

sigma = np.empty(10000)
GS_slope = np.empty(10000)
GIS_slope = np.empty(10000)

for i in range(10000):
    sigma[i] = 0.02 + 0.02 * (i%10)
    GS_slope[i] = -1.8e-3 + 4e-4 * int((i%100)/10)
    GIS_slope[i] = 0.0 - 3e-4 * int((i%100)/10)

df = pd.read_pickle('Data/decadal_synthetics/Original_Method/double_GS_pickle')
df = df.where(df['mean_time']>300,other=np.nan)
df = df.where(df['mean_time']<450,other=np.nan)
time = df['time_width']
d = {'time': time, 'sigma': sigma, 'GS_slope': GS_slope}
double_GS_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/decadal_synthetics/Original_Method/double_GIS_pickle')
df = df.where(df['mean_time']>300,other=np.nan)
df = df.where(df['mean_time']<450,other=np.nan)
time = df['time_width']
d = {'time': time, 'sigma': sigma, 'GIS_slope': GIS_slope}
double_GIS_df = pd.DataFrame(data=d)

double_GIS_df['sig_GIS'] = double_GIS_df['sigma'] + double_GIS_df['GIS_slope']
double_GS_df['sig_GS'] = double_GS_df['sigma'] + double_GS_df['GS_slope']

sigma_GIS_mean = np.empty((10,10))

for i in range(100):
    sigma_GIS_mean[i%10,int(i/10)] = double_GIS_df.groupby('sig_GIS').mean()['time'].values[i]

sigma_GS_mean = np.empty((10,10))

for i in range(100):
    sigma_GS_mean[i%10,int(i/10)] = double_GS_df.groupby('sig_GS').mean()['time'].values[i]

ax = fig.subplot(gs[0,:], fontsize=24)
ax.format(title='Original Method', titlepad=15, xticks=[], yticks=[], ec='white', abc=False)

ax1 = fig.subplot(gs[0,0], abc=True, abcloc='ul', number=1, fontsize=18)
ax1.format(xlabel='Noise / Signal', ylabel='Greenland Stadial Slope / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=1000*GS_slope[np.arange(10,110,step=20)], 
              yminorlocator=1000*GS_slope[np.arange(100,step=20)])
GS = ax1.contourf(x=sigma[:10], y=1000*GS_slope[np.arange(100,step=10)], cmap=cmap1,
                       z=sigma_GS_mean,extend='both',levels=levels)

ax2 = fig.subplot(gs[0,1], abc=True, abcloc='ul', number=2, fontsize=18)
ax2.format(xlabel='Noise / Signal', ylabel='Absolute Greenland Interstadial Slope \n / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=np.abs(1000*GIS_slope[np.arange(10,110,step=20)]),
              yminorlocator=0.3)
GIS = ax2.contourf(x=sigma[:10], y=np.abs(1000*GIS_slope[np.arange(100,step=10)]), cmap=cmap1,
                       z=np.flip(sigma_GIS_mean,axis=0),extend='both',levels=levels)

colorbar = fig.colorbar(GS, loc='bottom')
colorbar.ax.tick_params(labelsize=18)
colorbar.ax.set_title('Onset Time Uncertainty / Years', fontsize=18)

df = pd.read_pickle('Data/decadal_synthetics/Extended_Method/double_GS_pickle')
df = df.where(df['mean_time']>300,other=np.nan)
df = df.where(df['mean_time']<450,other=np.nan)
time = df['time_width']
d = {'time': time, 'sigma': sigma, 'GS_slope': GS_slope}
double_GS_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/decadal_synthetics/Extended_Method/double_GIS_pickle')
df = df.where(df['mean_time']>300,other=np.nan)
df = df.where(df['mean_time']<450,other=np.nan)
time = df['time_width']
d = {'time': time, 'sigma': sigma, 'GIS_slope': GIS_slope}
double_GIS_df = pd.DataFrame(data=d)

double_GIS_df['sig_GIS'] = double_GIS_df['sigma'] + double_GIS_df['GIS_slope']
double_GS_df['sig_GS'] = double_GS_df['sigma'] + double_GS_df['GS_slope']

sigma_GIS_mean = np.empty((10,10))

for i in range(100):
    sigma_GIS_mean[i%10,int(i/10)] = double_GIS_df.groupby('sig_GIS').mean()['time'].values[i]

sigma_GS_mean = np.empty((10,10))

for i in range(100):
    sigma_GS_mean[i%10,int(i/10)] = double_GS_df.groupby('sig_GS').mean()['time'].values[i]

ax = fig.subplot(gs[1,:], fontsize=24)
ax.format(title='Extended Method', titlepad=15, xticks=[], yticks=[], ec='white', abc=False)

ax3 = fig.subplot(gs[1,0], abc=True, abcloc='ul', number=3, fontsize=18)
ax3.format(xlabel='Noise / Signal', ylabel='Greenland Stadial Slope / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=1000*GS_slope[np.arange(10,110,step=20)], 
              yminorlocator=1000*GS_slope[np.arange(100,step=20)])
GS = ax3.contourf(x=sigma[:10], y=1000*GS_slope[np.arange(100,step=10)], cmap=cmap1,
                       z=sigma_GS_mean,extend='both',levels=levels)

ax4 = fig.subplot(gs[1,1], abc=True, abcloc='ul', number=4, fontsize=18)
ax4.format(xlabel='Noise / Signal', ylabel='Absolute Greenland Interstadial Slope \n / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=np.abs(1000*GIS_slope[np.arange(10,110,step=20)]),
              yminorlocator=0.3)
GIS = ax4.contourf(x=sigma[:10], y=np.abs(1000*GIS_slope[np.arange(100,step=10)]), cmap=cmap1,
                       z=np.flip(sigma_GIS_mean,axis=0),extend='both',levels=levels)

fig.save('Supplementary_Figures/Figure_A1')


'''Create Fig. A2, showing the reduced sensitivity to
the choice of window when using the extended method'''
'''We also look at this question more systematically for the CCSM4 200ppm AMOC 1st transition.'''


onset_time_array = np.empty((2,5,5))
start_years_array = np.empty((5,5))
end_years_array = np.empty((5,5))
fit_type_array = np.empty((50),dtype='str')
for i in range(50):
    if i < 25:
        fit_type_array[i] = 'slope'
    else:
        fit_type_array[i] = 'flat'

start_years = (300,400,500,600,700)
end_years = (1100,1200,1300,1400,1500)


for i, start_year in enumerate(start_years):
    for j, end_year in enumerate(end_years):
        start_years_array[i,j] = start_years[i]
        end_years_array[i,j] = end_years[j]
        
        traces = pd.read_pickle('Data/window_sensitivity/slope_'+str(start_year)+'_'+str(end_year))
        onset_time_array[0,i,j] = np.mean(traces['t0'])

        traces = pd.read_pickle('Data/window_sensitivity/flat_'+str(start_year)+'_'+str(end_year))
        onset_time_array[1,i,j] = np.mean(traces['t0'])
            

d = {'fit type': fit_type_array, 'start year': np.append(start_years_array.flatten(),start_years_array.flatten()),
     'end year': np.append(end_years_array.flatten(),end_years_array.flatten()),
     'onset time': np.append(onset_time_array[0,:,:].flatten(),onset_time_array[1,:,:].flatten())}
df = pd.DataFrame(data=d)

d = {'Original': df['onset time'][25:], 'Extended': df['onset time'][:25]}
df_reduced = pd.DataFrame(data=d)


fig, ax = pplt.subplots(figsize=(5,4), fontsize=16)
ax.format(ylabel='Posterior Mean Onset Time / Years', xlabel='Implementation of Ramp Fitting Method')
ax.scatter(df_reduced, mean=True, boxstd=True, barstd=True)
fig.save('Supplementary_Figures/Figure_A2')


'''Create Fig. A3, which shows low noise benchmarks.'''

tau = np.empty(1000)
for i in range(1000):
    tau[i] = 3.0 + 3.0 *(i%10)

GS_slope = np.empty(1000)
for i in range(1000):
    GS_slope[i] = -1.8e-3 + 4e-4*(i%10)

GIS_slope = np.empty(1000)
for i in range(1000):
    GIS_slope[i] = -2.7e-3 + 3e-4*(i%10)

duration = np.empty(1000)
for i in range(1000):
    duration[i] = 10.0 + 10.0*(i%10)


fig, axs = pplt.subplots(nrows=1,ncols=4,share=False)
fig.format(fontsize=14, abc=True, abcloc='ul')

df = pd.read_pickle('Data/benchmarks/benchmark_autocorr_pickle')
d = {'true_tau': tau, 'sampled_tau': df['tau']}
plotting_df = pd.DataFrame(data=d)
axs[0].scatter(tau[:10], plotting_df.groupby('true_tau').mean()['sampled_tau'])
axs[0].plot(tau[:10],tau[:10],color='black',ls='--')
axs[0].format(xlabel='True Autocorrelation \n Time / Years', ylabel = 'Estimated Autocorrelation \n Time / Years')

df = pd.read_pickle('Data/benchmarks/benchmark_GS_slope_pickle')
d = {'true_GS': GS_slope, 'sampled_GS': df['GS_slope']}
plotting_df = pd.DataFrame(data=d)
axs[1].scatter(1000*GS_slope[:10], 1000*plotting_df.groupby('true_GS').mean()['sampled_GS'])
axs[1].plot(1000*GS_slope[:10],1000*GS_slope[:10],color='black',ls='--')
axs[1].format(xlabel='True Greenland Stadial \n  Slope / Kiloyears$^{-1}$',
              ylabel = 'Estimated Greenland Stadial \n Slope / Kiloyears$^{-1}$')

df = pd.read_pickle('Data/benchmarks/benchmark_GIS_slope_pickle')
d = {'true_GIS': GIS_slope, 'sampled_GIS': df['GIS_slope']}
plotting_df = pd.DataFrame(data=d)
axs[2].scatter(1000*np.flip(np.abs(GIS_slope[:10])), 1000*np.flip(np.abs(plotting_df.groupby('true_GIS').mean()['sampled_GIS'])))
axs[2].plot(1000*np.flip(np.abs(GIS_slope[:10])),1000*np.flip(np.abs(GIS_slope[:10])),color='black',ls='--')
axs[2].format(xlabel='True Absolute Greenland \n Interstadial Slope / Kiloyears$^{-1}$',
              ylabel = 'Estimated Absolute Greenland \n Interstadial Slope / Kiloyears$^{-1}$')

df = pd.read_pickle('Data/benchmarks/benchmark_duration_pickle')
d = {'true_dur': duration, 'sampled_dur': df['duration']}
plotting_df = pd.DataFrame(data=d)
axs[3].scatter(duration[:10], plotting_df.groupby('true_dur').mean()['sampled_dur'])
axs[3].plot(duration[:10],duration[:10],color='black',ls='--')
axs[3].format(xlabel='True Duration / Years', ylabel = 'Estimated Duration / Years')

fig.save('Supplementary_Figures/Figure_A3')


'''Create Fig. C1, which shows the bias when
using uniform priors instead of the standard priors.'''

sigma = np.empty(1000)
for i in range(1000):
    sigma[i] = 0.02 + 0.02 * (i%10)

GS_slope = np.empty(1000)
for i in range(1000):
    GS_slope[i] = -1.8e-3+4e-4*(i%10)

GIS_slope = np.empty(1000)
for i in range(1000):
    GIS_slope[i] = -2.7e-3+3e-4*(i%10)

duration = np.empty(1000)
for i in range(1000):
    duration[i] = 10.0 + 10.0 * (i%10)

fig = pplt.figure(figsize=(10,12), sharey=True, sharex=False)
axs = fig.add_subplots(nrows=4, ncols=1)
fig.format(fontsize=16)
axs.format(ylim=(-32,3),)

df = pd.read_pickle('Data/decadal_synthetics/sigma_alt_prior_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']

d = {'time': time-400, 'sigma': sigma}
sigma_df = pd.DataFrame(data=d)

axs[0].scatter(np.arange(0.02,0.22,step=0.02), sigma_df.groupby('sigma').mean()['time'])
axs[0].format(ylabel='Bias / Years',xlabel='Noise / Signal')

fit = -605.036975*np.arange(0.02,0.22,step=0.02)**2
axs[0].plot(np.arange(0.02,0.22,step=0.02), fit, color='black')

df = pd.read_pickle('Data/decadal_synthetics/GS_alt_prior_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']

d = {'time': time-400, 'GS_slope': GS_slope}
GS_df = pd.DataFrame(data=d)

axs[1].scatter(1000*GS_slope[:10], GS_df.groupby('GS_slope').mean()['time'])
axs[1].format(ylabel='Bias / Years',xlabel='Greenland Stadial Slope / $Kiloyears^{-1}$')

fit = -2.148993e+01 + GS_slope[:10] * -6.693412e+03 + GS_slope[:10]**2 * 1.602649e+06
axs[1].plot(1000*GS_slope[:10],fit,color='black')

df = pd.read_pickle('Data/decadal_synthetics/GIS_alt_prior_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']

d = {'time': time-400, 'GIS_slope': GIS_slope}
GIS_df = pd.DataFrame(data=d)

axs[2].scatter(1000*np.flip(np.abs(GIS_slope[:10])), np.flip(GIS_df.groupby('GIS_slope').mean()['time']))
axs[2].format(ylabel='Bias / Years',xlabel='Absolute Greenland Interstadial Slope / $Kiloyears^{-1}$')

fit = -25.565183 + GIS_slope[:10] * -5695.166951
axs[2].plot(1000*np.flip(np.abs(GIS_slope[:10])),np.flip(fit),color='black')

df = pd.read_pickle('Data/decadal_synthetics/dur_alt_prior_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<500,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']

d = {'time': time-400, 'duration': duration}
plotting_df = pd.DataFrame(data=d)

axs[3].scatter(duration[:10], plotting_df.groupby('duration').mean()['time'])
axs[3].format(ylabel='Bias / Years',xlabel='Transition Duration / Years')

fit = -7.352681 -0.667823 * duration[:10] +  0.005423 * duration[:10]**2

axs[3].plot(duration[:10], fit, color='black')
fig.save('Supplementary_Figures/Figure_C1')


'''Create Fig. C2, showing the bias when using a 
simple least-squares fit.'''

sigma = np.empty(1000)
for i in range(1000):
    sigma[i] = 0.02 + 0.02 * (i%10)

GS_slope = np.empty(1000)
for i in range(1000):
    GS_slope[i] = -1.8e-3+4e-4*(i%10)

GIS_slope = np.empty(1000)
for i in range(1000):
    GIS_slope[i] = -2.7e-3+3e-4*(i%10)

duration = np.empty(1000)
for i in range(1000):
    duration[i] = 10.0 + 10.0 * (i%10)

fig = pplt.figure(figsize=(10,12), sharey=True, sharex=False)
axs = fig.add_subplots(nrows=4, ncols=1)
fig.format(fontsize=16, abc=True, abcloc='ul')
axs.format(ylim=(-16,16),)

df = pd.read_pickle('Data/least_squares/sigma_pickle')
df = df.where(df['ls_time']>300,other=np.nan)
df = df.where(df['ls_time']<450,other=np.nan)
time = df['ls_time']
d = {'time': time-400, 'sigma': sigma}
sigma_df = pd.DataFrame(data=d)
axs[0].scatter(np.arange(0.02,0.22,step=0.02), sigma_df.groupby('sigma').mean()['time'], label='Decadal \nResolution')
axs[0].format(ylabel='Bias / Years',xlabel='Noise / Signal')

df = pd.read_pickle('Data/least_squares/GS_slope_pickle')
df = df.where(df['ls_time']>300,other=np.nan)
df = df.where(df['ls_time']<450,other=np.nan)
time = df['ls_time']
d = {'time': time-400, 'GS_slope': GS_slope}
GS_df = pd.DataFrame(data=d)
axs[1].scatter(1000*GS_slope[:10], GS_df.groupby('GS_slope').mean()['time'], label=None)
axs[1].format(ylabel='Bias / Years',xlabel='Greenland Stadial Slope / $Kiloyears^{-1}$')
fit = GS_slope[:10] * -1.540120e+04 + GS_slope[:10]**3 * 3.028714e+09
axs[1].plot(1000*GS_slope[:10],fit,color='black')

df = pd.read_pickle('Data/least_squares/GIS_slope_pickle')
df = df.where(df['ls_time']>300,other=np.nan)
df = df.where(df['ls_time']<450,other=np.nan)
time = df['ls_time']
d = {'time': time-400, 'GIS_slope': GIS_slope}
GIS_df = pd.DataFrame(data=d)
axs[2].scatter(1000*np.flip(np.abs(GIS_slope[:10])), np.flip(GIS_df.groupby('GIS_slope').mean()['time']), label=None)
axs[2].format(ylabel='Bias / Years',xlabel='Absolute Greenland Interstadial Slope / $Kiloyears^{-1}$')
fit = -1.652046e+00 + GIS_slope[:10]**2 * 1.588440e+06
axs[2].plot(1000*np.flip(np.abs(GIS_slope[:10])),np.flip(fit),color='black')

df = pd.read_pickle('Data/least_squares/duration_pickle')
df = df.where(df['ls_time']>300,other=np.nan)
df = df.where(df['ls_time']<450,other=np.nan)
time = df['ls_time']
d = {'time': time-400, 'duration': duration}
plotting_df = pd.DataFrame(data=d)
axs[3].scatter(duration[:10], plotting_df.groupby('duration').mean()['time'], label=None)
axs[3].format(ylabel='Bias / Years',xlabel='Transition Duration / Years')

fig.save('Supplementary_Figures/Figure_C2')



'''Create Fig. C3 showing the bias in the posterior-median 
transition onset time.'''

gs = pplt.GridSpec(nrows=4, ncols=4, hpad=4)
fig = pplt.figure(refheight=4, share=False, fontsize=18)
contour_levs = [-11,-7,-4,-2,-1,0,1,2,4,7,11]
cmap1 = pplt.Colormap('RdBu_r')

#Original Method, Annual Resolution
sigma = np.empty(10000)
tau = np.empty(10000)
GS_slope = np.empty(10000)
GIS_slope = np.empty(10000)
duration = np.empty(10000)
for i in range(10000):
    sigma[i] = 0.05 + 0.05 * (i%10)
    tau[i] = 0.5 + 0.5 * int((i%100)/10)
    GS_slope[i] = -1.8e-3 + 4e-4 * int((i%100)/10)
    GIS_slope[i] = 0.0 - 3e-4 * int((i%100)/10)
    duration[i] = 10.0 + 10.0 * int((i%100)/10)

df = pd.read_pickle('Data/annual_synthetics/Original_Method/double_GS_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'GS_slope': GS_slope}
double_GS_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/annual_synthetics/Original_Method/double_GIS_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'GIS_slope': GIS_slope}
double_GIS_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/annual_synthetics/Original_Method/double_dur_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'duration': duration}
double_dur_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/annual_synthetics/Original_Method/double_tau_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'tau': tau}
double_tau_df = pd.DataFrame(data=d)

double_GIS_df['sig_GIS'] = double_GIS_df['sigma'] + double_GIS_df['GIS_slope']
double_GS_df['sig_GS'] = double_GS_df['sigma'] + double_GS_df['GS_slope']
double_dur_df['sig_dur'] = double_dur_df['sigma'] + double_dur_df['duration']
double_tau_df['sig_tau'] = double_tau_df['sigma'] + 10*double_tau_df['tau']

sigma_GIS_mean = np.empty((10,10))

for i in range(100):
    sigma_GIS_mean[i%10,int(i/10)] = double_GIS_df.groupby('sig_GIS').mean()['time'].values[i]

sigma_GS_mean = np.empty((10,10))

for i in range(100):
    sigma_GS_mean[i%10,int(i/10)] = double_GS_df.groupby('sig_GS').mean()['time'].values[i]

sigma_dur_mean = np.empty((10,10))

for i in range(100):
    sigma_dur_mean[i%10,int(i/10)] = double_dur_df.groupby('sig_dur').mean()['time'].values[i]

sigma_tau_mean = np.empty((10,10))

for i in range(100):
    sigma_tau_mean[i%10,int(i/10)] = double_tau_df.groupby('sig_tau').mean()['time'].values[i]

ax = fig.subplot(gs[0,:], fontsize=24)
ax.format(title='Original Method at Annual Resolution', titlepad=15, xticks=[], yticks=[], ec='white', abc=False)

ax1 = fig.subplot(gs[0,0], abc=True, abcloc='ul', number=1, fontsize=18)
ax1.format(xlabel='Noise / Signal', ylabel='Autocorrelation Time / Years',
              xticks=sigma[np.arange(1,11,step=2)], xminorticks=sigma[np.arange(10,step=2)], yticks=tau[np.arange(10,110,step=20)], 
              yminorticks=tau[np.arange(100,step=20)])
dur = ax1.contourf(x=sigma[:10], y=tau[np.arange(100,step=10)], cmap=cmap1,
                       z=np.transpose(sigma_tau_mean),levels=contour_levs, extend='both', inbounds=False)

ax2 = fig.subplot(gs[0,1], abc=True, abcloc='ul', number=2, fontsize=18)
ax2.format(xlabel='Noise / Signal', ylabel='Greenland Stadial Slope / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorticks=sigma[np.arange(10,step=2)], yticks=1000*GS_slope[np.arange(10,110,step=20)], 
              yminorlocator=1000*GS_slope[np.arange(100,step=20)])
GS = ax2.contourf(x=sigma[:10], y=1000*GS_slope[np.arange(100,step=10)], cmap=cmap1,
                       z=sigma_GS_mean,levels=contour_levs, extend='both')

ax3 = fig.subplot(gs[0,2], abc=True, abcloc='ul', number=3, fontsize=18)
ax3.format(xlabel='Noise / Signal', ylabel='Absolute Greenland Interstadial Slope \n / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorticks=sigma[np.arange(10,step=2)],
              yticks=np.abs(1000*GIS_slope[np.arange(10,110,step=20)]),
              yminorlocator=0.3)
slope = ax3.contourf(x=sigma[:10], y=np.abs(1000*GIS_slope[np.arange(100,step=10)]), cmap=cmap1,
                       z=np.flip(sigma_GIS_mean,axis=0),levels=contour_levs, extend='both')

ax4 = fig.subplot(gs[0,3], abc=True, abcloc='ul', number=4, fontsize=18)
ax4.format(xlabel='Noise / Signal', ylabel='Transition Duration / Years',
             xticks=sigma[np.arange(1,11,step=2)], xminorticks=sigma[np.arange(10,step=2)],
              yticks=duration[np.arange(10,110,step=20)], yminorlocator=10)
dur = ax4.contourf(x=sigma[:10], y=duration[np.arange(100,step=10)], cmap=cmap1,
                       z=np.transpose(sigma_dur_mean),levels=contour_levs, extend='both')



#Original Method, Decadal Resolution

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

df = pd.read_pickle('Data/decadal_synthetics/Original_Method/double_GS_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'GS_slope': GS_slope}
double_GS_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/decadal_synthetics/Original_Method/double_GIS_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'GIS_slope': GIS_slope}
double_GIS_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/decadal_synthetics/Original_Method/double_dur_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'duration': duration}
double_dur_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/decadal_synthetics/Original_Method/double_tau_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'tau': tau}
double_tau_df = pd.DataFrame(data=d)

double_GIS_df['sig_GIS'] = double_GIS_df['sigma'] + double_GIS_df['GIS_slope']
double_GS_df['sig_GS'] = double_GS_df['sigma'] + double_GS_df['GS_slope']
double_dur_df['sig_dur'] = double_dur_df['sigma'] + double_dur_df['duration']
double_tau_df['sig_tau'] = double_tau_df['sigma'] + double_tau_df['tau']

sigma_GIS_mean = np.empty((10,10))

for i in range(100):
    sigma_GIS_mean[i%10,int(i/10)] = double_GIS_df.groupby('sig_GIS').mean()['time'].values[i]

sigma_GS_mean = np.empty((10,10))

for i in range(100):
    sigma_GS_mean[i%10,int(i/10)] = double_GS_df.groupby('sig_GS').mean()['time'].values[i]

sigma_dur_mean = np.empty((10,10))

for i in range(100):
    sigma_dur_mean[i%10,int(i/10)] = double_dur_df.groupby('sig_dur').mean()['time'].values[i]

sigma_tau_mean = np.empty((10,10))

for i in range(100):
    sigma_tau_mean[i%10,int(i/10)] = double_tau_df.groupby('sig_tau').mean()['time'].values[i]

ax = fig.subplot(gs[1,:], fontsize=24)
ax.format(title='Original Method at Decadal Resolution', titlepad=15, xticks=[], yticks=[], ec='white', abc=False)

ax5 = fig.subplot(gs[1,0], abc=True, abcloc='ul', number=5, fontsize=18)
ax5.format(xlabel='Noise / Signal', ylabel='Autocorrelation Time / Years',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=tau[np.arange(10,110,step=20)], 
             yminorticks=tau[np.arange(100,step=20)])
dur = ax5.contourf(x=sigma[:10], y=tau[np.arange(100,step=10)], cmap=cmap1,
                       z=np.transpose(sigma_tau_mean),levels=contour_levs, extend='both')

ax6 = fig.subplot(gs[1,1], abc=True, abcloc='ul', number=6, fontsize=18)
ax6.format(xlabel='Noise / Signal', ylabel='Greenland Stadial Slope / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=1000*GS_slope[np.arange(10,110,step=20)], 
              yminorlocator=1000*GS_slope[np.arange(100,step=20)])
GS = ax6.contourf(x=sigma[:10], y=1000*GS_slope[np.arange(100,step=10)], cmap=cmap1,
                       z=sigma_GS_mean,levels=contour_levs, extend='both')

ax7 = fig.subplot(gs[1,2], abc=True, abcloc='ul', number=7, fontsize=18)
ax7.format(xlabel='Noise / Signal', ylabel='Absolute Greenland Interstadial Slope \n / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=np.abs(1000*GIS_slope[np.arange(10,110,step=20)]),
              yminorlocator=0.3)
slope = ax7.contourf(x=sigma[:10], y=np.abs(1000*GIS_slope[np.arange(100,step=10)]), cmap=cmap1,
                       z=np.flip(sigma_GIS_mean,axis=0),levels=contour_levs, extend='both')

ax8 = fig.subplot(gs[1,3], abc=True, abcloc='ul', number=8, fontsize=18)
ax8.format(xlabel='Noise / Signal', ylabel='Transition Duration / Years',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=duration[np.arange(10,110,step=20)], yminorlocator=10)
dur = ax8.contourf(x=sigma[:10], y=duration[np.arange(100,step=10)], cmap=cmap1,
                       z=np.transpose(sigma_dur_mean),levels=contour_levs, extend='both')



#Extended Method, Annual Resolution

for i in range(10000):
    sigma[i] = 0.05 + 0.05 * (i%10)
    tau[i] = 0.5 + 0.5 * int((i%100)/10)
    GS_slope[i] = -1.8e-3 + 4e-4 * int((i%100)/10)
    GIS_slope[i] = 0.0 - 3e-4 * int((i%100)/10)
    duration[i] = 10.0 + 10.0 * int((i%100)/10)

df = pd.read_pickle('Data/annual_synthetics/Extended_Method/double_GS_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'GS_slope': GS_slope}
double_GS_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/annual_synthetics/Extended_Method/double_GIS_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'GIS_slope': GIS_slope}
double_GIS_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/annual_synthetics/Extended_Method/double_dur_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'duration': duration}
double_dur_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/annual_synthetics/Extended_Method/double_tau_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'tau': tau}
double_tau_df = pd.DataFrame(data=d)

double_GIS_df['sig_GIS'] = double_GIS_df['sigma'] + double_GIS_df['GIS_slope']
double_GS_df['sig_GS'] = double_GS_df['sigma'] + double_GS_df['GS_slope']
double_dur_df['sig_dur'] = double_dur_df['sigma'] + double_dur_df['duration']
double_tau_df['sig_tau'] = double_tau_df['sigma'] + 10*double_tau_df['tau']

sigma_GIS_mean = np.empty((10,10))

for i in range(100):
    sigma_GIS_mean[i%10,int(i/10)] = double_GIS_df.groupby('sig_GIS').mean()['time'].values[i]

sigma_GS_mean = np.empty((10,10))

for i in range(100):
    sigma_GS_mean[i%10,int(i/10)] = double_GS_df.groupby('sig_GS').mean()['time'].values[i]

sigma_dur_mean = np.empty((10,10))

for i in range(100):
    sigma_dur_mean[i%10,int(i/10)] = double_dur_df.groupby('sig_dur').mean()['time'].values[i]

sigma_tau_mean = np.empty((10,10))

for i in range(100):
    sigma_tau_mean[i%10,int(i/10)] = double_tau_df.groupby('sig_tau').mean()['time'].values[i]

ax = fig.subplot(gs[2,:], fontsize=24)
ax.format(title='Extended Method at Annual Resolution', titlepad=15, xticks=[], yticks=[], ec='white', abc=False)

ax9 = fig.subplot(gs[2,0], abc=True, abcloc='ul', number=9, fontsize=18)
ax9.format(xlabel='Noise / Signal', ylabel='Autocorrelation Time / Years',
              xticks=sigma[np.arange(1,11,step=2)], xminorticks=sigma[np.arange(10,step=2)], yticks=tau[np.arange(10,110,step=20)], 
              yminorticks=tau[np.arange(100,step=20)])
dur = ax9.contourf(x=sigma[:10], y=tau[np.arange(100,step=10)], cmap=cmap1,
                       z=np.transpose(sigma_tau_mean),levels=contour_levs, extend='both')

ax10 = fig.subplot(gs[2,1], abc=True, abcloc='ul', number=10, fontsize=18)
ax10.format(xlabel='Noise / Signal', ylabel='Greenland Stadial Slope / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorticks=sigma[np.arange(10,step=2)], yticks=1000*GS_slope[np.arange(10,110,step=20)], 
              yminorlocator=1000*GS_slope[np.arange(100,step=20)])
GS = ax10.contourf(x=sigma[:10], y=1000*GS_slope[np.arange(100,step=10)], cmap=cmap1,
                       z=sigma_GS_mean,levels=contour_levs, extend='both')

ax11 = fig.subplot(gs[2,2], abc=True, abcloc='ul', number=11, fontsize=18)
ax11.format(xlabel='Noise / Signal', ylabel='Absolute Greenland Interstadial Slope \n / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorticks=sigma[np.arange(10,step=2)],
              yticks=np.abs(1000*GIS_slope[np.arange(10,110,step=20)]),
              yminorlocator=0.3)
slope = ax11.contourf(x=sigma[:10], y=np.abs(1000*GIS_slope[np.arange(100,step=10)]), cmap=cmap1,
                       z=np.flip(sigma_GIS_mean,axis=0),levels=contour_levs, extend='both')

ax12 = fig.subplot(gs[2,3], abc=True, abcloc='ul', number=12, fontsize=18)
ax12.format(xlabel='Noise / Signal', ylabel='Transition Duration / Years',
             xticks=sigma[np.arange(1,11,step=2)], xminorticks=sigma[np.arange(10,step=2)],
              yticks=duration[np.arange(10,110,step=20)], yminorlocator=10)
dur = ax12.contourf(x=sigma[:10], y=duration[np.arange(100,step=10)], cmap=cmap1,
                       z=np.transpose(sigma_dur_mean),levels=contour_levs, extend='both')



#Extended Method, Decadal Resolution
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

df = pd.read_pickle('Data/decadal_synthetics/Extended_Method/double_GS_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'GS_slope': GS_slope}
double_GS_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/decadal_synthetics/Extended_Method/double_GIS_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'GIS_slope': GIS_slope}
double_GIS_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/decadal_synthetics/Extended_Method/double_dur_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'duration': duration}
double_dur_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/decadal_synthetics/Extended_Method/double_tau_pickle')
df = df.where(df['median_time']>300,other=np.nan)
df = df.where(df['median_time']<450,other=np.nan)
time = df['median_time'] - 400
d = {'time': time, 'sigma': sigma, 'tau': tau}
double_tau_df = pd.DataFrame(data=d)

double_GIS_df['sig_GIS'] = double_GIS_df['sigma'] + double_GIS_df['GIS_slope']
double_GS_df['sig_GS'] = double_GS_df['sigma'] + double_GS_df['GS_slope']
double_dur_df['sig_dur'] = double_dur_df['sigma'] + double_dur_df['duration']
double_tau_df['sig_tau'] = double_tau_df['sigma'] + double_tau_df['tau']

sigma_GIS_mean = np.empty((10,10))

for i in range(100):
    sigma_GIS_mean[i%10,int(i/10)] = double_GIS_df.groupby('sig_GIS').mean()['time'].values[i]

sigma_GS_mean = np.empty((10,10))

for i in range(100):
    sigma_GS_mean[i%10,int(i/10)] = double_GS_df.groupby('sig_GS').mean()['time'].values[i]

sigma_dur_mean = np.empty((10,10))

for i in range(100):
    sigma_dur_mean[i%10,int(i/10)] = double_dur_df.groupby('sig_dur').mean()['time'].values[i]

sigma_tau_mean = np.empty((10,10))

for i in range(100):
    sigma_tau_mean[i%10,int(i/10)] = double_tau_df.groupby('sig_tau').mean()['time'].values[i]

ax = fig.subplot(gs[3,:], fontsize=24)
ax.format(title='Extended Method at Decadal Resolution', titlepad=15, xticks=[], yticks=[], ec='white', abc=False)

ax13 = fig.subplot(gs[3,0], abc=True, abcloc='ul', number=13, fontsize=18)
ax13.format(xlabel='Noise / Signal', ylabel='Autocorrelation Time / Years',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=tau[np.arange(10,110,step=20)], 
             yminorticks=tau[np.arange(100,step=20)])
dur = ax13.contourf(x=sigma[:10], y=tau[np.arange(100,step=10)], cmap=cmap1,
                       z=np.transpose(sigma_tau_mean),levels=contour_levs, extend='both')

ax14 = fig.subplot(gs[3,1], abc=True, abcloc='ul', number=14, fontsize=18)
ax14.format(xlabel='Noise / Signal', ylabel='Greenland Stadial Slope / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=1000*GS_slope[np.arange(10,110,step=20)], 
              yminorlocator=1000*GS_slope[np.arange(100,step=20)])
GS = ax14.contourf(x=sigma[:10], y=1000*GS_slope[np.arange(100,step=10)], cmap=cmap1,
                       z=sigma_GS_mean,levels=contour_levs, extend='both')

ax15 = fig.subplot(gs[3,2], abc=True, abcloc='ul', number=15, fontsize=18)
ax15.format(xlabel='Noise / Signal', ylabel='Absolute Greenland Interstadial Slope \n / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=np.abs(1000*GIS_slope[np.arange(10,110,step=20)]),
              yminorlocator=0.3)
slope = ax15.contourf(x=sigma[:10], y=np.abs(1000*GIS_slope[np.arange(100,step=10)]), cmap=cmap1,
                       z=np.flip(sigma_GIS_mean,axis=0),levels=contour_levs, extend='both')

ax16 = fig.subplot(gs[3,3], abc=True, abcloc='ul', number=16, fontsize=18)
ax16.format(xlabel='Noise / Signal', ylabel='Transition Duration / Years',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=duration[np.arange(10,110,step=20)], yminorlocator=10)
dur = ax16.contourf(x=sigma[:10], y=duration[np.arange(100,step=10)], cmap=cmap1,
                       z=np.transpose(sigma_dur_mean),levels=contour_levs, extend='both')




# Add shapes to decadal for CCSM4: sigma, tau, GS slope, GIS slope, duration
tas = (0.06633234778696487, 13.649751415738818, 0.00016431508734689263, -0.0009404944834246113, 56.32925780771254)
pre = (0.1542468176758728, 7.610589558843324, 0.0003347760370749247, -0.0012734065625610814, 69.84867995820262)
ice = (0.06772528109764206, 13.522050331034816, 0.0001679453269451725, -0.0009241650626852406, 58.975578975895765)
AMOC = (0.09302892310996252, 20.317394958495132, 0.00028905757180442527, -0.0016966433176966102, 62.129599258809385)
NAO = (0.07601909051802441, 6.768150630222373, 0.0001115667911889947, -0.0007890345342570018, 58.206677821434035)

var_sigma = (tas[0],pre[0],ice[0],AMOC[0],NAO[0])
var_tau = (tas[1],pre[1],ice[1],AMOC[1],NAO[1])
var_GS_slope = 1000* np.array((tas[2],pre[2],ice[2],AMOC[2],NAO[2]))
var_GIS_slope = np.abs(1000* np.array((tas[3],pre[3],ice[3],AMOC[3],NAO[3])))
var_duration = (tas[4],pre[4],ice[4],AMOC[4],NAO[4])
markers = ('o', 'v', 's', '^', '*')
labels=('Temperature', 'Precipitation', 'Sea Ice', 'AMOC', 'NAO')

for i in range(5):
    ax5.scatter(var_sigma[i], var_tau[i], edgecolor='white', color='black', s=200, marker=markers[i], label=labels[i])
    ax6.scatter(var_sigma[i], var_GS_slope[i], edgecolor='white', color='black', s=200, marker=markers[i])
    ax7.scatter(var_sigma[i], var_GIS_slope[i], edgecolor='white', color='black', s=200, marker=markers[i])
    ax8.scatter(var_sigma[i], var_duration[i], edgecolor='white', color='black', s=200, marker=markers[i])
    ax13.scatter(var_sigma[i], var_tau[i], edgecolor='white', color='black', s=200, marker=markers[i])
    ax14.scatter(var_sigma[i], var_GS_slope[i], edgecolor='white', color='black', s=200, marker=markers[i])
    ax15.scatter(var_sigma[i], var_GIS_slope[i], edgecolor='white', color='black', s=200, marker=markers[i])
    ax16.scatter(var_sigma[i], var_duration[i], edgecolor='white', color='black', s=200, marker=markers[i])


#Add shapes to Annual for NGRIP: sigma, alpha, GS_slope, GIS_slope, Duration
dO18 = (0.35512853369656316, 0.3175718070302185, 0.0004178923548203365, -0.0005940306849159514, 64.40223930765913)
Ca = (0.17855488347920445, 0.5290282552023026, 0.0008834874030537089, -0.0004044796230253164, 52.225378632514996)
Na = (0.40684912829722486, 0.2865799122908226, 0.0004588594005226332, -0.00021193378517724617, 70.11026366895894)
Thickness = (0.3690816188789908, 0.08363809288258092, 0.000822067869923353, -0.0007619642437050181, 58.70786182623836)

var_sigma = (dO18[0],Ca[0],Na[0],Thickness[0])
var_tau = (-5.0/np.log(dO18[1]),-2.0/np.log(Ca[1]),-2.0/np.log(Na[1]),-2.0/np.log(Thickness[1]))
var_GS_slope = 1000* np.array((dO18[2],Ca[2],Na[2],Thickness[2]))
var_GIS_slope = np.abs(1000* np.array((dO18[3],Ca[3],Na[3],Thickness[3])))
var_duration = (dO18[4],Ca[4],Na[4],Thickness[4])
markers = ('<', '>', 'P', 'D',)
labels=('$\delta ^{18}O$', 'Ca$^{2+}$', 'Na$^+$', 'Annual Layer Thickness')

for i in range(4):
    ax1.scatter(var_sigma[i], var_tau[i], edgecolor='white', color='black', s=200, marker=markers[i], label=labels[i])
    ax2.scatter(var_sigma[i], var_GS_slope[i], edgecolor='white', color='black', s=200, marker=markers[i])
    ax3.scatter(var_sigma[i], var_GIS_slope[i], edgecolor='white', color='black', s=200, marker=markers[i])
    ax4.scatter(var_sigma[i], var_duration[i], edgecolor='white', color='black', s=200, marker=markers[i])
    ax9.scatter(var_sigma[i], var_tau[i], edgecolor='white', color='black', s=200, marker=markers[i])
    ax10.scatter(var_sigma[i], var_GS_slope[i], edgecolor='white', color='black', s=200, marker=markers[i])
    ax11.scatter(var_sigma[i], var_GIS_slope[i], edgecolor='white', color='black', s=200, marker=markers[i])
    ax12.scatter(var_sigma[i], var_duration[i], edgecolor='white', color='black', s=200, marker=markers[i])

colorbar = fig.colorbar(slope, loc='bottom')
colorbar.ax.tick_params(labelsize=24)
colorbar.ax.set_title('Bias / Years', fontsize=24)

fig.legend(loc='bottom',fontsize=20,ncols=9)
fig.save('Supplementary_Figures/Figure_C3')

''''Create Fig. C4 showing how we down-sample annual resolution
data to decadal resolution'''

low_tau = np.load('Data/low_tau_99.npy')
dec_low_tau = np.load('Data/dec_low_tau_99.npy')
high_both = np.load('Data/high_both_99.npy')
dec_high_both = np.load('Data/dec_high_both_99.npy')
fig = pplt.figure(figsize=(10,8))
axs = fig.add_subplots(nrows=2, ncols=1)
fig.format(fontsize=16)
axs[0].plot(range(800),low_tau)
axs[0].plot(np.arange(800,step=10),dec_low_tau,lw=3)
axs[0].format(title='Whiter Noise', xlabel='Time / Years', fontsize=16)
axs[1].plot(range(800),high_both)
axs[1].plot(np.arange(800,step=10),dec_high_both,lw=3)
axs[1].format(title='Redder Noise', xlabel='Time / Years', fontsize=16)
fig.save('Supplementary_Figures/Figure_C4')


'''Create Fig. C5, which shows a time series
for precipitation in the CCSM4 model. This makes it
clear that there is higher noise during stadials.'''

temp=np.load("Data/CCSM4/PRECIP/200.npy")
fig = pplt.figure(figsize = (8,5),fontsize=16)
ax = fig.subplot(xlabel='Model Year', ylabel='Precipitation / mm $s{^-1}$',fontsize=16)
ax.plot(np.arange(10*len(temp),step=10),temp,label='Precipitation')

fig.save('Supplementary_Figures/Figure_C5')


'''Create Fig. C6, showing the bias due to
separate stadial and interstadial noise regimes.'''

fig = pplt.figure(figsize=(10,8), sharey=True, sharex=False)
axs = fig.add_subplots(nrows=2,ncols=1)
fig.format(fontsize=16)
axs.format(ylim=(-14,-1))

df = pd.read_pickle('Data/decadal_synthetics/systematic_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
time = df['time']

sigma1 = np.empty(10000)
sigma2 = np.empty(10000)

for i in range(10000):
    sigma1[i] = 0.03+0.03*int((i%100)/10)
    sigma2[i] = 0.03+0.03*int(i/1000)


d = {'time': time-400, 'sigma1': sigma1, 'sigma2': sigma2}
plotting_df = pd.DataFrame(data=d)

axs[0].scatter(np.arange(0.03,0.30,0.03), plotting_df.groupby('sigma1').mean()['time'])
axs[0].format(ylabel='Bias / Years',xlabel='Stadial Noise / Signal')
axs[1].scatter(np.arange(0.03,0.30,0.03), plotting_df.groupby('sigma2').mean()['time'])
axs[1].format(ylabel='Bias Years',xlabel='Interstadial Noise / Signal')


sigma1_fit =  4.312053 -64.820610 * np.arange(0.03,0.30,0.03) -35.727530 * np.mean(np.arange(0.03,0.30,0.03)) \
            + 182.806446 * np.arange(0.03,0.30,0.03) * np.mean(np.arange(0.03,0.30,0.03))

sigma2_fit =  4.312053 -64.820610 * np.mean(np.arange(0.03,0.30,0.03)) -35.727530 * np.arange(0.03,0.30,0.03) \
            + 182.806446 * np.mean(np.arange(0.03,0.30,0.03)) * np.arange(0.03,0.30,0.03)


axs[0].plot(np.arange(0.03,0.30,0.03), sigma1_fit, color = 'black')
axs[1].plot(np.arange(0.03,0.30,0.03), sigma2_fit, color = 'black')
fig.save('Supplementary_Figures/Figure_C6')
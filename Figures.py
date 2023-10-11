import numpy as np
import pandas as pd
import proplot as pplt
import os
from scipy.stats import gaussian_kde
import sys
sys.path.append(os.path.join(sys.path[0],'Updated_Method'))
from transition_characterization import combined_transition
sys.path.append(os.path.join(sys.path[0],'Original_Method'))
from transition_characterization_flat import combined_transition_flat

'''Create Figure 1, showing the improvement resulting from our 
updated version of the method.'''

fig = pplt.figure(figsize=(12,7),
        share=False, includepanels=True)

axs = fig.add_subplots(nrows=2, ncols=3)

fig.format(
        fontsize=14,
        abc=True, abcloc='ul', titleloc='c', titleabove=False,
)

for i in range(3):
    if i == 0:
        traces = pd.read_pickle('Data/NGRIP/NGRIP_dO18_DO12')
        ylabel = '$\delta ^{18} O$ / ‰'
        xlabel = 'Years Before 1950'
        title = 'NGRIP $\delta ^{18} O$'
        yticks = [-46,-44,-42,-40,-38,-36]
        xticks = [47000,46900,46800,46700]

        NGRIP = pd.read_csv('Data/NGRIP/NGRIP_d18O_and_dust_5cm.csv',header=0)
        NGRIP = NGRIP.where(NGRIP['GICC05 age (yr b2k)'] > 46600)
        NGRIP = NGRIP.where(NGRIP['GICC05 age (yr b2k)'] < 47100)
        NGRIP_dO18 = NGRIP['Delta O18 (permil)'].dropna().values
        NGRIP_age = NGRIP['GICC05 age (yr b2k)'].dropna().values
        time = np.flip(47200-NGRIP_age)
        trans = np.flip(NGRIP_dO18)
        p5, p50, p95 = combined_transition(time, traces)
        time = np.flip(NGRIP_age)
        traces['t0'] = 47200 - traces['t0']
        traces['dt'] = -traces['dt']

    elif i == 1:
        start_year = 100
        end_year = 600
        traces = pd.read_pickle('Data/CCSM4/FIG_1/TAS_SLOPE_230ippm_0')
        CCSM_full = np.load('Data/CCSM4/FIG_1/TAS_ANN_230i.npy')
        xlabel = 'Model Year'
        ylabel = 'Temperature / K'
        title = 'CCSM4 Temperature'
        yticks = [220,225,230,235,240]
        xticks = [200,300,400,500]

        trans = CCSM_full[start_year:end_year]
        time = np.arange(start_year,end_year,dtype=float)
    
        p5, p50, p95 = combined_transition(time, traces)

    elif i == 2:
        start_year = 100
        end_year = 600
        traces = pd.read_pickle('Data/CCSM4/FIG_1/AMOC_SLOPE_230ippm_0')
        CCSM_full = np.load('Data/CCSM4/FIG_1/AMOC_ANN_230i.npy')
        xlabel = 'Model Year'
        ylabel = 'AMOC / Sv'
        title = 'CCSM4 AMOC'
        yticks = [15,20,25,30]
        xticks = [200,300,400,500]

        trans = CCSM_full[start_year:end_year]
        time = np.arange(start_year,end_year,dtype=float)
    
        p5, p50, p95 = combined_transition(time, traces)

    reduced_time = time

    ax = axs[1,i]
    ax.format(xlabel = xlabel, ylabel = ylabel, xticks=xticks, yticks = yticks, fontsize=14)
    px = ax.panel_axes("b", width='3em', share=True)
    px.format(fontsize=14)

    ax.plot(reduced_time, trans, lw=0.5)
    ax.plot(reduced_time, p50, color='k', lw=1.2,
                label='50th percentile', zorder=12)
    ax.plot(reduced_time, p5, color='slategray', lw=0.8)
    ax.plot(reduced_time, p95, color='slategray',  lw=0.8)
    ax.fill_between(x=reduced_time, y1=p5,
                    y2=p95, color='C1', alpha=.8,
                    zorder=10, label='90th percentile range')

    px.hist(traces['t0'].values,bins=np.sort(time),density=True)
    px.hist(traces['t0'].values + traces['dt'].values,bins=np.sort(time),density=True)


    if i == 0:
        traces = pd.read_pickle('Data/NGRIP/NGRIP_dO18_DO12_flat')

        NGRIP = pd.read_csv('Data/NGRIP/NGRIP_d18O_and_dust_5cm.csv',header=0)
        NGRIP = NGRIP.where(NGRIP['GICC05 age (yr b2k)'] > 46600)
        NGRIP = NGRIP.where(NGRIP['GICC05 age (yr b2k)'] < 47100)
        NGRIP_dO18 = NGRIP['Delta O18 (permil)'].dropna().values
        NGRIP_age = NGRIP['GICC05 age (yr b2k)'].dropna().values
        time = np.flip(47200-NGRIP_age)
        trans = np.flip(NGRIP_dO18)
        p5, p50, p95 = combined_transition_flat(time, traces)
        time = np.flip(NGRIP_age)
        traces['t0'] = 47200 - traces['t0']
        traces['dt'] = -traces['dt']

    elif i == 1:
 
        traces = pd.read_pickle('Data/CCSM4/FIG_1/TAS_FLAT_230ippm_0')
        CCSM_full = np.load('Data/CCSM4/FIG_1/TAS_ANN_230i.npy')

        trans = CCSM_full[start_year:end_year]
        time = np.arange(start_year,end_year,dtype=float)
    
        p5, p50, p95 = combined_transition_flat(time, traces)

    elif i == 2:
 
        traces = pd.read_pickle('Data/CCSM4/FIG_1/AMOC_FLAT_230ippm_0')
        CCSM_full = np.load('Data/CCSM4/FIG_1/AMOC_ANN_230i.npy')

        trans = CCSM_full[start_year:end_year]
        time = np.arange(start_year,end_year,dtype=float)
    
        p5, p50, p95 = combined_transition_flat(time, traces)


    ax = axs[0,i]
    ax.format(ylabel = ylabel, xticks = xticks, yticks = yticks, title = title)
    px = ax.panel_axes("b", width='3em', share=True)
    px.format(fontsize=14)

    ax.plot(time, trans, lw=0.5)
    ax.plot(time, p50, color='k', lw=1.2,
                label='50th percentile', zorder=12)
    ax.plot(time, p5, color='slategray', lw=0.8)
    ax.plot(time, p95, color='slategray',  lw=0.8)
    ax.fill_between(x=time, y1=p5,
                    y2=p95, color='C1', alpha=.8,
                    zorder=10, label='90th percentile range')

    px.hist(traces['t0'].values,bins=np.sort(time),density=True)
    px.hist(traces['t0'].values + traces['dt'].values,bins=np.sort(time),density=True)

fig.save('Figures/Figure_1')


'''Create Figure 2, highlighting the key parameters that
lead to bias in the ramp fitting method.'''

fig = pplt.figure(share=False, figsize=(12,5), includepanels=True)
axs = fig.add_subplots(nrows=1, ncols=2, wratios=(2, 1))
fig.format(fontsize=16, 
    abc=True, abcloc='ul', titleloc='uc', titleabove=False)


trans = np.load('Data/annual_synthetics/example_synthetic.npy')
time = np.arange(800,step=1,dtype=float)
    
traces = pd.read_pickle("Data/annual_synthetics/example_traces")

p5, p50, p95 = combined_transition(time, traces)

#reduced_time = time
reduced_time = np.arange(300.0,550.0,dtype='float')

ax = axs[0]
ax.format(xlabel = 'Year')
px = ax.panel_axes("b", width='3em', share=True)
    
ax.plot(reduced_time, trans[int(reduced_time[0]-time[0]):int(reduced_time[-1]-time[0]+1)], lw=0.5)
ax.plot(reduced_time, p50[int(reduced_time[0]-time[0]):int(reduced_time[-1]-time[0]+1)], color='k', lw=1.2, label='50th percentile', zorder=12)
ax.plot(reduced_time, p5[int(reduced_time[0]-time[0]):int(reduced_time[-1]-time[0]+1)], color='slategray', lw=0.8)
ax.plot(reduced_time, p95[int(reduced_time[0]-time[0]):int(reduced_time[-1]-time[0]+1)], color='slategray',  lw=0.8)
ax.fill_between(x=reduced_time, y1=p5[int(reduced_time[0]-time[0]):int(reduced_time[-1]-time[0]+1)],
                    y2=p95[int(reduced_time[0]-time[0]):int(reduced_time[-1]-time[0]+1)], color='C1', alpha=.8,
                    zorder=10, label='90th percentile range')

px.hist(traces['t0'].values, bins=np.arange(300.0,550.0,dtype='float'), density=True)
px.hist(traces['t0'].values+traces['dt'].values, bins=np.arange(300.0,550.0,dtype='float'), density=True)
px.format(fontsize=16)

axs[1].axvline(400.0,color='black', label = 'True Onset Time')
axs[1].axvline(np.mean(traces['t0']),color='black',linestyle='--', label = 'Mean Estimated Onset Time')
axs[1].format(ylabel = 'Probability Density', xlabel='Year', ylim=(0.00,0.082))
leg = axs[1].legend(fontsize=14, ncols=1, alpha=1.0)
axs[1].hist(traces['t0'], bins=np.arange(374,404,step=1), density=True)

fig.save('Figures/Figure_2')



'''Create Figure 3, showing the bias due to the different parameters.'''

'''Bias Interaction Plots With 100 realizations of the
synthetic transition for each combination of parameters.'''

#Decadal Data
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

df = pd.read_pickle('Data/decadal_synthetics/double_GS_pickle')
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
time = df['time'] - 400
d = {'time': time, 'sigma': sigma, 'GS_slope': GS_slope}
double_GS_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/decadal_synthetics/double_GIS_pickle')
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
time = df['time'] - 400
d = {'time': time, 'sigma': sigma, 'GIS_slope': GIS_slope}
double_GIS_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/decadal_synthetics/double_dur_pickle')
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
time = df['time'] - 400
d = {'time': time, 'sigma': sigma, 'duration': duration}
double_dur_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/decadal_synthetics/double_tau_pickle')
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
time = df['time'] - 400
d = {'time': time, 'sigma': sigma, 'tau': tau}
double_tau_df = pd.DataFrame(data=d)


fig = pplt.figure(refwidth=4, share=False)
axs = fig.add_subplots(ncols=4, nrows=2)
fig.format(abc=True, abcloc='ul', fontsize=18)
axs.format(fontsize=18)

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

contour_levs = [-11,-7,-4,-2,-1,0,1,2,4,7,11]

cmap1 = pplt.Colormap('RdBu_r')

axs[4].format(xlabel='Noise / Signal', ylabel='Autocorrelation Time / Years',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=tau[np.arange(10,110,step=20)], 
             yminorticks=tau[np.arange(100,step=20)])
dur = axs[4].contourf(x=sigma[:10], y=tau[np.arange(100,step=10)], cmap=cmap1,
                       z=np.transpose(sigma_tau_mean),levels=contour_levs, extend='both')

axs[5].format(xlabel='Noise / Signal', ylabel='Greenland Stadial Slope / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=1000*GS_slope[np.arange(10,110,step=20)], 
              yminorlocator=1000*GS_slope[np.arange(100,step=20)])
GS = axs[5].contourf(x=sigma[:10], y=1000*GS_slope[np.arange(100,step=10)], cmap=cmap1,
                       z=sigma_GS_mean,levels=contour_levs, extend='both')

axs[6].format(xlabel='Noise / Signal', ylabel='Absolute Greenland Interstadial Slope \n / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=np.abs(1000*GIS_slope[np.arange(10,110,step=20)]),
              yminorlocator=0.3)
slope = axs[6].contourf(x=sigma[:10], y=np.abs(1000*GIS_slope[np.arange(100,step=10)]), cmap=cmap1,
                       z=np.flip(sigma_GIS_mean,axis=0),levels=contour_levs, extend='both')

axs[7].format(xlabel='Noise / Signal', ylabel='Transition Duration / Years',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=duration[np.arange(10,110,step=20)], yminorlocator=10)
dur = axs[7].contourf(x=sigma[:10], y=duration[np.arange(100,step=10)], cmap=cmap1,
                       z=np.transpose(sigma_dur_mean),levels=contour_levs, extend='both')


colorbar = fig.colorbar(slope, loc='bottom')
colorbar.ax.tick_params(labelsize=18)
colorbar.ax.set_title('Bias / Years', fontsize=18)

#Annual data

for i in range(10000):
    sigma[i] = 0.04 + 0.04 * (i%10)
    tau[i] = 0.4 + 0.4 * int((i%100)/10)
    GS_slope[i] = -1.8e-3 + 4e-4 * int((i%100)/10)
    GIS_slope[i] = 0.0 - 3e-4 * int((i%100)/10)
    duration[i] = 10.0 + 10.0 * int((i%100)/10)

df = pd.read_pickle('Data/annual_synthetics/double_GS_pickle')
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
time = df['time'] - 400
d = {'time': time, 'sigma': sigma, 'GS_slope': GS_slope}
double_GS_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/annual_synthetics/double_GIS_pickle')
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
time = df['time'] - 400
d = {'time': time, 'sigma': sigma, 'GIS_slope': GIS_slope}
double_GIS_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/annual_synthetics/double_dur_pickle')
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
time = df['time'] - 400
d = {'time': time, 'sigma': sigma, 'duration': duration}
double_dur_df = pd.DataFrame(data=d)

df = pd.read_pickle('Data/annual_synthetics/double_tau_pickle')
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
time = df['time'] - 400
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

cmap1 = pplt.Colormap('RdBu_r')

axs[0].format(xlabel='Noise / Signal', ylabel='Autocorrelation Time / Years',
              xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=tau[np.arange(10,110,step=20)], 
              yminorticks=tau[np.arange(100,step=20)])
dur = axs[0].contourf(x=sigma[:10], y=tau[np.arange(100,step=10)], cmap=cmap1,
                       z=np.transpose(sigma_tau_mean),levels=contour_levs, extend='both')

axs[1].format(xlabel='Noise / Signal', ylabel='Greenland Stadial Slope / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=1000*GS_slope[np.arange(10,110,step=20)], 
              yminorlocator=1000*GS_slope[np.arange(100,step=20)])
GS = axs[1].contourf(x=sigma[:10], y=1000*GS_slope[np.arange(100,step=10)], cmap=cmap1,
                       z=sigma_GS_mean,levels=contour_levs, extend='both')

axs[2].format(xlabel='Noise / Signal', ylabel='Absolute Greenland Interstadial Slope \n / $Kiloyears^{-1}$',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=np.abs(1000*GIS_slope[np.arange(10,110,step=20)]),
              yminorlocator=0.3)
slope = axs[2].contourf(x=sigma[:10], y=np.abs(1000*GIS_slope[np.arange(100,step=10)]), cmap=cmap1,
                       z=np.flip(sigma_GIS_mean,axis=0),levels=contour_levs, extend='both')

axs[3].format(xlabel='Noise / Signal', ylabel='Transition Duration / Years',
             xticks=sigma[np.arange(1,11,step=2)], xminorlocator=0.02, yticks=duration[np.arange(10,110,step=20)], yminorlocator=10)
dur = axs[3].contourf(x=sigma[:10], y=duration[np.arange(100,step=10)], cmap=cmap1,
                       z=np.transpose(sigma_dur_mean),levels=contour_levs, extend='both')

# decadal: sigma, tau, GS slope, GIS slope, duration
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
    axs[4].scatter(var_sigma[i], var_tau[i], edgecolor='white', color='black', s=200, marker=markers[i], label=labels[i])
    axs[5].scatter(var_sigma[i], var_GS_slope[i], edgecolor='white', color='black', s=200, marker=markers[i])
    axs[6].scatter(var_sigma[i], var_GIS_slope[i], edgecolor='white', color='black', s=200, marker=markers[i])
    axs[7].scatter(var_sigma[i], var_duration[i], edgecolor='white', color='black', s=200, marker=markers[i])


fig.legend(loc='bottom',fontsize=18,ncols=5, mode='expand')

fig.save('Figures/Figure_3')



'''Create Figure 4, showing the uncorrected and corrected time lags
in the CCSM4 model, relative to temperature.'''

rng = np.random.default_rng(12345)
runs = ('185','200','210','220','225','230i') #CO2 values of CCSM4 simulations
vars = ('PRECIP','SEA_ICE','AMOC', 'NAO') #variables we compare to TEMP
nlags = 100000 #size of distribution of time lag
lags = np.empty((len(vars),nlags))
corr_lags = np.empty((len(vars),nlags))

for m, var in enumerate(vars):
    samples = np.empty((2000,19))
    j=0
    for run in runs:
        for i in range(8):
            if os.path.isfile('Data/CCSM4/TEMP/traces/'+str(run)+'ppm_'+str(i)) == True:
                var_t0 = pd.read_pickle('Data/CCSM4/'+var+'/traces/'+str(run)+'ppm_'+str(i))['t0']
                temp = pd.read_pickle('Data/CCSM4/TEMP/traces/'+str(run)+'ppm_'+str(i))['t0']
                samples[:,j] = np.flip(var_t0) - temp #comparing each variable to temperature
                j+=1
            else:
                break

    for k in range(nlags):
        sample = np.empty(19)
        for l in range(19):
            sample[l] = samples[rng.integers(low=0, high=2000),l] #sampling randomly from the distribution of each individual time lag
        lags[m,k] = np.mean(sample) #taking the sample mean

fig = pplt.figure(figsize=(8,4),)
ax = fig.add_subplot(ylabel='Sample Mean Time Lag / Years')
fig.format(fontsize=14,)

d = {'Precipitation': lags[0,:], 'Sea Ice': lags[1,:], 'AMOC': lags[2,:], 'NAO': lags[3,:]}
lags_df = pd.DataFrame(data=d)

vars = ('PRECIP_LIKE', 'SEA_ICE_LIKE', 'AMOC_LIKE', 'NAO_LIKE') #analogous transitions to calculate bias
for m, var in enumerate(vars):
    samples = np.empty((2000,1000))
    for run in range(1000):
        var_like_t0 = pd.read_pickle('Data/CCSM4/'+str(var)+'/traces/'+str(run))['t0']
        temp_like = pd.read_pickle('Data/CCSM4/TEMP_LIKE/traces/'+str(run))['t0']
        samples[:,run] = var_like_t0 - temp_like #bias for each variable

    bias = np.mean(samples.flatten())
    corr_lags[m,:] = lags[m,:] - bias #bias correction



d = {'Precipitation': corr_lags[0,:], 'Sea Ice': corr_lags[1,:], 'AMOC': corr_lags[2,:], 'NAO': corr_lags[3,:]}
corr_lags_df = pd.DataFrame(data=d)

ax.violin(lags_df) # violins of uncorrected lags
ax.violinplot(corr_lags_df, cycle=False, # transparent violins of corrected lags
                      facecolor=((1, 1, 1, 0),(1, 1, 1, 0),(1, 1, 1, 0),(1, 1, 1, 0)), alpha=np.array((0.99,0.99,0.99,0.99)))
fig.save('Figures/Figure_4')


'''Create FIgure 5, showing the time lags in the NGRIP ice core.'''

rng = np.random.default_rng(12345)
vars = ('Na', 'dO18', 'Thickness')
nlags = 100000
lags = np.empty((len(vars),nlags))
corr_lags = np.empty((len(vars),nlags))

for m, var in enumerate(vars):

    aerosol_highres_df = pd.read_csv('Data/NGRIP/Erhardt_NGRIP_aerosol_highres.tab', header = 18, sep='\t', engine='python')
    events = aerosol_highres_df['Ageprof dat des'].unique()

    samples = np.empty((2000,16))
    j = 0
    for i in [0,1,4,7,9,10,12,13,14,15,16,17,18,20,21,22]:
        event = events[int(i)]
        temp_var = pd.read_pickle('Data/NGRIP/'+str(var)+'/traces/'+str(event))['t0'].values
        Ca = pd.read_pickle('Data/NGRIP/Ca/traces/'+str(event))['t0'].values
        samples[:,j] = np.flip(temp_var) - Ca
        j += 1
            
    for k in range(nlags):
        sample = np.empty(16)
        for l in range(16):
            sample[l] = samples[rng.integers(low=0, high=2000),l]
        lags[m,k] = np.mean(sample)

fig = pplt.figure(figsize=(8,4),)
ax = fig.add_subplot(ylabel='Sample Mean Time Lag / Years')
fig.format(fontsize=14,)

d = {'Na': lags[0,:], 'dO18': lags[1,:], 'Thickness': lags[2,:]}
lags_df = pd.DataFrame(data=d)


nruns = 1000
runs = range(nruns)

for m, var in enumerate(vars):
    j = 0
    fail_count = 0
    samples = np.empty((2000,nruns))

    for run in runs:
        var1_t0 = pd.read_pickle('Data/NGRIP/'+str(var)+'_like/traces/'+str(run))['t0'].values
        Ca_like = pd.read_pickle('Data/NGRIP/Ca_like/traces/'+str(run))['t0'].values
        if np.mean(var1_t0)-250 > 50:
            fail_count += 1
        if np.mean(var1_t0)-250 < -100:
            fail_count += 1
        else:
            samples[:,j] = var1_t0-Ca_like
            j += 1

    bias = np.mean(samples.flatten())
    corr_lags[m,:] = lags[m,:] - bias

d = {'Na': corr_lags[0,:], '$\delta ^{18} O$': corr_lags[1,:], 'Thickness': corr_lags[2,:]}
corr_lags_df = pd.DataFrame(data=d)

ax.violin(lags_df, cycle=True)

ax.violinplot(corr_lags_df, cycle=False,
                      facecolor=((1, 1, 1, 0),(1, 1, 1, 0),(1, 1, 1, 0)), alpha=np.array((0.99,0.99,0.99)))
fig.save('Figures/Figure_5')



'''Create Figure B1, visualizing the hypothesis test.'''


nruns = 1000
runs = range(nruns)
rng = np.random.default_rng(12345)

samples = np.empty((2000,nruns))

for run in runs:
    pre_like = pd.read_pickle('Data/CCSM4/PRECIP_LIKE/traces/'+str(run))['t0']
    tas_like = pd.read_pickle('Data/CCSM4/TEMP_LIKE/traces/'+str(run))['t0']
    samples[:,run] = pre_like - tas_like

null = np.empty(100000)
for k in range(100000):
    sample = np.empty(19)
    for l in range(19):
        sample[l] = samples[rng.integers(low=0, high=2000),rng.integers(low=0, high=nruns)]
    null[k]=np.mean(sample)

runs = ('185','200','210','220','225','230i')
samples = np.empty((2000,19))

j=0
for run in runs:
    for i in range(4):
        if os.path.isfile('Data/CCSM4/TEMP/traces/'+str(run)+'ppm_'+str(i)) == True:
            PRE = pd.read_pickle('Data/CCSM4/PRECIP/traces/'+str(run)+'ppm_'+str(i))['t0']
            TAS = pd.read_pickle('Data/CCSM4/TEMP/traces/'+str(run)+'ppm_'+str(i))['t0']
            samples[:,j] = PRE - TAS
            j+=1
        else:
            break

obs = np.empty(100000)
for k in range(100000):
    for l in range(19):
        sample[l] = samples[rng.integers(low=0, high=2000),l]
    obs[k]=np.mean(sample)


pval = np.sum(np.abs(obs-np.mean(null))<np.abs(null-np.mean(null)))/(100000)


fig = pplt.figure(figsize=(10,6), fontsize=16)
ax = fig.add_subplot(xlabel='Sample Mean Time Lag / Years', ylabel = 'Probability Density', fontsize=16)

kde = gaussian_kde(obs)
ax.plot(np.arange(-30,8,0.01),kde.evaluate(np.arange(-30,8,0.01)), label = 'Observed')

null_kde = gaussian_kde(null)
ax.plot(np.arange(-30,8,0.01),null_kde.evaluate(np.arange(-30,8,0.01)), label = 'Empirical Null Distribution')

thresholds = np.quantile(null,(0.025,0.975))
ax.axvline(thresholds[0],linestyle='--',color='black',label='Significance Thresholds')
ax.axvline(thresholds[1],linestyle='--',color='black')

fig.legend(loc='bottom', fontsize=16, mode='expand')
fig.save('Figures/Figure_B1')
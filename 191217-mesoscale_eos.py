#! /usr/bin/env python

import read_pve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import uncertainties as uct
import pandas as pd
import os

from lmfit import Parameters, conf_interval, Minimizer, conf_interval2d
from lmfit.models import LinearModel
from lmfit.printfuncs import report_fit, report_ci
from scipy.stats import t

def retrieve_data(directory):
    '''
    Input: directory path to .PVE files.
    Output: nested dictionary with all data.
    '''
    files = read_pve.get_files(directory)
    parameters_dict = read_pve.create_param_dict(files)
    
    return parameters_dict

def bm2(params, x, data=None, uncert=None):
    #Isothermal 2nd order Birch-Murnaghan equation of state

    v = x
    vo = params['vo']
    ko = params['ko']
    
    p = 3.*(ko/2.)*(((vo/v)**(7./3.))-((vo/v)**(5./3.)))

    if data is None:
        return p

    if uncert is None:
        return (data-p)

    sigP = uncert[:,0]
    sigV = uncert[:,1]
    
    #weight
    w = 1.0/((sigP**2.0)+((sigV**2.0)*((ko/vo)**2.0)))

    return w*(data-p)
    
def bm3(params, x, data=None, uncert=None):
    #Isothermal 3rd order Birch-Murnaghan equation of state

    v = x
    vo = params['vo']
    ko = params['ko']
    kp = params['kp']

    p = 3.*(ko/2.)*(((vo/v)**(7./3.))-((vo/v)**(5./3.)))*(1.+((3./4.)*(kp-4.)*(((vo/v)**(2./3.))-1.)))

    if data is None:
        return p

    if uncert is None:
        return (data-p)

    sigP = uncert[:,0]
    sigV = uncert[:,1]
    
    #weight
    w = 1.0/((sigP**2.0)+((sigV**2.0)*((ko/vo)**2.0)))

    return w*(data-p)

def vinet(params, x, data=None, uncert=None):
    v = x
    vo = params['vo']
    ko = params['ko']
    kp = params['kp']

    f = (v/vo)**(1./3.)
    
    p = 3.*ko*((1.-f)/(f**2.0))*np.exp((3./2.)*(kp-1.)*(1.-f))
    
    if data is None:
        return p

    if uncert is None:
        return (data-p)

    sigP = uncert[:,0]
    sigV = uncert[:,1]
    
    #weight
    w = 1.0/((sigP**2.0)+((sigV**2.0)*((ko/vo)**2.0)))

    return w*(data-p)

def calc_k(results,volumes):

    vo = results['vo']
    ko = results['ko']
    kp = results['kp']

    x = vo/volumes
    f = 0.5*((x**(2./3.))-1.0)

    k = ko*((1.+(2.*f))**(5./2.))*(1.+(((3.*kp)-5.)*f)+((27./2.)*(kp-4.)*(f**2.)))
    
    return k

def k_plot(k,out_params,PVs,interval, plot_conf_band, fix_vo):
    
    pressures = PVs[:,0]
    
    plt.figure(2)
    plt.plot(pressures,k, label='Bulk Modulus (K) [GPa]')
    plt.xlim(np.min(pressures),np.max(pressures))
    plt.xlabel('Pressure (GPa)',fontweight='bold')
    plt.ylabel('K (GPa)',fontweight='bold')
    plt.tick_params(direction='in',bottom=1,top=1,left=1,right=1)
    
    #test_data = np.loadtxt('/Users/johnlazarz/python/scripts/mesoscale-eos/test_PK.csv',delimiter=',')
    #plt.plot(test_data[:,0],test_data[:,1], label = 'eos_fit')
    

    if plot_conf_band == True:
        pcov = out_params.covar
        
        cp_bands = confidence_prediction_bands(out_params, PVs[:,1], pcov, interval, calc_k, fix_vo)
    
        plt.plot(PVs[:,0],cp_bands[0],linestyle='--', linewidth=0.75, color='r', label='%s%% confidence bands' % (interval*100.0))
        plt.plot(PVs[:,0],cp_bands[1],linestyle='--', linewidth=0.75, color='r')
    plt.legend()

    return 
    
def bm3_calc_kp(results,volumes):
    
    k = calc_k(results,volumes)
    
    vo = results['vo']
    ko = results['ko']
    kpo = results['kp']

    x = vo/volumes
    f = 0.5*((x**(2./3.))-1.0)

    kp = (ko/k)*((1.+(2.*f))**(5./2.))*(kpo+(((16.*kpo)-(143./3.))*f)+((81./2.)*(kpo-4.)*(f**2.)))

    return kp

def kp_plot(kp,out_params,PVs,interval, plot_conf_band, fix_vo):
    
    pressures = PVs[:,0]
    
    plt.figure(3)
    plt.plot(pressures,kp, label='Bulk Modulus (K) [GPa]')
    plt.xlim(np.min(pressures),np.max(pressures))
    plt.xlabel('Pressure (GPa)',fontweight='bold')
    plt.ylabel('K\'',fontweight='bold')
    plt.tick_params(direction='in',bottom=1,top=1,left=1,right=1)

    if plot_conf_band == True:
        pcov = out_params.covar
        
        cp_bands = confidence_prediction_bands(out_params, PVs[:,1], pcov, interval, bm3_calc_kp, fix_vo)
    
        plt.plot(PVs[:,0],cp_bands[0],linestyle='--', linewidth=0.75, color='r', label='%s%% confidence bands' % (interval*100.0))
        plt.plot(PVs[:,0],cp_bands[1],linestyle='--', linewidth=0.75, color='r')
    plt.legend()

    return
    
def bm3_calc_kpp(out_params):
    #BM3 implied 2nd derivative of bulk modulus (Anderson 1995)

    results = out_params.params.valuesdict()
        
    ko = results['ko']
    kp = results['kp']
    
    kpp = (-1.0/ko)*(((2.0-kp)*(4.0-kp))+(35.0/9.0))
    
    return kpp

    
def perform_fit(exp_volumes,exp_pressures, starting_params, interval, fix_vo = False, bm2 = False):

    volumes = exp_volumes[:,0]
    sigV = exp_volumes[:,1]
    p = exp_pressures[:,0]
    sigP = exp_pressures[:,1]

    uncert = np.column_stack((sigV,sigP))
    
    #initial starting parameters
    fit_params = Parameters()
    fit_params.add('vo', value = starting_params['vo'], min=0.0)
    fit_params.add('ko', value = starting_params['ko'], min=0.0)
    fit_params.add('kp', value = starting_params['kp'])

    if fix_vo:
        fit_params['vo'].vary = False
    if bm2:
        fit_params['kp'].vary = False

    mini = Minimizer(bm3, fit_params, fcn_args=(volumes,), fcn_kws={'data': p,\
                                                          'uncert': uncert})
    out1 = mini.minimize(method='Nedler')

    out2 = mini.minimize(method='leastsq', params=out1.params)

    results = out2.params.valuesdict()

    plot_volumes = np.linspace((np.min(volumes)-10),results['vo'],1000)
    fit = bm3(out2.params,plot_volumes)

    report_fit(out2, show_correl=True, min_correl=0.001)
    
    PVs = np.column_stack((fit,plot_volumes))

    ci = conf_interval(mini,out2)
    report_ci(ci)

    return ci, mini, out2, PVs

def plot_fit(volumes,pressures,out_params,interval,plot_conf_band,sec_order,fix_vo,sample):

    plt.figure(1)

    sigV = volumes[:,1]
    sigP = pressures[:,1]
    volumes = volumes[:,0]
    pressures = pressures[:,0]
    
    results = out_params.params.valuesdict()

    plot_volumes = np.linspace((np.min(volumes)-10),results['vo'],1000)
    fit = bm3(out_params.params,plot_volumes)
    
    plt.plot(fit,plot_volumes,'b')
    plt.errorbar(pressures, volumes, fmt='o', xerr=sigP, yerr=sigV,\
             label=sample,alpha=1.0,capsize = 5., fillstyle='none') 
    plt.xlim(np.min(fit),np.max(fit))

    plt.xlabel('Pressure (GPa)',fontweight='bold')
    plt.ylabel('Volume ($\mathbf{\AA^3}$)',fontweight='bold')
    plt.tick_params(direction='in',bottom=1,top=1,left=1,right=1)
    plt.tight_layout()

    if plot_conf_band == True:
        pcov = out_params.covar
    
        if sec_order == True:
            f = bm2
        if sec_order == False:
            f = bm3    
    
        cp_bands = confidence_prediction_bands(out_params, plot_volumes, pcov, interval, f, fix_vo)
    
        plt.plot(cp_bands[0],plot_volumes,linestyle='--', linewidth=0.75, color='r', label='%s%% confidence bands' % (interval*100.0))
        plt.plot(cp_bands[1],plot_volumes,linestyle='--', linewidth=0.75, color='r')

    plt.legend()
    #plt.savefig('PV-plot_comparison.eps',dpi=1800,bbox_inches='tight')
    return cp_bands

def confidence_intervals(ci,mini,out2,fix_vo,bm2):
    plt.figure(4)
    results = out2.params.valuesdict()

    ko_unc = float(str(out2.params['ko']).split()[4].replace(',',''))
    
    if fix_vo:
        kp_unc = out2.params['kp'].stderr
        fig, axes = plt.subplots(1, 1)
        cx, cy, grid = conf_interval2d(mini, out2, 'ko', 'kp', 80, 80)
        ctp = axes.contourf(cx, cy, grid, np.linspace(0, 1, 100))
        axes.errorbar(results['ko'],results['kp'], xerr=ko_unc, yerr=kp_unc,\
                        linestyle='None', marker='None', label=sample,color='white', capsize = 3.,elinewidth=1.5)
        #fig.colorbar(ctp, ax=axes[1])
        axes.set_xlabel('$\mathbf{K_0 (GPa)}$')
        axes.set_ylabel('$\mathbf{K_0}$\'')
        axes.tick_params(direction='in',bottom=1,top=1,left=1,right=1)
        
    elif bm2:
        vo_unc = out2.params['vo'].stderr
        fig, axes = plt.subplots(1, 1)
        cx, cy, grid = conf_interval2d(mini, out2, 'vo', 'ko', 80, 80)
        ctp = axes.contourf(cx, cy, grid, np.linspace(0, 1, 100))
        axes.errorbar(results['vo'],results['ko'], xerr=vo_unc, yerr=ko_unc,\
                        linestyle='None', marker='None', label=sample,color='white', capsize = 3.,elinewidth=1.5)
        axes.set_xlabel('$\mathbf{V_0 (\AA^3)}$')
        axes.set_ylabel('$\mathbf{K_0 (GPa)}$')
        axes.tick_params(direction='in',bottom=1,top=1,left=1,right=1)
    else:
        vo_unc = out2.params['vo'].stderr
        kp_unc = out2.params['kp'].stderr
        # plot confidence intervals
        fig, axes = plt.subplots(2, 2)
        cx, cy, grid = conf_interval2d(mini, out2, 'vo', 'kp', 80, 80)
        ctp = axes[1,0].contourf(cx, cy, grid, np.linspace(0, 1, 100))
        axes[1,0].errorbar(results['vo'],results['kp'], xerr=vo_unc, yerr=kp_unc,\
                        linestyle='None', marker='None', label=sample,color='white', capsize = 3.,elinewidth=1.5)
        axes[1,0].set_xlabel('$\mathbf{V_0 (\AA^3)}$')
        axes[1,0].set_ylabel('$\mathbf{K_0}$\'')
        axes[1,0].tick_params(direction='in',bottom=1,top=1,left=1,right=1)
        
        cx, cy, grid = conf_interval2d(mini, out2, 'vo', 'ko', 80, 80)
        ctp = axes[0,0].contourf(cx, cy, grid, np.linspace(0, 1, 100))
        #axes[0,0].set_xlabel('$\mathbf{V_0 (\AA^3)}$')
        axes[0,0].errorbar(results['vo'],results['ko'], xerr=vo_unc, yerr=ko_unc,\
                        linestyle='None', marker='None', label=sample,color='white', capsize = 3.,elinewidth=1.5)
        axes[0,0].set_ylabel('$\mathbf{K_0 (GPa)}$')
        axes[0,0].tick_params(direction='in',bottom=1,top=1,left=1,right=1)
    
        cx, cy, grid = conf_interval2d(mini, out2, 'ko', 'kp', 80, 80)
        ctp = axes[1,1].contourf(cx, cy, grid, np.linspace(0, 1, 100))
        #fig.colorbar(ctp, ax=axes[1])
        axes[1,1].errorbar(results['ko'],results['kp'], xerr=ko_unc, yerr=kp_unc,\
                        linestyle='None', marker='None', label=sample,color='white', capsize = 3.,elinewidth=1.5)
        axes[1,1].set_xlabel('$\mathbf{K_0 (GPa)}$')
        #axes[1,1].set_ylabel('$\mathbf{K_0}$\'')
        axes[1,1].tick_params(direction='in',bottom=1,top=1,left=1,right=1)
    
        axes[0,1].set_axis_off()
    return

def confidence_prediction_bands(model, x_array, pcov, confidence_interval, f, fix_vo):
    results = model.params.valuesdict()

    vo = results['vo']
    ko = results['ko']
    kpo = results['kp']

    if kpo == 4.0:
        order = 2
    else:
        order = 3

    if order == 3:
        if fix_vo:
            param_values = [ko,kpo]
            delta_params = [ko*(1e-5),kpo*(1e-5)]     
        else:
            param_values = [vo,ko,kpo]
            delta_params = [vo*(1e-5),ko*(1e-5),kpo*(1e-5)]
    elif order == 2:
        param_values = [vo,ko]
        delta_params = [vo*(1e-5),ko*(1e-5)]
    
    x_m_0s = np.empty_like(x_array)
    f_m_0s = np.empty_like(x_array)
    for i, x in enumerate(x_array):
        x_m_0s[i] = x_array[i]
        f_m_0s[i] = f(results,x)

    diag_delta = np.diag(delta_params)
    dxdbeta = np.empty([len(param_values), len(x_array)])

    for i, value in enumerate(param_values):

        adj_param_values = param_values + diag_delta[i]
                

        if order == 2:
            results['vo'] = adj_param_values[0]
            results['ko'] = adj_param_values[1]    
        elif order == 3 and not fix_vo:
            results['vo'] = adj_param_values[0]
            results['ko'] = adj_param_values[1]
            results['kp'] = adj_param_values[2]
        elif order == 3 and fix_vo:
            results['ko'] = adj_param_values[0]
            results['kp'] = adj_param_values[1]  
            
        for j, x_m_0 in enumerate(x_m_0s):
            dxdbeta[i][j] = (f(results,x_m_0) - f_m_0s[j])/diag_delta[i][i]
    
    variance = np.empty(len(x_array))
    for i, Gprime in enumerate(dxdbeta.T):
        variance[i] = Gprime.T.dot(pcov).dot(Gprime)
        
    critical_value = t.isf(0.5*(confidence_interval + 1.), 3)

    confidence_half_widths = critical_value*np.sqrt(variance)
    #prediction_half_widths = critical_value*np.sqrt(variance + 1e-15 ) #model.noise_variance

    confidence_bound_0 = f_m_0s - confidence_half_widths
    confidence_bound_1 = f_m_0s + confidence_half_widths
    #prediction_bound_0 = f_m_0s - prediction_half_widths
    #prediction_bound_1 = f_m_0s + prediction_half_widths
    
    return np.array([confidence_bound_0, confidence_bound_1])

def fF_plot(pressures,volumes,out_params):    
    p = pressures[:,0]
    sigP = pressures[:,1]
    V = volumes[:,0]
    sigV = volumes[:,1]

    results = out_params.params.valuesdict()

    Vo = results['vo']
    ko = results['ko']
    kpo = results['kp'] 
       
    sigVo = out_params.params['vo'].stderr

    #ignore the divide by zero error if the first piece of data is at 0 GPa
    np.seterr(divide='ignore', invalid='ignore')
    
    #f = (1.0/2.0)*(((V/Vo)**(-2.0/3.0))-1.0)
    f = (((Vo/V)**(2./3.))-1.)/2.
    F = p/(3.*f*(1.+(2.*f))**(5./2.))
    eta = V/Vo
    sigeta = np.abs(eta)*((((sigV/V)**2.0)+((sigVo/Vo)**2))**(1.0/2.0))
    sigprime = ((7.0*(eta**(-2.0/3.0))-5.0)*sigeta)/(2.0*(1.0-(eta**-2.0/3.0))*eta)
    sigF = F*np.sqrt(((sigP/p)**2.0)+(sigprime**2))
    
    line_mod = LinearModel()
    pars = line_mod.guess(f)
    out = line_mod.fit(F, pars, x=f)

    plt.figure(4)
    plt.plot(f, out.best_fit, '-',color='black')
    
    plt.errorbar(f, F, fmt='ko', xerr=0, yerr=sigF,alpha=1.0,capsize = 3.)
    
    plt.xlabel('Eulerian strain $\mathit{f_E}$',fontweight='bold')
    plt.ylabel('Normalized pressure $\mathbf{F_E}$ (GPa)',fontweight='bold')
    plt.tick_params(direction='in',bottom=1,top=1,left=1,right=1)
    plt.title("$\mathit{f_E}$-F",fontweight='bold')

    #plt.savefig('Ff-plot.png',dpi=600,bbox_inches='tight')

    print(out.fit_report())
    
    slope = uct.ufloat(out.params['slope'], out.params['slope'].stderr)
    inter = uct.ufloat(out.params['intercept'], out.params['intercept'].stderr)

    k_p = ((2.0*slope)/(3*inter))+4

    return k_p, f, F, sigF, out.best_fit

def plt_ellipse(cov, pos, nstdl, ax=None):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()
    
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    palet = ['red', 'cyan', 'blue']

    for i in range(1,4):
        nstd = nstdl[i-1]
        hue = palet[i-1]
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, color = hue, alpha=0.8)
        ax.add_artist(ellip)
        ax.tick_params(direction='in',bottom=1,top=1,left=1,right=1)
        
    return ellip

def confidence_ellipses(out_params,sample):

    results = out_params.params.valuesdict()

    vo = results['vo']
    ko = results['ko']
    kpo = results['kp']

    popt = [vo,ko,kpo]

    pcov = out_params.covar
    
    n_params = len(pcov[0])

    fig, ax_array = plt.subplots(n_params-1, n_params-1)
    nstd = [3.0,2.0,1.0]

    err = [np.sqrt(pcov[0,0]),np.sqrt(pcov[1,1]),\
           np.sqrt(pcov[2,2])]

         
    err = np.outer(err,err)
    
    for i in range(n_params-1):
        
        for j in range(i+1, n_params):
            indices = np.array([i, j])
            projected_cov = (pcov)[indices[:, None], indices]
    
            scaled_pos = np.array([popt[i],\
                                   popt[j]]) 
            
            cov = projected_cov
            pos = scaled_pos
                
            ellipse = plt_ellipse(cov,pos,nstd,ax=ax_array[j-1][i])
            maxx = 1.5*2.2*np.sqrt(projected_cov[0][0])
            maxy = 1.5*2.2*np.sqrt(projected_cov[1][1])
            ax_array[j-1][i].set_xlim(scaled_pos[0]-maxx, scaled_pos[0]+maxx)
            ax_array[j-1][i].set_ylim(scaled_pos[1]-maxy, scaled_pos[1]+maxy)
            ax_array[j-1][i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            
            if i == 0 and j == 1:
                ax_array[j-1][i].errorbar(pos[0],pos[1], xerr=np.sqrt(err[0,0]), yerr=np.sqrt(err[1,1]),\
                        linestyle='None', marker='None', label=sample,color='black', capsize = 3.,elinewidth=0.5)
                print('V0 [A^3]: %s +/- %s' % (pos[0],np.sqrt(err[0,0])))
                print('K0 [GPa]: %s +/- %s' % (pos[1],np.sqrt(err[1,1])))
            if i == 0 and j == 2:
                ax_array[j-1][i].errorbar(pos[0],pos[1], xerr=np.sqrt(err[0,0]), yerr=np.sqrt(err[2,2]),\
                        linestyle='None', marker='None', label=sample,color='black', capsize = 3.,elinewidth=0.5)
            if i == 1 and j == 2:
                ax_array[j-1][i].errorbar(pos[0],pos[1], xerr=np.sqrt(err[1,1]), yerr=np.sqrt(err[2,2]),\
                        linestyle='None', marker='None', label=sample,color='black', capsize = 3.,elinewidth=0.5)
                print('K0_prime: %s +/- %s' % (pos[1],np.sqrt(err[2,2])))
            
    red_patch = mpatches.Patch(color='red', alpha=0.8, label='3$\sigma$')
    cyan_patch = mpatches.Patch(color='cyan', alpha=0.8, label='2$\sigma$')
    blue_patch = mpatches.Patch(color='blue', alpha=0.8, label='1$\sigma$')
    us_bm3 = mlines.Line2D([],[],linestyle='none',marker='+', color='black', label='%s' % sample)
    
    ax_array[0][1].set_axis_off()
    legend = ax_array[0][1].legend(handles=[red_patch,cyan_patch,blue_patch,us_bm3], loc = 'center')
    legend.get_frame().set_linewidth(0.0)
    param_names = ['$\mathbf{V_0}$','$\mathbf{K_0}$','$\mathbf{K_0}$\'']
    param_units = ['$\mathbf{\AA^3}$','GPa','']
    if param_names != []:
        for i in range(n_params-1):
            ax_array[n_params-2][i].set_xlabel('{0:s} ({1:s})'.format(param_names[i],param_units[i]),fontweight='bold')
        for j in range(1, n_params):
            if param_units[j] == param_units[2]:
                ax_array[j-1][0].set_ylabel('{0:s}'.format(param_names[j]),fontweight='bold')
            else:
                ax_array[j-1][0].set_ylabel('{0:s} ({1:s})'.format(param_names[j],param_units[j]),fontweight='bold')
            ax_array[j-1][0].yaxis.set_label_coords(-0.25,0.5)
    
    #fig.savefig('CEn-confidence-ellipse.png',dpi=600,bbox_inches='tight')

    return

def fit_volumes(volumes,pressures,fit_options,starting_params):
    results_dict = {}
    
    eos_type = fit_options[0]
    fix_vo = fit_options[1]
    plot_k_conf_band = fit_options[2]
    plot_kp_conf_band = fit_options[3]    
    plot_fit_conf_band = fit_options[4]
    plot_confidence_intervals = fit_options[5]
    interval = fit_options[6]

    if eos_type == 'bm2':
        bm2 = True
    elif eos_type == 'bm3':
        bm2 = False
    elif eos_type == 'vinet':
        bm2 = False
    else:
        print('Please select appropriate eos.')
        
    ci, mini, out_params, PVs = perform_fit(volumes, pressures, starting_params, interval, fix_vo, bm2)
    results_dict['PVs'] = PVs

    if plot_confidence_intervals:
        confidence_intervals(ci,mini,out_params,fix_vo,bm2)
    
    results = out_params.params.valuesdict()
    results_dict['results'] = results

    p_cp_bands = plot_fit(volumes,pressures,out_params,interval,plot_fit_conf_band,bm2,fix_vo,sample)
    results_dict['p_cp_bands'] = p_cp_bands

    k = calc_k(results,PVs[:,1])
    results_dict['k'] = k

    results_dict['interval'] = interval
    k_plot(k,out_params,PVs,interval,plot_k_conf_band,fix_vo)
    
    pcov = out_params.covar
    results_dict['pcov'] = pcov
    
    results_dict['k_cp_bands'] = confidence_prediction_bands(out_params, PVs[:,1], pcov, interval, calc_k, fix_vo)
    
    kp = bm3_calc_kp(results,PVs[:,1])
    results_dict['kp'] = kp
    
    kp_plot(kp,out_params,PVs,interval,plot_kp_conf_band,fix_vo)
    
    kp_cp_bands = confidence_prediction_bands(out_params, PVs[:,1], pcov, interval, bm3_calc_kp, fix_vo)
    results_dict['kp_cp_bands'] = kp_cp_bands    

    kpp = bm3_calc_kpp(out_params)
    results_dict['implied_Kpp'] = kpp

    fF_kp, f, F, sigF, F_fit = fF_plot(pressures,volumes,out_params)
    results_dict['fF_implied_kp'] = fF_kp
    results_dict['f'] = f
    results_dict['F'] = F
    results_dict['sigF'] = sigF
    results_dict['F_fit'] = F_fit
     
    print('\nImplied K\'\' : %0.3f (inverse pressure units)\n' % kpp)
    print('Implied fF K\' : %s\n' % fF_kp)
    
    confidence_ellipses(out_params,sample)
    
    return results_dict

def output_write(vol_results_dict,a_results_dict,b_results_dict,c_results_dict,directory):

    columns = ['P (GPa)', 'V(A\u00b3)', 'P_conf_1', 'P_conf_2',\
               'K (GPa)', 'K_conf_1', 'K_conf_2','K\'', 'K\'_conf_1', 'K\'_conf_2',\
               'a : P (GPa)', 'a : V(A\u00b3)', 'a : P_conf_1', 'a : P_conf_2',\
               'a : K (GPa)', 'a : K_conf_1', 'a : K_conf_2','K\'', 'a : K\'_conf_1', 'a : K\'_conf_2',\
               'b : P (GPa)', 'b : V(A\u00b3)', 'b : P_conf_1', 'b : P_conf_2',\
               'b : K (GPa)', 'b : K_conf_1', 'b : K_conf_2','b : K\'', 'b : K\'_conf_1', 'b : K\'_conf_2',\
               'c : P (GPa)', 'c : V(A\u00b3)', 'c : P_conf_1', 'c : P_conf_2',\
               'c : K (GPa)', 'c : K_conf_1', 'c : K_conf_2','c : K\'', 'c : K\'_conf_1', 'c : K\'_conf_2']
    df = pd.DataFrame(zip(vol_results_dict['PVs'][:,0][::-1], vol_results_dict['PVs'][:,1][::-1], vol_results_dict['p_cp_bands'][1,:][::-1],\
                          vol_results_dict['p_cp_bands'][0,:][::-1], vol_results_dict['k'][::-1], vol_results_dict['k_cp_bands'][1,:][::-1],\
                          vol_results_dict['k_cp_bands'][0,:][::-1], vol_results_dict['kp'][::-1], vol_results_dict['kp_cp_bands'][1,:][::-1],\
                          vol_results_dict['kp_cp_bands'][0,:][::-1],\
                          a_results_dict['PVs'][:,0][::-1], a_results_dict['PVs'][:,1][::-1], a_results_dict['p_cp_bands'][1,:][::-1],\
                          a_results_dict['p_cp_bands'][0,:][::-1], a_results_dict['k'][::-1], a_results_dict['k_cp_bands'][1,:][::-1],\
                          a_results_dict['k_cp_bands'][0,:][::-1], a_results_dict['kp'][::-1], a_results_dict['kp_cp_bands'][1,:][::-1],\
                          a_results_dict['kp_cp_bands'][0,:][::-1],\
                          b_results_dict['PVs'][:,0][::-1], b_results_dict['PVs'][:,1][::-1], b_results_dict['p_cp_bands'][1,:][::-1],\
                          b_results_dict['p_cp_bands'][0,:][::-1], b_results_dict['k'][::-1], b_results_dict['k_cp_bands'][1,:][::-1],\
                          b_results_dict['k_cp_bands'][0,:][::-1], b_results_dict['kp'][::-1], b_results_dict['kp_cp_bands'][1,:][::-1],\
                          b_results_dict['kp_cp_bands'][0,:][::-1],\
                          c_results_dict['PVs'][:,0][::-1], c_results_dict['PVs'][:,1][::-1], c_results_dict['p_cp_bands'][1,:][::-1],\
                          c_results_dict['p_cp_bands'][0,:][::-1], c_results_dict['k'][::-1], c_results_dict['k_cp_bands'][1,:][::-1],\
                          c_results_dict['k_cp_bands'][0,:][::-1], c_results_dict['kp'][::-1], c_results_dict['kp_cp_bands'][1,:][::-1],\
                          c_results_dict['kp_cp_bands'][0,:][::-1]), columns = columns)

    df.to_csv(os.path.join(directory,'eos_output.csv'), sep=',')

    return
    
def main(directory,sample,fit_options):

    fit_lattice_params = fit_options[7]

    parameters_dict = retrieve_data(directory)
    
    # get volumes and lattice parameters
    volumes = []
    a = []
    b = []
    c = []
    
    # create lists of volumes and lattice parameters
    for i in parameters_dict.keys():
        volumes.append(parameters_dict[i]['1VOL'])
        
        a.append(parameters_dict[i].get('1A'))
        b.append(parameters_dict[i].get('1B'))
        c.append(parameters_dict[i].get('1C'))

    #Volumes in A^3
    volumes = np.asarray(volumes)
    if None not in a:
        a = np.asarray(a)       
        a[:,1] = 3.0*(a[:,0]**2)*a[:,1]
        a[:,0] = a[:,0]**3
    if None not in b:
        b = np.asarray(b)
        b[:,1] = 3.0*(b[:,0]**2)*b[:,1]
        b[:,0] = b[:,0]**3
    if None not in c:
        c = np.asarray(c)
        c[:,1] = 3.0*(c[:,0]**2)*c[:,1]
        c[:,0] = c[:,0]**3
    print(np.max(c[:,0]))

    #Pressures in GPa
    pressures = np.array([[5.34,0.0534],\
                         [5.78,0.0578],\
                         [6.43,0.0643],\
                         [6.93,0.0693],\
                         [7.3,0.073],\
                         [7.93,0.0793],\
                         [9.5,0.095],\
                         [10.437,0.10437],\
                         [17.0,0.17],\
                         [24.35,0.2435],\
                         [30.091,0.30091],\
                         [35.493,0.35493]])
    
    labels = ['Pressures','sigP','Volumes','sigV']
    data = pd.DataFrame(np.column_stack((pressures,volumes)),columns=labels)

    # print input data
    print('\n%s\n' % data)

    #initial starting parameter guesses
    starting_params = {'vo':400.0, \
                       'ko':134.0, \
                       'kp':4.0}

    #starting_params = {'vo':np.max(c), \
    #                    'ko':150.0, \
    #                    'kp':4.0}

    # perform curve fitting to volumes
    vol_results_dict = fit_volumes(volumes,pressures,fit_options,starting_params)

    # creates dummy variable incase lattice parameters aren't fit
    a_results_dict = vol_results_dict
    b_results_dict = vol_results_dict
    c_results_dict = vol_results_dict

    # fit lattice parameters
    if fit_lattice_params:
        fit_options[0] = 'bm2'
        if None not in a:
            starting_params = {'vo':np.max(a), \
                        'ko':150.0, \
                        'kp':4.0}
            a_results_dict = fit_volumes(a,pressures,fit_options,starting_params)
        if None not in b:
            starting_params = {'vo':np.max(b), \
                        'ko':150.0, \
                        'kp':4.0}
            b_results_dict = fit_volumes(b,pressures,fit_options,starting_params)
        if None not in c:
            starting_params = {'vo':np.max(c), \
                        'ko':100.0, \
                        'kp':4.0}
            c_results_dict = fit_volumes(c,pressures,fit_options,starting_params)
    
    output_write(vol_results_dict,a_results_dict,b_results_dict,c_results_dict,directory)
    
    return

if __name__ == "__main__":
    '''
    Retrieve volume from GSAS .PVE files and fit to equation of state.

    -Equations of state: Birch-Murnaghan 2nd and 3rd order, Vinet
    -Fit lattice parameter "volumes" to get axial compressibilities.
    -Plot equation of state
    -Plot confidence intervals
    -Plot confidence ellipses
    -Plot confidence bands
    -Plot bulk modulus (K)
    -Plot first derivitive of the bulk modulus (K')
    -Plot F-f figure (F - Normalized Pressure, f - Eulerian Strain)
    
    John Lazarz
    191217
    '''
    
    # replace directory string with appropriate path to .PVE files
    directory = '.' #/Users/johnlazarz/Desktop/mesoscale_eos'
    
    #assign a sample name
    sample = '$Mg_2Si_2O_6\ -\ C2/c$'

    #select the type of equation of state
    #options: bm2,bm3,vinet
    eos_type = 'bm3'
    fix_vo = False

    #set confidece interval to be calculated (not heat map)
    interval = 0.95
    
    #set which confidence bands should be calculated and plotted
    plot_k_conf_band = True
    plot_kp_conf_band = True    
    plot_fit_conf_band = True
    
    #choose to calculate and plot confidence intervals
    plot_confidence_intervals = False

    #choose to fit lattice parameters
    fit_lattice_params = False

    fit_options = [eos_type,\
                   fix_vo,\
                   plot_k_conf_band,\
                   plot_kp_conf_band,\
                   plot_fit_conf_band,\
                   plot_confidence_intervals,\
                   interval,\
                   fit_lattice_params]
    
    if eos_type == 'bm2' and fix_vo:
        print('Cannot fix both K\' and Vo.')
    else:
        main(directory,sample,fit_options)

    plt.show()

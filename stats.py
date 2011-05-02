import numpy as np
from itertools import groupby

import matplotlib
matplotlib.use('Agg')
import pylab as pl
import tables

def load_scalars(fname):
    dt = np.dtype({'names':('index', 'wall_time', 'basic_steps', 'sim_time', 'be','ve', 'ne', 'msf'),
                   'formats':('I', 'f', 'I', 'f', 'f', 'f', 'f', 'f')})
    scalar_arr = np.loadtxt(fname, dt)
    return scalar_arr

def load_scalars_h5(fname):
    dta = tables.openFile(fname)
    scals = dta.getNode('/scalars')
    sdta = scals.scalar_data.read()
    return sdta

def plot_energy(fname, title=None, loader=load_scalars_h5, plotter=pl.semilogy):
    scalar_arr = loader(fname)
    pl.figure(figsize=(11,8))
    colors = 'bgkw'
    symbols = 'ov*^'
    etot = scalar_arr['be'] + scalar_arr['ne'] + scalar_arr['ve']
    arr_names = ['be', 'ne', 've']
    arrs = [scalar_arr[nm] for nm in arr_names]
    arr_names += ['total']
    arrs += [etot]
    for idx, (nm, field) in enumerate(zip(arr_names, arrs)):
        plotter(scalar_arr['sim_time'], field, colors[idx]+symbols[idx]+'-', label=nm)
    pl.legend()
    pl.title(title)
    pl.xlabel('Time (arb. units)')
    pl.ylabel('Energy (arb. units)')

def load_stats_h5(fname):
    dta = tables.openFile(fname)
    stats = dta.getNode('/stats')
    all_stats = {}
    for tb in stats:
        all_stats[tb.name] = tb.read()
    dta.close()
    return all_stats

def load_stats(fname):
    dt = np.dtype({'names':('index', 'name', 'ave', 'std_dev', 'skew', 'kurt'),
                   'formats':('I', 'S3', 'f', 'f', 'f', 'f')})
    st_arr = np.loadtxt(fname, dt)
    st_arr.sort(order=['name', 'index'])
    all_stats = {}
    for k,g in groupby(st_arr, lambda x: x['name']):
        all_stats[k] = np.array(list(g), dtype=dt)
    return all_stats

def plot_stats(fname, loader):
    pl.figure(figsize=(11,8))
    stat_arrs = loader(fname)
    nfields = len(stat_arrs)
    colors = 'bgrcmykw'
    symbols = 'ov^<>sp*h+xD'
    for idx, field in enumerate(stat_arrs):
        pl.plot(stat_arrs[field]['kurt'], colors[idx]+symbols[idx]+'-', label=field+' kurt')
    pl.legend()

def plot_many_stats(fnames, loader=load_stats_h5, figsize=(11,8), ext="eps"):
    narrs = len(fnames)
    stat_arrs_list = [loader(fname) for fname in fnames]
    keys = stat_arrs_list[0].keys()
    moments = 'ave', 'std_dev', 'kurt', 'skew'
    colors = 'bgrcmykw'
    symbols = 'ov^<>sp*h+xD'
    ncos = len(colors)
    for key in keys:
        arrs = [stat_arr[key] for stat_arr in stat_arrs_list]
        for moment in moments:
            name = "%s %s" % (key, moment)
            pl.figure(figsize=figsize)
            for idx, arr in enumerate(arrs):
                pl.plot(arr[moment], colors[idx%ncos]+symbols[idx%ncos]+'-')
            pl.xlabel('time (simulation units)')
            pl.ylabel(moment)
            pl.title('%s %s' % (key, moment))
            pl.savefig('%s_%s.%s' % (key, moment, ext))

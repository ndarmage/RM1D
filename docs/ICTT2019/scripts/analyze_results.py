#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
Analyze the results of the 1g homogeneous test cases by Sood et al. [Sood2003]_.

.. [Sood2003] Sood, A., Forster, R. A., & Parsons, D. K. (2003). Analytical
              benchmark test set for criticality code verification. Progress
              in Nuclear Energy, 42(1), 55-106.
"""
import sys, os
import itertools
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt

rootdir = os.path.join(os.getcwd(), "..", "..", "..")
cpmdir = os.path.join(rootdir, "CPM1D", "tests", "output")
rmtdir = os.path.join(rootdir, "FD1dMGdiff", "tests", "output")
figdir = os.path.join("..", "figures")

sys.path.insert(0, os.path.join("..", "..", "..", "FD1dMGdiff"))
from data.SoodPNE2003 import *
from GeoMatTools import *

# plt.rcParams.update({'font.size': 14})
# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]



def get_files(dir):
    return [f for f in os.listdir(dir) if f.endswith('npy')]


# get the list of cases in both dirs
cpm_files = get_files(cpmdir)
rmt_files = get_files(rmtdir)


refflx = {
    'PUb-1-0-SL': np.array([0.9701734, 0.8810540, 0.7318131, 0.4902592]),
    'PUb-1-0-CY': np.array([np.nan, 0.8093, np.nan, 0.2926]),
    'PUb-1-0-SP': np.array([0.93538006, 0.75575352, 0.49884364, 0.19222603])
}


def get_allIs(list_of_files, case, RM=False):
    case_files = [f for f in list_of_files if get_case(f, RM) == case]
    return sorted(list(set([get_I(c) for c in case_files])))


def get_I(fn):
    return int(fn.split('_I')[1].replace('.npy', ''))


def get_case(fn, RM=False):
    return fn.split('_LBC' if RM else '_ref')[0]


def get_filename(case, I):
    return case + '_ref_I%d.npy' % I


def get_filelist_per_case(files, case, RM=False):
    return [f for f in files if get_case(f, RM) == case]


def plot_cpm1d_hslab_error(L, I_hslab, figname):
    ks = np.zeros(len(I_hslab))
    hs = np.zeros_like(ks)
    ws = np.zeros_like(ks)
    Ds = np.zeros_like(ks)
    shw = np.zeros(len(I_hslab), dtype=np.bool)
    
    fig, (a1, a2) = plt.subplots(1, 2)
    # ms = itertools.cycle(('.o>^+*8spx|_'))
    # ms = itertools.cycle(('.^+*'))
    ls = itertools.cycle([l for i,l in linestyle_tuple])
    # from cycler import cycler
    # a1.set_prop_cycle(cycler('color', list('rbgykcm')) * 
                   # # cycler('marker', ['.', '^', '+', '*']) *
                   # cycler('ls', ['-', ':', '-.', '--']))
    for i, I in enumerate(I_hslab):
        if not 2*I in I_slab: continue
        f_RM = os.path.join(rmtdir, case + "_LBC2RBC0_I%d.npy" % I)
        if not os.path.isfile(f_RM):
            raise RuntimeError("Missing file " + f_RM)
        shw[i] = 1
        x = equivolume_mesh(I, 0, L, 'slab') / L
        V = compute_cell_volumes(x, 'slab', per_unit_angle=False)
        Ds[i] = V[0]
        k, flx, _ = np.load(os.path.join(cpmdir, get_filename(case, 2*I)),
                            allow_pickle=True)  # full slab
        h, flh, _ = np.load(os.path.join(cpmdir, get_filename(case + 'h', I)),
                            allow_pickle=True)  # half slab
        w_save, flw_save, xi, xst, dD = np.load(os.path.join(rmtdir, f_RM),
                            allow_pickle=True)  # half slab
        if not np.allclose(np.diff(x), V):
            raise RuntimeError('spatial mesh mismatch')
        
        idx = np.argwhere(w_save != -1)[-1]
        w, flw = w_save[idx], flw_save[0, :, idx]
        
        ks[i], hs[i], ws[i] = (1 - k)*1e5, (1 - h)*1e5, (1 - w)*1e5
        flx /= np.sum(flx[0, :I] * V)
        flh /= np.sum(flh * V)
        flw /= np.sum(flw) * V
        if i%3 == 0:
            rerr = (1 - flh[0,:] / flx[0,:I])*100
            a1.plot(xim(x), rerr, label='$\Delta_{CP} = %.3e$' % V[0]
                    #, marker=next(ms)
                    , linestyle = next(ls), color='C1'
                    )
            rerr = (1 - flw[0,:][::-1] / flx[0,:I])*100
            a1.plot(xim(x), rerr, label='$\Delta_{RM} = %.3e$' % V[0]
                    #, marker=next(ms)
                    , linestyle = next(ls), color='C2'
                    )
        
    a1.legend()
    a1.set_xlabel('$x/L_c$')
    a1.set_ylabel('$1 - \phi_{HS}/\phi_{FS} \: (\%)$')
    a2.plot(Ds[shw], ks[shw], label='full-slab (FS)')
    a2.plot(Ds[shw], hs[shw], label='half-slab (HS-CP)', ls='--')
    # a2.plot(Ds[shw], ws[shw], label='half-slab (HS-RM)', ls='-.')
    a2.set_ylabel('$1-k$ (pcm)')
    a2.set_xlabel('$\Delta$')
    a2.set_xscale('log')
    a2.set_yscale('log')
    a2.legend()
    plt.tight_layout()
    plt.savefig(figname)
    plt.close(fig)


def plot_cpm1d_PUb_flx(figname):
    fs_sl = get_filelist_per_case(cpm_files, 'PUb-1-0-SL')
    fs_cy = get_filelist_per_case(cpm_files, 'PUb-1-0-CY')
    fs_sp = get_filelist_per_case(cpm_files, 'PUb-1-0-SP')
    I_sl = get_allIs(cpm_files, 'PUb-1-0-SL')
    I_cy = get_allIs(cpm_files, 'PUb-1-0-CY')
    I_sp = get_allIs(cpm_files, 'PUb-1-0-SP')
    rf = np.linspace(0, 1, 5)
    
    if I_cy != [i/2 for i in I_sl]:
        print(I_cy)
        print(I_sl)
        print('mismatching nb. of I''s in cy and sl')
    if I_cy != I_sp:
        print('different nb. of I''s in cy and sp')
    
    def fetch_RM_results(f_RM):
        k, flx, rel_err_pc = None, None, None
        if os.path.isfile(f_RM):
            w_save, flw_save, xi, xst, dD = np.load(
                f_RM, allow_pickle=True)  # half slab
            idx = np.argwhere(w_save != -1)[-1]
            k, flx = w_save[idx][0], flw_save[0, :, idx]
            rel_err_pc = np.load(f_RM.replace('.npy', '_RMre.npy'),
                allow_pickle=True)
        else:
            # raise RuntimeError("Missing file " + f_RM)
            print("Missing file " + f_RM)
        return k, flx, rel_err_pc
    
    c, d = dict(), dict()
    d['SL'], d['CY'], d['SP'] = dict(), dict(), dict()
    c['SL'], c['CY'], c['SP'] = dict(), dict(), dict()
    shw = np.zeros(len(I_cy), dtype=np.bool)
    for i, I in enumerate(I_cy):
        if I > 125: break
        r = equivolume_mesh(I*2, 0, Lc['PUb-1-0-SL']*2, 'slab') \
            / (Lc['PUb-1-0-SL']*2)
        d['SL']['V'] = compute_cell_volumes(r, 'slab', False)
        d['SL'][I] = np.load(
            os.path.join(cpmdir, get_filename('PUb-1-0-SL', I*2)),
                       allow_pickle=True) # = k, flx, e
        r = equivolume_mesh(I, 0, Lc['PUb-1-0-CY'], 'cylinder') \
            / Lc['PUb-1-0-CY']
        d['CY']['V'] = compute_cell_volumes(r, 'cylinder', False)
        d['CY'][I] = np.load(
            os.path.join(cpmdir, get_filename('PUb-1-0-CY', I)),
                       allow_pickle=True) # = k, flx, e
        r = equivolume_mesh(I, 0, Lc['PUb-1-0-SP'], 'sphere') \
            / Lc['PUb-1-0-SP']
        d['SP']['V'] = compute_cell_volumes(r, 'sphere', False)
        d['SP'][I] = np.load(
            os.path.join(cpmdir, get_filename('PUb-1-0-SP', I)),
                       allow_pickle=True) # = k, flx, e
        
        case = 'PUb-1-0-%s' % 'SL'
        f_RM = os.path.join(rmtdir, case + "_LBC0RBC0_I%d.npy" % (2*I))
        c['SL'][I] = fetch_RM_results(f_RM)
        
        case = 'PUb-1-0-%s' % 'CY'
        f_RM = os.path.join(rmtdir, case + "_LBC2RBC0_I%d.npy" % I)
        c['CY'][I] = fetch_RM_results(f_RM)
        
        case = 'PUb-1-0-%s' % 'SP'
        f_RM = os.path.join(rmtdir, case + "_LBC2RBC0_I%d.npy" % I)
        c['SP'][I] = fetch_RM_results(f_RM)
        if c['CY'][I][0] is not None:
            shw[i] = 1
            # print(c['SL'][I])
            # print(I, 1-d['SL'][I][0], 1-d['CY'][I][0], 1-d['SP'][I][0])
            # print(I, 1-c['SL'][I][0], 1-c['CY'][I][0], 1-c['SP'][I][0])
    
    geos = ['SL', 'CY', 'SP']
    I_cy = np.array(I_cy)
    
    y = lambda geo, pos: [d[geo][i][2][pos] for i in I_cy[shw]]
    z = lambda geo, pos: [c[geo][i][2][pos] for i in I_cy[shw]]
    fig, axes = plt.subplots(2,2, sharex=True)
    ls = ['-', ':', '--']
    mr = ['*', '.', '+']
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for g, geo in enumerate(geos):
        x = 2 * I_cy[shw] if 'SL' in geo else I_cy[shw]
        axes[0,0].semilogx(x, y(geo, 0), label=geo+'-CP', linestyle=ls[g], color='C0')
        axes[0,0].semilogx(x, z(geo, 0), label=geo+'-RM', linestyle=ls[g], color='C1', marker=mr[g])
        axes[0,1].semilogx(x, y(geo, 1), label=geo+'-CP', linestyle=ls[g], color='C0')
        axes[0,1].semilogx(x, z(geo, 1), label=geo+'-RM', linestyle=ls[g], color='C1', marker=mr[g])
        axes[1,0].semilogx(x, y(geo, 2), label=geo+'-CP', linestyle=ls[g], color='C0')
        axes[1,0].semilogx(x, z(geo, 2), label=geo+'-RM', linestyle=ls[g], color='C1', marker=mr[g])
        axes[1,1].semilogx(x, y(geo, 3), label=geo+'-CP', linestyle=ls[g], color='C0')
        axes[1,1].semilogx(x, z(geo, 3), label=geo+'-RM', linestyle=ls[g], color='C1', marker=mr[g])
    axes[1,0].set_xlabel('$I$')
    axes[1,1].set_xlabel('$I$')
    axes[0,0].set_ylabel('Rel. Err. (%)')
    axes[1,0].set_ylabel('Rel. Err. (%)')
    axes[0,0].set_yscale('symlog')
    axes[0,1].set_yscale('symlog')
    axes[1,0].set_yscale('symlog')
    axes[1,1].set_yscale('symlog')
    axes[0,0].text(0.9, 0.1,'$r_1$', horizontalalignment='center',
        verticalalignment='center', transform = axes[0,0].transAxes, bbox=props)
    axes[0,1].text(0.9, 0.1,'$r_2$', horizontalalignment='center',
        verticalalignment='center', transform = axes[0,1].transAxes, bbox=props)
    axes[1,0].text(0.9, 0.1,'$r_3$', horizontalalignment='center',
        verticalalignment='center', transform = axes[1,0].transAxes, bbox=props)
    axes[1,1].text(0.9, 0.1,'$r_4$', horizontalalignment='center',
        verticalalignment='center', transform = axes[1,1].transAxes, bbox=props)
    handles, labels = axes[0,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, handletextpad=0.25,
        bbox_to_anchor=(0.5, 0), bbox_transform=plt.gcf().transFigure)
    plt.tight_layout(rect=(0,0.05,1,1))
    plt.savefig(figname % 'flxerr')
    plt.close(fig)
    
    fig = plt.figure()
    print(I_cy[shw])
    y = lambda geo: [(1-d[geo][i][0])*1e5 for i in I_cy[shw]]
    z = lambda geo: [(1-c[geo][i][0])*1e5 for i in I_cy[shw]]
    for g, geo in enumerate(geos):
        x = 2 * I_cy[shw] if 'SL' in geo else I_cy[shw]
        plt.loglog(x, y(geo), label=geo+'-CP', linestyle=ls[g], color='C0')
        plt.loglog(x, z(geo), label=geo+'-RM', linestyle=ls[g], color='C1',
                   marker=mr[g])
    plt.legend()
    plt.xlabel('$I$')
    plt.ylabel('$1-k$ (pcm)')
    plt.savefig(figname % 'kerr')
    plt.close(fig)


if __name__ == "__main__":
    
    I_slab = get_allIs(cpm_files, 'PUa-1-0-SL')
    I_hslab = get_allIs(cpm_files, 'PUa-1-0-SLh')
    # if I_slab != [i*2 for i in I_hslab]:
        # print(I_slab)
        # print(I_hslab)
        # raise RuntimeError('lists differ')

    if False:
        case = 'PUa-1-0-SL'
        plot_cpm1d_hslab_error(Lc[case], I_hslab,
            os.path.join(figdir, "cpm_hslab_err.pdf"))
    
    if True:
        plot_cpm1d_PUb_flx(os.path.join(figdir, "cpm_PUb_%s.pdf"))

    sys.exit()
    fs_sl = get_filelist_per_case(rmt_files, 'PUb-1-0-SL', RM=True)
    fs_cy = get_filelist_per_case(rmt_files, 'PUb-1-0-CY', RM=True)
    fs_sp = get_filelist_per_case(rmt_files, 'PUb-1-0-SP', RM=True)
    I_sl = get_allIs(rmt_files, 'PUb-1-0-SL', RM=True)
    I_cy = get_allIs(rmt_files, 'PUb-1-0-CY', RM=True)
    I_sp = get_allIs(rmt_files, 'PUb-1-0-SP', RM=True)

    print(I_sl)
    print(I_cy)
    print(I_sp)
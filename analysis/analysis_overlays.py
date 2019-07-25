#################### imports ####################
# standard
import sys
import os
import numpy as np
import scipy.sparse as ssp
import yaml
import argparse
import shutil
#from PIL import Image
import tifffile as ti
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.ticker
from matplotlib.colors import cnames as colornames

# custom
origin=os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.join(origin,'..','image_processing'))
from utils import *

#################### global params ####################
# yaml formats
npfloat_representer = lambda dumper,value: dumper.represent_float(float(value))
nparray_representer = lambda dumper,value: dumper.represent_list(value.tolist())
float_representer = lambda dumper,value: dumper.represent_scalar(u'tag:yaml.org,2002:float', "{:<.8e}".format(value))
unicode_representer = lambda dumper,value: dumper.represent_unicode(value.encode('utf-8'))
yaml.add_representer(float,float_representer)
yaml.add_representer(np.float_,npfloat_representer)
yaml.add_representer(np.ndarray,nparray_representer)
yaml.add_representer(unicode,unicode_representer)

# matplotlib controls
plt.rcParams['svg.fonttype'] = 'none'  # to embed fonts in output ('path' is to convert as text as paths)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['axes.linewidth']=0.5

#################### function ####################
def default_parameters():
    """Generate a default parameter dictionary."""

    print "Loading default parameters"
    params={}
    # filtering - select only a subset
    params['analysis_overlays']={}
    # queen signal
    params['analysis_overlays']['hist_queen']={}
    mydict = params['analysis_overlays']['hist_queen']
    mydict['channels'] = [0,1]
    mydict['bins'] = [16, 16, 16]
    mydict['colors']=[None, None, None]
    mydict['units_dx']=None

    # fluorescence
    params['analysis_overlays']['hist_channel']={}
    mydict = params['analysis_overlays']['hist_channel']
    mydict['channel'] = 0
    mydict['bins'] = [16,16,16]
    mydict['colors']=[None,None,None]
    mydict['units_dx']=None
    mydict['mode']='total_fl'  # alternative value is \'concentration\'
    return params

def hist_dimensions(celldicts, labels, outputdir='.', bins=None, colors=None, units_dx=None, title=None, mode='um',qcut=0):
    """
    Make an histogram of the signal obtained per cell.
    """

    # initialization
    if mode == 'um':
        titles = [u'length (\u03BCm)', u'width (\u03BCm)', u'area (\u03BCm\u00B2)', u'volume (\u03BCm\u00B3)']
        attrs = ['height_um','width_um','area_um2', 'volume_um3']
    else:
        titles = [u'length (px)', u'width (px)', u'area (px\u00B2)', u'volume (px\u00B3)']
        attrs = ['height','width','area', 'volume']

    nattrs = len(attrs)
    if len(units_dx) != nattrs:
        raise ValueError("units_dx has the wrong dimensions!")

    ndata = len(celldicts)
    if ndata == 0:
        raise ValueError("Empty input data")
    print "ndata = {:d}".format(ndata)

    # colors
    if colors is None:
        colors = colornames.keys()[:ndata]

    # bins
    if bins is None:
        bins={j: None for j in range(nattrs)}
    for j in range(nattrs):
        if bins[j] is None:
            bins[j] = ['auto']*ndata

    # units_dx
    if units_dx is None:
        units_dx={i:None for i in range(nattrs)}

    # titles
    if titles is None:
        titles=[None for i in range(nattrs)]

    # filling up the data
    data=[]
    for i in range(ndata):
        cells = celldicts[i]
        ncells = len(cells)
        if ncells == 0:
            raise ValueError("Empty cell dictionary!")

        # make lists
        dimensions = [ [] for i in range(nattrs)]
        keys = cells.keys()
        for n in range(ncells):
            key = keys[n]
            cell = cells[key]
            for i in range(nattrs):
                attr = attrs[i]
                dimensions[i].append(cell[attr])

        dimensions = np.array(dimensions)

        data.append(dimensions)
    # end loop on data sets

    Ns = [len(d) for d in data]
    mus = [np.mean(d, axis=1).astype(d.dtype) for d in data]
    meds = [np.median(d, axis=1).astype(d.dtype) for d in data]
    sigs = [np.std(d, axis=1).astype(d.dtype) for d in data]
    errs = [s/np.sqrt(N) for s,N in zip(sigs,Ns)]

    # make figure
    fig = plt.figure(num=None, facecolor='w', figsize=(4*nattrs,3))
    gs = mgs.GridSpec(1,nattrs)

    ax0 = fig.add_subplot(gs[0,0])
    axes = [ax0]
    for j in range(1,nattrs):
        #ax = fig.add_subplot(gs[0,j],sharey=ax0)
        ax = fig.add_subplot(gs[0,j])
        axes.append(ax)

    for j in range(nattrs):
        attr = attrs[j]
        print "attr {:d} / {:d}".format(j,nattrs-1)
        ax = axes[j]

        for i in range(ndata):
            print "{:<2s}data {:d} / {:d}".format("", i, ndata-1)
            # compute histogram
            d = data[i][j]
            N = len(d)
            d = np.sort(d)
            n0 = int(qcut*float(N))
            n1 = min(int((1.-qcut)*float(N)),N-1)
            d = d[n0:n1+1]
            print "{:<4s}qcut = {:.1f} %".format("",qcut*100)
            hist,edges = np.histogram(d, bins=bins[j][i], density=True)
            print "{:<4s}nbins = {:,d}".format("",len(edges)-1)

            # plot histogram
            color = colors[i]
            #ax.bar(edges[:-1], hist, np.diff(edges), color='none', edgecolor=color, lw=0.5, label=labels[i])
            if j == 0:
                label = labels[i]
            else:
                label = None
            ax.plot(0.5*(edges[:-1]+edges[1:]), hist, '-', color=color, lw=0.5, label=label)
        # end loop
    # end loop


        # add legends
        if not (titles[j] is None):
            ax.set_title(titles[j], fontsize='large')
        #ax.annotate(fmts[i].format(mu=mus[i],sig=sigs[i], N=len(data[i]), med=meds[i]), xy=(0.70,0.98), xycoords='axes fraction', ha='left', va='top')

        # adjust the axis
        if (j == 0):
            ax.legend(loc='best',fontsize="medium",frameon=False)
        #ax.set_ylabel("pdf",fontsize="medium",labelpad=10)
        ax.tick_params(length=4)
        ax.tick_params(axis='both', labelsize='medium', labelleft=False, left=False)

        if not (units_dx[j] is None):
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dx[j]))
#        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))
#        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
#        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))

        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    gs.tight_layout(fig)
    if mode == 'um':
        filename = 'analysis_overlay_dimensions_um'
    else:
        filename = 'analysis_overlay_dimensions_px'

    exts=['.pdf', '.svg', '.png']
    for ext in exts:
        fileout = os.path.join(outputdir,filename+ext)
        fig.savefig(fileout, bbox_inches='tight', pad_inches=0)
        print "Fileout: {:<s}".format(fileout)
    return

def hist_channel(celldicts, labels, outputdir='.', bins=None, colors=None, units_dx=None, titles=None, mode='total_fl',qcut=0, xminxmax=None, backgrounds=None):
    """
    Make an histogram of the signal obtained per cell.
    """

    # initialization
    if mode == 'concentration_fl':
        print "concentration_fl mode"
        fl_dtype = np.float_
        fmt = "$\\mu = {mu:,.2f}$\n$\\sigma = {sig:,.2f}$\n$N = {N:,d}$\n$\\mathrm{{med}} = {med:,.2f}$"
    elif mode == 'total_fl':
        print "total_fl mode"
        fl_dtype = np.uint16
        fmt = "$\\mu = {mu:,d}$\n$\\sigma = {sig:,d}$\n$N = {N:,d}$\n$\\mathrm{{med}} = {med:,d}$"
    else:
        raise ValueError('Wrong mode selection: \'total_fl\' or \'concentration_fl\'')

    # input data
    ndata = len(celldicts)
    if ndata == 0:
        raise ValueError("Empty input data")
    if ndata != len(labels):
        raise ValueError("Labels should have the same dimension as cell dicts")
    print "ndata = {:d}".format(ndata)

    # channels
    nchannel = len(celldicts[0].values()[0]['fluorescence']['total'])
    print "nchannel = {:d}".format(nchannel)

    # colors
    if colors is None:
        colors = colornames.keys()[:ndata]

    # bins
    if bins is None:
        bins={i: None for i in range(nchannel)}
    for c in range(nchannel):
        if bins[c] is None:
            bins[c] = ['auto']*ndata

    # units_dx
    if units_dx is None:
        units_dx={i:None for i in range(nchannel)}

    # titles
    if titles is None:
        titles=[None for i in range(nchannel)]

    # xminxmax
    if xminxmax is None:
        xminxmax=[[None, None] for i in range(nchannel)]

    # filling up the data
    data = [[] for c in range(nchannel)]
    data_bg = [[] for c in range(nchannel)]
    for c in range(nchannel):
        for i in range(ndata):
            cells = celldicts[i]
            ncells = len(cells)
            if ncells == 0:
                raise ValueError("Empty cell dictionary!")
            FL = []
            BG = []

            # make lists
            keys = cells.keys()
            for n in range(ncells):
                key = keys[n]
                cell = cells[key]
                npx = cell['area']
                fl = cell['fluorescence']['total']
                bg_px = cell['fluorescence']['background_px']
                x = fl[c]
                bg = bg_px[c]*npx
                if mode == 'concentration_fl':
                    try:
                        volume = cell['volume']
                    except KeyError:
                        raise ValueError('Missing volume attribute in cell!')
                    x = float(x) / volume
                    bg = float(bg) / float(volume)
                elif mode == 'total_fl':
                    pass
                FL.append(x)
                BG.append(bg)
            # end loop on cells
            FL = np.array(FL,dtype=fl_dtype)
            BG = np.array(BG,dtype=fl_dtype)
            data[c].append(FL)
            data_bg[c].append(BG)
        # end loop on data sets
    # end loop on channels

#    Ns = [len(d) for d in data]
#    mus = [np.mean(d).astype(d.dtype) for d in data]
#    meds = [np.median(d).astype(d.dtype) for d in data]
#    sigs = [np.std(d).astype(d.dtype) for d in data]
#    errs = [s/np.sqrt(N) for s,N in zip(sigs,Ns)]
    if backgrounds is None:
        bgcolor='r'
        bgs = [np.nanmedian(np.concatenate(data_bg[c])).astype(data_bg[c][0].dtype) for c in range(nchannel)]
    else:
        bgcolor='g'
        bgs = [np.float_(backgrounds[c]) for c in range(nchannel)]

    # make figure
    fig = plt.figure(num=None, facecolor='w', figsize=(4*nchannel,3))
    gs = mgs.GridSpec(1,nchannel)

    ax0 = fig.add_subplot(gs[0,0])
    axes = [ax0]
    for c in range(1,nchannel):
        ax = fig.add_subplot(gs[0,c])
        axes.append(ax)

    for c in range(nchannel):
        print "channel = {:d} / {:d}".format(c,nchannel-1)
        ax=axes[c]
        for i in range(ndata):
            print "{:<2s}data {:d} / {:d}".format("", i, ndata-1)
            # compute histogram
            d = data[c][i]
            N = len(d)
            d = np.sort(d)
            n0 = int(0.5*qcut*float(N))
            print "{:<4s}qcut = {:.1f} %".format("",qcut*100)
            n1 = N - n0
#            print n0, n1
            d = d[n0:n1]
            hist,edges = np.histogram(d, bins=bins[c][i], density=True)
            print "{:<4s}nbins = {:,d}".format("",len(edges)-1)

            # plot histogram
            color = colors[i]
            #ax.bar(edges[:-1], hist, np.diff(edges), color='none', edgecolor=color, lw=0.5, label=labels[i])
            if (c ==0):
                label=labels[i]
            else:
                label=None

            ax.plot(0.5*(edges[:-1]+edges[1:]), hist, '-', color=color, lw=0.5, label=label)
            # end loop on data

        # plot background
        ax.axvline(x=bgs[c], color=bgcolor, lw=0.5, ls='--')

        # add legends
        if not (titles[c] is None):
            ax.set_title(titles[c], fontsize='large')

        # adjust the axis
        if (c == 0):
            ax.legend(loc='best',fontsize="medium",frameon=False)
        #ax.set_ylabel("pdf",fontsize="medium",labelpad=10)
        ax.tick_params(length=4)
        ax.tick_params(axis='both', labelsize='medium', labelleft=False, left=False)

        if not (units_dx[c] is None):
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dx[c]))
#        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))
#        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
#        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))

        ax.set_xlim(xminxmax[c])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    # end loop on channels

    gs.tight_layout(fig)
    if mode == 'concentration_fl':
        filename = 'analysis_overlay_concentration_fl'
    elif mode == 'total_fl':
        filename = 'analysis_overlay_total_fl'

    exts=['.pdf', '.svg', '.png']
    for ext in exts:
        fileout = os.path.join(outputdir,filename+ext)
        fig.savefig(fileout, bbox_inches='tight', pad_inches=0)
        print "Fileout: {:<s}".format(fileout)
    return

def hist_queen(celldicts, labels, outputdir='.', channels=[0,1], bins=None, colors=None, units_dx=None, titles=None, mode='total_fl',qcut=0, xminxmax=None, backgrounds=None):
    """
    Make an histogram of the signal obtained per cell for the fluorescence channels used to measure the Queen ratio. Also plot the distribution of the Queen ratio.
    """

    # initialization
    if mode == 'concentration_fl':
        print "concentration_fl mode"
        fl_dtype = np.float_
        fmt = "$\\mu = {mu:,.2f}$\n$\\sigma = {sig:,.2f}$\n$N = {N:,d}$\n$\\mathrm{{med}} = {med:,.2f}$"
    elif mode == 'total_fl':
        print "total_fl mode"
        fl_dtype = np.uint16
        fmt = "$\\mu = {mu:,d}$\n$\\sigma = {sig:,d}$\n$N = {N:,d}$\n$\\mathrm{{med}} = {med:,d}$"
    else:
        raise ValueError('Wrong mode selection: \'total_fl\' or \'concentration_fl\'')

    # input data
    ndata = len(celldicts)
    if ndata == 0:
        raise ValueError("Empty input data")
    if ndata != len(labels):
        raise ValueError("Labels should have the same dimension as cell dicts")
    print "ndata = {:d}".format(ndata)

    # channels
    nchannel = 2
    c1 = channels[0]
    c2 = channels[1]
    print "nchannel = {:d}, c1 = {:d}, c2 = {:d}".format(nchannel, c1, c2)
    nplot = nchannel + 1

    # colors
    if colors is None:
        colors = colornames.keys()[:ndata]

    # bins
    if bins is None:
        bins={i: None for i in range(nplot)}
    for c in range(nplot):
        if bins[c] is None:
            bins[c] = ['auto']*ndata

    # units_dx
    if units_dx is None:
        units_dx={i:None for i in range(nplot)}

    # titles
    if titles is None:
        titles=[None for i in range(nplot)]

    # xminxmax
    if xminxmax is None:
        xminxmax=[[None, None] for i in range(nplot)]

    # filling up the data
    data_I1 = []
    data_I2 = []
    data_BG1 = []
    data_BG2 = []
    data_queen = []
    for i in range(ndata):
        cells = celldicts[i]
        ncells = len(cells)
        if ncells == 0:
            raise ValueError("Empty cell dictionary!")
        FL1 = []
        FL2 = []
        BG1 = []
        BG2 = []
        QUEEN = []

        # make lists
        keys = cells.keys()
        for n in range(ncells):
            key = keys[n]
            cell = cells[key]
            npx = cell['area']
            fl = cell['fluorescence']['total']
            bg_px = cell['fluorescence']['background_px']
            x1 = fl[c1]
            x2 = fl[c2]
            bg_x1 = bg_px[c1]*npx
            bg_x2 = bg_px[c2]*npx
            if mode == 'concentration_fl':
                try:
                    volume = cell['volume']
                except KeyError:
                    raise ValueError('Missing volume attribute in cell!')
                x1 = float(x1) / volume
                x2 = float(x2) / volume
                if backgrounds is None:
                    bg_x1 = float(bg_x1) / volume
                    bg_x2 = float(bg_x2) / volume
                else:
                    bg_x1  = backgrounds[c1]
                    bg_x2  = backgrounds[c2]
            elif mode == 'total_fl':
                sys.exit("Not programmed yet!")
                pass
            z = ((float(x1) -  float(bg_x1)) / (float(x2) - float(bg_x2)) )
            FL1.append(x1)
            FL2.append(x2)
            BG1.append(bg_x1)
            BG2.append(bg_x2)
            QUEEN.append(z)
        # end loop on cells
        data_I1.append(np.array(FL1))
        data_I2.append(np.array(FL2))
        data_BG1.append(np.array(BG1))
        data_BG2.append(np.array(BG2))
        data_queen.append(np.array(QUEEN))
    # end loop on data sets

    if backgrounds is None:
        bgcolor='r'
        bgs = [np.nanmedian(np.concatenate(data_BG1)), np.nanmedian(np.concatenate(data_BG2))]
    else:
        bgcolor='g'
        bgs = [np.float_(backgrounds[c]) for c in range(2)]

    # make figure
    fig = plt.figure(num=None, facecolor='w', figsize=(4*nplot,3))
    gs = mgs.GridSpec(1,nplot)

    ax0 = fig.add_subplot(gs[0,0])
    axes = [ax0]
    for c in range(1,nplot):
        ax = fig.add_subplot(gs[0,c])
        axes.append(ax)

    # plot fluorescence
    data=[data_I1, data_I2]
    for c in [c1,c2]:
        print "channel = {:d} / {:d}".format(c,1)
        ax=axes[c]
        for i in range(ndata):
            print "{:<2s}data {:d} / {:d}".format("", i, ndata-1)
            # compute histogram
            d = data[c][i]
            N = len(d)
            d = np.sort(d)
            n0 = int(0.5*qcut*float(N))
            print "{:<4s}qcut = {:.1f} %".format("",qcut*100)
            n1 = N - n0
#            print n0, n1
            d = d[n0:n1]
            hist,edges = np.histogram(d, bins=bins[c][i], density=True)
            print "{:<4s}nbins = {:,d}".format("",len(edges)-1)

            # plot histogram
            color = colors[i]
            #ax.bar(edges[:-1], hist, np.diff(edges), color='none', edgecolor=color, lw=0.5, label=labels[i])
            if (c ==0):
                label=labels[i]
            else:
                label=None

            ax.plot(0.5*(edges[:-1]+edges[1:]), hist, '-', color=color, lw=0.5, label=label)

        # plot background
        ax.axvline(x=bgs[c], color=bgcolor, lw=0.5, ls='--')
        print "background = {:.2f}".format(bgs[c])
        # end loop on data

    # plot Queen ratio
    ax=axes[2]
    for i in range(ndata):
        print "{:<2s}data {:d} / {:d}".format("", i, ndata-1)
        # compute histogram
        d = data_queen[i]
        N = len(d)
        d = np.sort(d)
        n0 = int(0.5*qcut*float(N))
        print "{:<4s}qcut = {:.1f} %".format("",qcut*100)
        n1 = N - n0
#            print n0, n1
        d = d[n0:n1]
        hist,edges = np.histogram(d, bins=bins[2][i], density=True)
        print "{:<4s}nbins = {:,d}".format("",len(edges)-1)

        # plot histogram
        color = colors[i]
        #ax.bar(edges[:-1], hist, np.diff(edges), color='none', edgecolor=color, lw=0.5, label=labels[i])
        if (c ==0):
            label=labels[i]
        else:
            label=None

        ax.plot(0.5*(edges[:-1]+edges[1:]), hist, '-', color=color, lw=0.5, label=label)

    # customize axes
    for c in range(nplot):
        ax=axes[c]
        # add legends
        if not (titles[c] is None):
            ax.set_title(titles[c], fontsize='large')

        # adjust the axis
        if (c == 0):
            ax.legend(loc='best',fontsize="medium",frameon=False)
        #ax.set_ylabel("pdf",fontsize="medium",labelpad=10)
        ax.tick_params(length=4)
        ax.tick_params(axis='both', labelsize='medium', labelleft=False, left=False)

        if not (units_dx[c] is None):
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dx[c]))
#        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))
#        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
#        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))

        ax.set_xlim(xminxmax[c])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    # end loop on channels

    gs.tight_layout(fig)
    if mode == 'concentration_fl':
        filename = 'analysis_overlay_queen_concentration_fl'
    elif mode == 'total_fl':
        filename = 'analysis_overlay_queen_total_fl'

    exts=['.pdf', '.svg', '.png']
    for ext in exts:
        fileout = os.path.join(outputdir,filename+ext)
        fig.savefig(fileout, bbox_inches='tight', pad_inches=0)
        print "Fileout: {:<s}".format(fileout)
    return

def hist_queen_old(celldicts, labels, outputdir='.', bins=['auto','auto','auto'], channels=[0,1], colors=['darkblue', 'darkgreen', 'darkblue'], units_dx=None, qcut=0):
    """
    Make a plot of the QUEEN signal obtained from the input dictionary of cells.
    """

    # initialization
    data = []
    c1 = channels[0]
    c2 = channels[1]
    fmt = "$\\mu = {mu:.2f}$\n$\\sigma = {sig:.2f}$\n$N = {N:,d}$\n$\\mathrm{{med}} = {med:.2f}$"

    # filling up the data
    ndata = len(celldicts)
    if ndata == 0:
        raise ValueError("Empty input data")
    print "ndata = {:d}".format(ndata)

    for i in range(ndata):
        cells = celldicts[i]
        ncells = len(cells)
        if ncells == 0:
            raise ValueError("Empty cell dictionary!")
#        I1 = []
#        I2 = []
        QUEEN=[]

        # make lists
        keys = cells.keys()
        for n in range(ncells):
            key = keys[n]
            cell = cells[key]
            fl = cell['fluorescence']['total']
            bg_px = cell['fluorescence']['background_px']
            npx = cell['area']
            x = fl[c1]
            bg_x = bg_px[c1]*npx
            y = fl[c2]
            bg_y = bg_px[c2]*npx
            z = (float(x)-float(bg_x))/(float(y)-float(bg_y))
#            I1.append(x)
#            I2.append(y)
            QUEEN.append(z)
            cell['queen_ratio']=z

#        I1 = np.array(I1)
#        I2 = np.array(I2)
        QUEEN = np.array(QUEEN, dtype=np.float_)
        data.append(QUEEN)

    # make plot
    Ns = [len(d) for d in data]
    mus = [np.mean(d).astype(d.dtype) for d in data]
    meds = [np.median(d).astype(d.dtype) for d in data]
    sigs = [np.std(d).astype(d.dtype) for d in data]
    errs = [s/np.sqrt(N) for s,N in zip(sigs,Ns)]

    # make figure
    fig = plt.figure(num=None, facecolor='w', figsize=(4,3))
    ax = fig.gca()
    for i in range(ndata):
        # compute histogram
        d = data[i]
        N = len(d)
        d = np.sort(d)
        n0 = int(qcut*float(N))
        n1 = min(int((1.-qcut)*float(N)),N-1)
        d = d[n0:n1+1]
        hist,edges = np.histogram(d, bins=bins[i], density=True)
        print "nbins = {:,d}".format(len(edges)-1)

        # plot histogram
        color = colors[i]
        #ax.bar(edges[:-1], hist, np.diff(edges), color='none', edgecolor=color, lw=0.5, label=labels[i])
        ax.plot(0.5*(edges[:-1]+edges[1:]), hist, '-', color=color, lw=0.5, label=labels[i])
    # end loop

    # add legends
    ax.set_title("QUEEN ratio", fontsize='large')
    #ax.annotate(fmts[i].format(mu=mus[i],sig=sigs[i], N=len(data[i]), med=meds[i]), xy=(0.70,0.98), xycoords='axes fraction', ha='left', va='top')

    # adjust the axis
    ax.legend(loc='best',fontsize="medium",frameon=False)
    #ax.set_ylabel("pdf",fontsize="medium",labelpad=10)
    ax.tick_params(length=4)
    ax.tick_params(axis='both', labelsize='medium')
    ax.tick_params(axis='both', labelsize='medium', labelleft=False, left=False)
    ax.set_ylim(0.,None)

    if not (units_dx is None):
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dx))
#        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))
#        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
#        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

#        ax.set_xlim(xmin,xmax)
#        ax.set_ylim(xmin,xmax)
#        ax.set_aspect(aspect='equal', adjustable='box')

    fig.tight_layout()
    filename = 'analysis_overlay_queen'

    exts=['.pdf', '.svg', '.png']
    for ext in exts:
        fileout = os.path.join(outputdir,filename+ext)
        fig.savefig(fileout, bbox_inches='tight', pad_inches=0)
        print "Fileout: {:<s}".format(fileout)
    return

#################### main ####################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Analysis tool -- Overlay of several data sets.")
    parser.add_argument('cellfiles',  type=str, nargs='+', help='Path to a cell dictionary in json format.')
    parser.add_argument('-f', '--paramfile',  type=file, required=False, help='Yaml file containing parameters.')
    parser.add_argument('-d', '--outputdir',  type=str, required=False, help='Output directory')
    parser.add_argument('--labels',  type=str, nargs='+', required=False, help='Add labels')

    # INITIALIZATION
    # load arguments
    namespace = parser.parse_args(sys.argv[1:])

    # input cell files
    cellfiles = []
    for f in namespace.cellfiles:
        cellfile = os.path.realpath(f)
        if not os.path.isfile(cellfile):
            raise ValueError("Cell file does not exist! {:<s}".format(cellfile))
        cellfiles.append(cellfile)

    nfiles = len(cellfiles)
    celldicts=[]
    for n in range(nfiles):
        cellfile = cellfiles[n]
        print cellfile
        cells = load_json2dict(cellfile)
        ncells = len(cells)
        print "ncells = {:d}".format(ncells)
        celldicts.append(cells)

    # labels
    labels = []
    labels = [None for n in range(nfiles)]
    if not (namespace.labels is None):
        for i,l in enumerate(namespace.labels):
            labels[i] = l

    # output directory
    outputdir = namespace.outputdir
    if (outputdir is None):
        outputdir = os.getcwd()
    else:
        outputdir = os.path.relpath(outputdir, os.getcwd())
    outputdir = os.path.relpath(outputdir, os.getcwd())
    rootdir = os.path.join(outputdir,'analysis')
    outputdir = os.path.join(rootdir,'overlays')
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
    print "{:<20s}{:<s}".format("outputdir", outputdir)

    # parameter file
    if namespace.paramfile is None:
        params = default_parameters()
        paramfile = "analysis_overlays_default.yml"
        with open(paramfile,'w') as fout:
            yaml.dump(params,fout)
    else:
        paramfile = namespace.paramfile.name
        params = yaml.load(namespace.paramfile)

    dest = os.path.join(outputdir, os.path.basename(paramfile))
    if (os.path.realpath(dest) != os.path.realpath(paramfile)):
        shutil.copy(paramfile,dest)
    paramfile = dest

    # make queen analysis
    if 'queen' in params:
        hist_queen(celldicts, labels, outputdir=outputdir, **params['queen'])

    if 'fluorescence' in params:
        hist_channel(celldicts, labels, outputdir=outputdir, **params['fluorescence'])

    if 'dimensions' in params:
        hist_dimensions(celldicts, labels, outputdir=outputdir, **params['dimensions'])

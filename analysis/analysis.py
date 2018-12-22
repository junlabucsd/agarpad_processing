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
    # dimensions
    params['dimensions']={}
    mydict = params['dimensions']
    mydict['bins'] = ['auto', 'auto', 'auto', 'auto']
    mydict['units_dx']=[None, None, None, None]
    mydict['mode']='um'
    # queen
    params['queen']={}
    mydict = params['queen']
    mydict['channels'] = [0,1]
    mydict['bins'] = [16, 16, 16]
    mydict['titles'] = ['390 ex', '475 ex', 'QUEEN']
    mydict['colors']=['darkblue', 'darkgreen', 'darkblue']
    mydict['units_dx']=[None, None, None]
    mydict['mode']='total_fl'  # alternative value is \'concentration_fl\'

    return params

def hist_dimensions(cells, outputdir='.', bins=['auto','auto','auto','auto'], units_dx=[None, None, None, None], mode='um'):
    """
    Plot the histograms for several cell dimensions
    """

    # initialization
    if mode == 'um':
        titles = [u'length (\u03BCm)', u'width (\u03BCm)', u'area (\u03BCm\u00B2)', u'volume (\u03BCm\u00B3)']
        attrs = ['height_um','width_um','area_um2', 'volume_um3']
    else:
        titles = [u'length (px)', u'width (px)', u'area (px\u00B2)', u'volume (px\u00B3)']
        attrs = ['height','width','area', 'volume']

    nattrs = len(attrs)
    if len(bins) != nattrs:
        raise ValueError("bins has the wrong dimensions!")
    if len(units_dx) != nattrs:
        raise ValueError("units_dx has the wrong dimensions!")
    ncells = len(cells)

    if ncells == 0:
        raise ValueError("Empty cell dictionary!")

    pfmt = "$\\mu = {mu:,.2f}$\n$\\sigma = {sig:,.2f}$\n$N = {N:,d}$\n$\\mathrm{{med}} = {med:,.2f}$"

    # make lists
    data = [ [] for i in range(nattrs)]
    keys = cells.keys()
    for n in range(ncells):
        key = keys[n]
        cell = cells[key]
        for i in range(nattrs):
            attr = attrs[i]
            data[i].append(cell[attr])

    data = np.array(data)

    # make plot
    x,N = data.shape
    mus = [np.mean(d).astype(d.dtype) for d in data]
    meds = [np.median(d).astype(d.dtype) for d in data]
    sigs = [np.std(d).astype(d.dtype) for d in data]
    errs = [s/np.sqrt(N) for s in sigs]

    fig = plt.figure(num=None, facecolor='w', figsize=(nattrs*4,3))
    gs = mgs.GridSpec(1,nattrs)

    ax0 = fig.add_subplot(gs[0,0])
    axes = [ax0]
    for i in range(1,nattrs):
        ax = fig.add_subplot(gs[0,i],sharey=ax0)
        axes.append(ax)

    for i in range(nattrs):
        attr = attrs[i]
        print "attr = {}".format(attr)
        ax = axes[i]

        # compute histogram
        hist,edges = np.histogram(data[i], bins=bins[i], density=False)
        nbins = len(edges)-1
        print "nbins = {:d}".format(nbins)

        # plot histogram
        color = 'grey'
        ax.bar(edges[:-1], hist, np.diff(edges), facecolor=color, lw=0)

        # add legends
        ax.set_title(titles[i], fontsize='large')
        ax.annotate(pfmt.format(mu=mus[i],sig=sigs[i], N=len(data[i]), med=meds[i]), xy=(0.70,0.98), xycoords='axes fraction', ha='left', va='top')

        # adjust the axis
        ax.tick_params(length=4)
        if (i==0):
            ax.set_ylabel("count",fontsize="medium",labelpad=10)
            ax.tick_params(axis='both', labelsize='medium')
        else:
            ax.tick_params(axis='both', labelsize='medium', labelleft='off')

        if not (units_dx[i] is None):
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dx[i]))
#        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))
#        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
#        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

#        ax.set_xlim(xmin,xmax)
#        ax.set_ylim(xmin,xmax)
#        ax.set_aspect(aspect='equal', adjustable='box')

    gs.tight_layout(fig, w_pad=1.0)
    if mode == 'um':
        filename = 'analysis_dimensions_um'
    else:
        filename = 'analysis_dimensions_px'

    exts=['.pdf', '.svg', '.png']
    for ext in exts:
        fileout = os.path.join(outputdir,filename+ext)
        fig.savefig(fileout, bbox_inches='tight', pad_inches=0)
        print "Fileout: {:<s}".format(fileout)
    return

def hist_channels(cells, outputdir='.', bins=None, units_dx=None, titles=None, mode='total',qcut=0, bg_val=None):
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
    data_fl = []
    data_bg = []

    # filling up the fluorescence
    cellref = cells.values()[0]
    nchannel = len(cellref['fluorescence']['total'])

    if bins is None:
        bins = ['auto']*nchannel
    if units_dx is None:
        units_dx = [None]*nchannel
    if titles is None:
        titles = ["channel {:d}".format(i) for i in range(nchannel)]

    for i in range(nchannel):
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
            x = fl[i]
            bg = bg_px[i]*npx
            if mode == 'concentration_fl':
                try:
                    volume = cell['volume']
                except KeyError:
                    raise ValueError('Missing volume attribute in cell!')
                x = float(x) / float(volume)
                bg = float(bg) / float(volume)
            elif mode == 'total_fl':
                pass
            FL.append(x)
            BG.append(bg)
        # end loop on cells
        FL = np.array(FL,dtype=fl_dtype)
        BG = np.array(BG,dtype=fl_dtype)
        data_fl.append(FL)
        data_bg.append(BG)
    # end loop on data sets

    Ns = [len(d) for d in data_fl]
    mus = [np.mean(d).astype(d.dtype) for d in data_fl]
    meds = [np.median(d).astype(d.dtype) for d in data_fl]
    sigs = [np.std(d).astype(d.dtype) for d in data_fl]
    errs = [s/np.sqrt(N) for s,N in zip(sigs,Ns)]
    bgs = [np.median(d).astype(d.dtype) for d in data_bg]

    # make figure
    fig = plt.figure(num=None, facecolor='w', figsize=(nchannel*4,3))
    gs = mgs.GridSpec(1,nchannel)

    ax0 = fig.add_subplot(gs[0,0])
    axes = [ax0]
    for i in range(1,nchannel):
        ax = fig.add_subplot(gs[0,i],sharey=ax0)
        axes.append(ax)

    for i in range(nchannel):
        print "channel = {}".format(i)
        ax = axes[i]

        # compute histogram
        d = data_fl[i]
        N = len(d)
        d = np.sort(d)
        n0 = int(qcut*float(N))
        n1 = min(int((1.-qcut)*float(N)),N-1)
        d = d[n0:n1+1]
        hist,edges = np.histogram(d, bins=bins[i], density=False)
        nbins = len(edges)-1
        print "nbins = {:d}".format(nbins)

        # plot histogram
        color = 'grey'
        ax.bar(edges[:-1], hist, np.diff(edges), facecolor=color, lw=0)

        # add legends
        ax.set_title(titles[i], fontsize='large')
        ax.annotate(fmt.format(mu=mus[i],sig=sigs[i], N=len(data_fl[i]), med=meds[i]), xy=(0.70,0.98), xycoords='axes fraction', ha='left', va='top')
        ax.axvline(x=bgs[i], color='r', lw=0.5, ls='--')

        # adjust the axis
        ax.tick_params(length=4)
        if (i==0):
            ax.set_ylabel("count",fontsize="medium",labelpad=10)
            ax.tick_params(axis='both', labelsize='medium')
        else:
            ax.tick_params(axis='both', labelsize='medium', labelleft='off')

        if not (units_dx[i] is None):
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dx[i]))
#        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))
#        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
#        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

#        ax.set_xlim(xmin,xmax)
#        ax.set_ylim(xmin,xmax)
#        ax.set_aspect(aspect='equal', adjustable='box')

        if not (titles[i] is None):
            ax.set_title(titles[i], fontsize='large')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    # end loop

    fig.tight_layout()
    if mode == 'concentration_fl':
        filename = 'analysis_concentration_fl'
    elif mode == 'total_fl':
        filename = 'analysis_total_fl'

    exts=['.pdf', '.svg', '.png']
    for ext in exts:
        fileout = os.path.join(outputdir,filename+ext)
        fig.savefig(fileout, bbox_inches='tight', pad_inches=0)
        print "Fileout: {:<s}".format(fileout)
    return

def make_plot_queen(cells, outputdir='.', bins=['auto','auto','auto'], titles=['I1','I2','QUEEN'], channels=[0,1], colors=['darkblue', 'darkgreen', 'darkblue'], units_dx=[None, None, None], mode='total_fl'):
    """
    Make a plot of the QUEEN signal obtained from the input dictionary of cells.
    """

    # initialization
    ncells = len(cells)
    if ncells == 0:
        raise ValueError("Empty cell dictionary!")
    c1 = channels[0]
    c2 = channels[1]
    I1 = []
    I2 = []
    QUEEN=[]

    if mode == 'concentration_fl':
        print "concentration_fl mode"
        fl_dtype = np.float_
        fl_fmt = "$\\mu = {mu:,.2f}$\n$\\sigma = {sig:,.2f}$\n$N = {N:,d}$\n$\\mathrm{{med}} = {med:,.2f}$"

    elif mode == 'total_fl':
        print "total_fl mode"
        fl_dtype = np.uint16
        fl_fmt = "$\\mu = {mu:,d}$\n$\\sigma = {sig:,d}$\n$N = {N:,d}$\n$\\mathrm{{med}} = {med:,d}$"
    else:
        raise ValueError('Wrong mode selection: \'total_fl\' or \'concentration_fl\'')
    # make lists
    keys = cells.keys()
    for n in range(ncells):
        key = keys[n]
        cell = cells[key]
        fl = cell['fluorescence']['total']
        bg = cell['fluorescence']['background']
        x = fl[c1]-bg[c1]
        y = fl[c2]-bg[c2]
        if mode == 'concentration_fl':
            try:
                volume = cell['volume']
            except KeyError:
                raise ValueError('Missing volume attribute in cell!')
            x = float(x) / volume
            y = float(y) / volume
        elif mode == 'total_fl':
            pass
        z = float(x)/float(y)
        I1.append(x)
        I2.append(y)
        QUEEN.append(z)
        cell['queen_ratio']=z

    I1 = np.array(I1, dtype=fl_dtype)
    I2 = np.array(I2, dtype=fl_dtype)
    QUEEN = np.array(QUEEN, dtype=np.float_)

    # make plot
    N = len(I1)
    data = [I1, I2, QUEEN]
    mus = [np.mean(d).astype(d.dtype) for d in data]
    meds = [np.median(d).astype(d.dtype) for d in data]
    sigs = [np.std(d).astype(d.dtype) for d in data]
    errs = [s/np.sqrt(N) for s in sigs]
    fmts = [fl_fmt,fl_fmt,"$\\mu = {mu:.2f}$\n$\\sigma = {sig:.2f}$\n$N = {N:,d}$\n$\\mathrm{{med}} = {med:.2f}$"]

    fig = plt.figure(num=None, facecolor='w', figsize=(3*4,3))
    gs = mgs.GridSpec(1,3)

    ax0 = fig.add_subplot(gs[0,0])
    axes = [ax0]
    for i in range(1,3):
        ax = fig.add_subplot(gs[0,i],sharey=ax0)
        axes.append(ax)

    for i in range(3):
        ax = axes[i]

        # compute histogram
        hist,edges = np.histogram(data[i], bins=bins[i], density=False)
        nbins = len(edges)-1
        print "nbins = {:d}".format(nbins)

        # plot histogram
        color = colors[i]
        ax.bar(edges[:-1], hist, np.diff(edges), facecolor=color, lw=0)

        # add legends
        ax.set_title(titles[i], fontsize='large')
        ax.annotate(fmts[i].format(mu=mus[i],sig=sigs[i], N=len(data[i]), med=meds[i]), xy=(0.70,0.98), xycoords='axes fraction', ha='left', va='top')

        # adjust the axis
        if (i==0):
            ax.set_ylabel("count",fontsize="medium",labelpad=10)
            ax.tick_params(length=4)
            ax.tick_params(axis='both', labelsize='medium')
        else:
            ax.tick_params(axis='both', labelsize='medium', labelleft='off')

        if not (units_dx[i] is None):
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dx[i]))
#        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))
#        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
#        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

#        ax.set_xlim(xmin,xmax)
#        ax.set_ylim(xmin,xmax)
#        ax.set_aspect(aspect='equal', adjustable='box')

    gs.tight_layout(fig, w_pad=1.0)
    if mode == 'concentration_fl':
        filename = 'analysis_queen_concentration_fl'
    elif mode == 'total_fl':
        filename = 'analysis_queen_total_fl'

    exts=['.pdf', '.svg', '.png']
    for ext in exts:
        fileout = os.path.join(outputdir,filename+ext)
        fig.savefig(fileout, bbox_inches='tight', pad_inches=0)
        print "Fileout: {:<s}".format(fileout)
    return

#################### main ####################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Analysis tool.")
    parser.add_argument('cellfile',  type=str, help='Path to a cell dictionary in json format.')
    parser.add_argument('-f', '--paramfile',  type=file, required=False, help='Yaml file containing parameters.')
    parser.add_argument('-d', '--outputdir',  type=str, required=False, help='Output directory')
    parser.add_argument('--debug',  action='store_true', required=False, help='Enable debug mode')

    # INITIALIZATION
    # load arguments
    namespace = parser.parse_args(sys.argv[1:])

    # input cell file
    cellfile = os.path.realpath(namespace.cellfile)
    if not os.path.isfile(cellfile):
        raise ValueError("Cell file does not exist! {:<s}".format(cellfile))

    cells = load_json2dict(cellfile)
    ncells = len(cells)
    print "ncells = {:d}".format(ncells)

    # output directory
    outputdir = namespace.outputdir
    if (outputdir is None):
        outputdir = os.path.dirname(cellfile)
    else:
        outputdir = os.path.relpath(outputdir, os.getcwd())
    outputdir = os.path.join(outputdir,'analysis')
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
    print "{:<20s}{:<s}".format("outputdir", outputdir)

    # parameter file
    if namespace.paramfile is None:
        allparams = default_parameters()
        paramfile = "analysis.yml"
        with open(paramfile,'w') as fout:
            yaml.dump(allparams,fout)
    else:
        paramfile = namespace.paramfile.name
        allparams = yaml.load(namespace.paramfile)

    dest = os.path.join(outputdir, os.path.basename(paramfile))
    if (os.path.realpath(dest) != os.path.realpath(paramfile)):
        shutil.copy(paramfile,dest)
    paramfile = dest
    # dimensions
    if 'dimensions' in allparams:
        mydir = os.path.join(outputdir,'dimensions')
        if not os.path.isdir(mydir):
            os.makedirs(mydir)
        print "{:<20s}{:<s}".format("outputdir", mydir)
        hist_dimensions(cells, outputdir=mydir, **allparams['dimensions'])

    # make queen analysis
    if 'fluorescence' in allparams:
        mydir = os.path.join(outputdir,'fluorescence')
        if not os.path.isdir(mydir):
            os.makedirs(mydir)
        print "{:<20s}{:<s}".format("outputdir", mydir)
        hist_channels(cells, outputdir=mydir, **allparams['fluorescence'])

    # make queen analysis
    if 'queen' in allparams:
        mydir = os.path.join(outputdir,'queen')
        if not os.path.isdir(mydir):
            os.makedirs(mydir)
        print "{:<20s}{:<s}".format("outputdir", mydir)
        make_plot_queen(cells, outputdir=mydir, **allparams['queen'])


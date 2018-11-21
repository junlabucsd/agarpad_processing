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

def hist_channel(celldicts, labels, outputdir='.', bins=['auto','auto','auto'], channel=0, colors=[None,None,None], units_dx=None, title=None, mode='total',qcut=0):
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
    data = []

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
        FL = []

        # make lists
        keys = cells.keys()
        for n in range(ncells):
            key = keys[n]
            cell = cells[key]
            fl = cell['fluorescence']['total']
            bg = cell['fluorescence']['background']
            x = fl[channel]-bg[channel]
            if mode == 'concentration_fl':
                try:
                    volume = cell['volume']
                except KeyError:
                    raise ValueError('Missing volume attribute in cell!')
                x = float(x) / volume
            elif mode == 'total_fl':
                pass
            FL.append(x)
        # end loop on cells
        FL = np.array(FL,dtype=fl_dtype)
        data.append(FL)
    # end loop on data sets

    Ns = [len(d) for d in data]
    mus = [np.mean(d).astype(d.dtype) for d in data]
    meds = [np.median(d).astype(d.dtype) for d in data]
    sigs = [np.std(d).astype(d.dtype) for d in data]
    errs = [s/np.sqrt(N) for s,N in zip(sigs,Ns)]

    print meds
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
    if not (title is None):
        ax.set_title(title, fontsize='large')
    #ax.annotate(fmts[i].format(mu=mus[i],sig=sigs[i], N=len(data[i]), med=meds[i]), xy=(0.70,0.98), xycoords='axes fraction', ha='left', va='top')

    # adjust the axis
    ax.legend(loc='best',fontsize="medium",frameon=False)
    ax.set_ylabel("pdf",fontsize="medium",labelpad=10)
    ax.tick_params(length=4)
    ax.tick_params(axis='both', labelsize='medium')
    ax.tick_params(axis='both', labelsize='medium', labelleft='off')

    if not (units_dx is None):
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dx))
#        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))
#        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
#        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()
    if mode == 'concentration_fl':
        filename = 'analysis_overlay_concentration_fl'
    elif mode == 'total_fl':
        filename = 'analysis_overlay_total_fl'
    filename += "_c{:d}".format(channel)

    exts=['.pdf', '.svg', '.png']
    for ext in exts:
        fileout = os.path.join(outputdir,filename+ext)
        fig.savefig(fileout, bbox_inches='tight', pad_inches=0)
        print "Fileout: {:<s}".format(fileout)
    return

def hist_queen(celldicts, labels, outputdir='.', bins=['auto','auto','auto'], channels=[0,1], colors=['darkblue', 'darkgreen', 'darkblue'], units_dx=None,qcut=0):
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
        I1 = []
        I2 = []
        QUEEN=[]

        # make lists
        keys = cells.keys()
        for n in range(ncells):
            key = keys[n]
            cell = cells[key]
            fl = cell['fluorescence']['total']
            bg = cell['fluorescence']['background']
            x = fl[c1]-bg[c1]
            y = fl[c2]-bg[c2]
            z = float(x)/float(y)
            I1.append(x)
            I2.append(y)
            QUEEN.append(z)
            cell['queen_ratio']=z

        I1 = np.array(I1)
        I2 = np.array(I2)
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
    ax.set_ylabel("pdf",fontsize="medium",labelpad=10)
    ax.tick_params(length=4)
    ax.tick_params(axis='both', labelsize='medium')
    ax.tick_params(axis='both', labelsize='medium', labelleft='off')

    if not (units_dx is None):
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=units_dx))
#        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))
#        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
#        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=0.1))

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
    parser = argparse.ArgumentParser(prog="Analysis tool -- Overlay of several data set.")
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
        allparams = default_parameters()
        paramfile = "analysis_overlays_default.yml"
        with open(paramfile,'w') as fout:
            yaml.dump(allparams,fout)
    else:
        paramfile = namespace.paramfile.name
        allparams = yaml.load(namespace.paramfile)

    dest = os.path.join(outputdir, os.path.basename(paramfile))
    if (os.path.realpath(dest) != os.path.realpath(paramfile)):
        shutil.copy(paramfile,dest)
    paramfile = dest

    # make queen analysis
    params=allparams['analysis_overlays']
    if 'hist_queen' in params:
        hist_queen(celldicts, labels, outputdir=outputdir, **params['hist_queen'])

    if 'hist_channel' in params:
        for channel in params['hist_channel']['channels']:
            hist_channel(celldicts, labels, outputdir=outputdir, channel=channel, **params['hist_channel']['args'])

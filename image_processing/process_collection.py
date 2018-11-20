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
    params['process_collection']={}
    mydict = params['process_collection']
    mydict['new_attributes'] = []

    return params

#################### main ####################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Processing tool -- Collection of cells.")
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
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
    print "{:<20s}{:<s}".format("outputdir", outputdir)

    # parameter file
    if namespace.paramfile is None:
        allparams = default_parameters()
        paramfile = "process_collection.yml"
        with open(paramfile,'w') as fout:
            yaml.dump(allparams,fout)
    else:
        paramfile = namespace.paramfile.name
        allparams = yaml.load(namespace.paramfile)

    dest = os.path.join(outputdir, os.path.basename(paramfile))
    if (os.path.realpath(dest) != os.path.realpath(paramfile)):
        shutil.copy(paramfile,dest)
    paramfile = dest

    # PROCESS CELLS
    params = allparams['process_collection']
    new_attributes = params['new_attributes']
    mpp = params['mpp']
    keys = cells.keys()
    keys_sel = []
    for n in range(ncells):
        key = keys[n]
        cell = cells[key]
        print "Cell {:d} / {:d} : {:s}".format(n,ncells,cell['id'])

        # filtering operations

        # creating new attributes
        if 'volume' in new_attributes:
            # in cubic pixels
            w=cell['width']
            h=cell['height']
            cell['volume']=np.pi/4.* w**2*h - np.pi/12.*w**3 # cylinder with hemispherical caps of length h and width w

            # conversion to cubic micrometers
        if 'volume_um' in new_attributes:
            if mpp is None:
                try:
                    mpp = cell['mpp']
                except KeyError, ValueError:
                    print "Unit is lacking: mpp = 1! Skipping cell."
                    continue
            if not 'volume' in cell:
                print 'Volume must be computed for \'volume_um\''
                continue
            cell['volume_um']  = cell['volume']*mpp*mpp*mpp

        # if everything went correctly, add cell key to selection
        keys_sel.append(key)

    # enc loop

    # WRITE PROCESSED CELL FILE
    cells_new = {key: cells[key] for key in keys_sel}
    ncells_new = len(cells_new)
    print "ncells_new = {:d}".format(ncells_new)

    cells_new = make_dict_serializable(cells_new)
    bname = os.path.splitext(os.path.basename(cellfile))[0]
    fname = "{}_processed".format(bname)
    pathtocells = os.path.join(outputdir, fname + '.js')
    write_dict2json(pathtocells,cells_new)
    print "{:<20s}{:<s}".format('fileout', pathtocells)

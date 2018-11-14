#################### imports ####################
# standard
import sys
import os
import numpy as np
import yaml
import argparse
import shutil
#from PIL import Image
import tifffile as ti

# custom
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

debug=False

#################### function ####################
def default_parameters():
    """Generate a default parameter dictionary."""

    print "Loading default parameters"
    params={}
    # filtering - select only a subset
    params['segmentation']={}
    mydict = params['segmentation']
    mydict['method'] = 'contours'
    mydict['method_params'] = {\
            'contours': dict(w0=1, w1=100, l0=10, l1=1000, acut=0.9)\
            }
    mydict['channel'] = 0

    return params

def get_tiff2ndarray(tiff_file,channel=0):
    """
    Open a tiff_file and return a numpy array.
    """
    try:
        img = ti.imread(tiff_file)
    except:
        raise ValueError("Opening tiff with PIL failed.")

    naxis = len(img.shape)
    if naxis == 2:
        arr = img
    elif naxis == 3:
        arr = img[channel]
    else:
        raise ValueError("Hyperstacked handling not implemented.")

    # determine input mode
    dtype = arr.dtype
    if (dtype == np.uint8):
        norm = float(2**8-1)
    elif (dtype == np.uint16):
        norm = float(2**16-1)
    elif (dtype == np.uint32):
        norm = float(2**32-1)
    else:
        raise ValueError("Format not recognized.")
    arr = arr / norm

    return arr


def get_estimator(tiff_file, method='contours', outputdir='.', channel=0, method_params=dict(w0=1, w1=100, l0=10, l1=1000, acut=0.9)):
    """
    Compute the estimator for a given images. The estimator is a real matrix where each entry is the estimation that the corresponding pixel belongs to the segmented class.
    INPUT:
        * path to a tiff image.
    OUTPUT:
        * path to a matrix (written in a file) of same size of the orignal image.
    """
    # read the input tiff_file
    img = get_tiff2ndarray(tiff_file, channel=channel)

    # test
#    import matplotlib.pyplot as plt
#    plt.imshow(img)
#    fname = os.path.splitext(os.path.basename(tiff_file))[0]
#    fileout = os.path.join(outputdir,fname+'.png')
#    print fileout
#    plt.savefig(fileout)
    # test

    sys.exit()

    # perform the segmentation
    if method == 'contours':
        estimator = get_estimator_contours(img, **method_params)
    else:
        raise ValueError("Segmentation method not implemented.")
    return f

#################### main ####################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Segmentation tool -- cells.")
    parser.add_argument('tiffs',  type=str, nargs='+', help='tiff files to open.')
    parser.add_argument('-f', '--paramfile',  type=file, required=False, help='Yaml file containing parameters.')
    parser.add_argument('-d', '--outputdir',  type=str, required=False, help='Output directory')
    parser.add_argument('--debug',  action='store_true', required=False, help='Enable debug mode')

    # load arguments
    namespace = parser.parse_args(sys.argv[1:])

    # output directory
    outputdir = namespace.outputdir
    if (outputdir is None):
        outputdir = os.path.dirname(os.path.realpath())
    else:
        outputdir = os.path.relpath(outputdir, os.getcwd())
        if not os.path.isdir(outputdir):
            os.makedirs(outputdir)
    print "{:<20s}{:<s}".format("outputdir", outputdir)

    # parameter file
    if namespace.paramfile is None:
        allparams = default_parameters()
        paramfile = "segmentation_cells.yml"
        with open(paramfile,'w') as fout:
            yaml.dump(allparams,fout)
    else:
        paramfile = namespace.paramfile.name
        allparams = yaml.load(namespace.paramfile)

    dest = os.path.join(outputdir, os.path.basename(paramfile))
    if (os.path.realpath(dest) != os.path.realpath(paramfile)):
        shutil.copy(paramfile,dest)
    paramfile = dest

    # check that tiff files exists
    tiff_files = []
    for f in namespace.tiffs:
        test = check_tiff_file(f)
        if test:
            tiff_files.append(f)
    ntiffs = len(tiff_files)
    if ntiffs == 0:
        raise ValueError("Number of tiff files is zero!")
    else:
        print "{:<20s}{:<d}".format("ntiffs", ntiffs)

    # Segmentation
    params=allparams['segmentation']
    segmentation_method=params['method']

    ## make tiff index
    pathtoindex = os.path.join(outputdir,"index_tiffs.txt")
    paths = np.asarray([os.path.relpath(f,outputdir) for f in tiff_files])
    with open(pathtoindex,'w') as fout:
        np.savetxt(fout,paths,fmt='%s')
    print "{:<20s}{:<s}".format("fileout", pathtoindex)

    ## build estimator matrices
    estimator_dir=os.path.join(outputdir,'estimators')
    estimator_dir=os.path.join(estimator_dir, segmentation_method)
    if not os.path.isdir(estimator_dir):
        os.makedirs(estimator_dir)
    print "{:<20s}{:<s}".format("est. dir.", estimator_dir)
    est_files = []
    for f in tiff_files:
        ef=get_estimator(f, method=segmentation_method, outputdir=estimator_dir, method_params=params['method_params'][segmentation_method])
        est_files.append(ef)
    paths = np.asarray([os.path.relpath(ef,outputdir) for ef in est_files])
    pathtoindex=os.path.join(estimator_dir, "index_estimators.txt")
    with open(pathtoindex,'w') as fout:
        np.savetxt(fout,paths,fmt='%s')
    print "{:<20s}{:<s}".format("fileout", pathtoindex)



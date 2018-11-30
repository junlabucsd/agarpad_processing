#################### imports ####################
import sys,os
import numpy as np
import scipy.sparse as ssp
import json
import datetime
import argparse
import tifffile as ti
import cv2 as cv2

def check_tiff_file(tiff_file):
    """
    Check if a tiff file is valid.
    """
    test=True
    # existence of file
    if not os.path.isfile(tiff_file):
        test=False

    # return
    return test

def check_estimator_file(estimator_file):
    """
    Check if a estimator file is valid.
    """
    test=True
    # existence of file
    if not os.path.isfile(estimator_file):
        test=False
    # reading
    try:
        ssp.load_npz(estimator_file)
    except ValueError:
        test=False

    # return
    return test

def check_mask_file(mask_file):
    """
    Check if a mask file is valid.
    """
    test=True
    # existence of file
    if not os.path.isfile(mask_file):
        test=False
    # reading
    try:
        ssp.load_npz(mask_file)
    except ValueError:
        test=False

    # return
    return test

def check_labels_file(labels_file):
    """
    Check if a labels file is valid.
    """
    test=True
    # existence of file
    if not os.path.isfile(labels_file):
        test=False
    # reading
    try:
        ssp.load_npz(labels_file)
    except ValueError:
        test=False

    # return
    return test

def write_index(pathtoindex,index):
    fout = open(pathtoindex,'w')
    nfile = len(index)
    for n in range(nfile):
        tab = index[n]
        fout.write(','.join(tab)+'\n')
    fout.close()

def load_index(pathtoindex):
    fin = open(pathtoindex,'r')
    index = []
    while True:
        line = fin.readline()
        if line == "":
            break
        elif line == '\n':
            break
        else:
            line = line.replace(" ","").replace("\n","")
            tab = line.split(",")
            index.append(tab)
    return index

def write_dict2json(pathtojson,mydict):
    with open(pathtojson, 'w') as fout:
        json.dump(mydict, fout, sort_keys=True, indent=2)
    return

def load_json2dict(pathtojson):
    with open(pathtojson, 'r') as fin:
        mydict = json.load(fin)
    return mydict

def make_dict_serializable(mydict):
    for k, v in mydict.iteritems():
        if isinstance(v, dict):
            make_dict_serializable(v)
        else:
            if type(v) == np.ndarray:
                mydict[k]=v.tolist()
            elif type(v) == datetime.datetime:
                mydict[k]=v.strftime('%Y-%m-%d')
    return mydict

def get_img_norm(dtype):
    """
    Return the maximum value that can be taken in this image.
    """

    # determine input mode
    if (dtype == np.uint8):
        norm = float(2**8-1)
    elif (dtype == np.uint16):
        norm = float(2**16-1)
    elif (dtype == np.uint32):
        norm = float(2**32-1)
    else:
        raise ValueError("Format not recognized.")

    return norm

def get_tiff2ndarray(tiff_file,channel=0,metadata=False, normalize=True):
    """
    Open a tiff_file and return a numpy array normalized between 0 and 1.
    """
    try:
        with ti.TiffFile(tiff_file) as tif:
            img = tif.asarray()
            meta = tif.imagej_metadata
    except Exception, e:
        print e
        raise ValueError("Opening tiff with tiff failed.")

    naxis = len(img.shape)
    if naxis == 2:
        arr = img
    elif naxis == 3:
        if channel == None:
            arr = img
        else:
            arr = img[channel]
    else:
        raise ValueError("Hyperstacked handling not implemented.")

    if normalize:
        norm = get_img_norm(arr.dtype)
        arr = np.array(arr, dtype=np.float_) / norm

    if metadata:
        return arr, meta
    else:
        return arr

def get_OTSU(tiff_files, outputdir='.', write=True):
    """
    Read the input list of files and compute the OTSU threshold (normalized between 0 and 1).
    The result is written in the output directory.
    """
    # make data arrays per channels
    data = np.array([get_tiff2ndarray(t,channel=None) for t in tiff_files])
    nfiles, nchannels, height, width = data.shape
    data = np.moveaxis(data, source=1, destination=0)
    data = np.reshape(data, (nchannels, nfiles*height*width))

    # scale to 8-bit
    norm = float(2**8-1)
    data = np.array(norm*data, dtype=np.uint8)

    # compute threshold per channel
    thresholds = []
    for c in range(nchannels):
        thres8, res = cv2.threshold(data[c], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thres16 = np.uint16(float(thres8) / norm * float(2**16-1))
        thresholds.append([c, thres8,thres16])

    thresholds = np.array(thresholds)
    # write
    if write:
        header="channel, uint8, uint16"
        fileout = os.path.join(outputdir, "thresholds_otsu.txt")
        np.savetxt(fileout,thresholds, header=header,fmt='%-8d%-8d%-8d')
        print "{:<20s}{:<s}".format("fileout",fileout)

    # return
    return thresholds

#################### main ####################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Utility tools.")
    parser.add_argument('-f', '--paramfile',  type=file, required=False, help='Yaml file containing parameters.')
    parser.add_argument('-i', '--images',  type=str, nargs='+', required=False, help='tiff files to open.', default=[])
    parser.add_argument('--otsu',  action='store_true', required=False, help='Compute the OTSU thresholds for each channel.')
    parser.add_argument('-d', '--outputdir',  type=str, required=False, help='Output directory')

    # INITIALIZATION
    # load arguments
    namespace = parser.parse_args(sys.argv[1:])

    # output directory
    outputdir = namespace.outputdir
    if (outputdir is None):
        outputdir = os.getcwd()
    else:
        outputdir = os.path.relpath(outputdir, os.getcwd())
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
    print "{:<20s}{:<s}".format("outputdir", outputdir)

#    # parameter file
#    if namespace.paramfile is None:
#        allparams = default_parameters()
#        paramfile = "segmentation_cells.yml"
#        with open(paramfile,'w') as fout:
#            yaml.dump(allparams,fout)
#    else:
#        paramfile = namespace.paramfile.name
#        allparams = yaml.load(namespace.paramfile)
#
#    dest = os.path.join(outputdir, os.path.basename(paramfile))
#    if (os.path.realpath(dest) != os.path.realpath(paramfile)):
#        shutil.copy(paramfile,dest)
#    paramfile = dest
#
    # check that tiff files exists
    tiff_files = []
    for f in namespace.images:
        test = check_tiff_file(f)
        if test:
            tiff_files.append(f)
    ntiffs = len(tiff_files)
    print "ntiffs = {:d}".format(ntiffs)

    # OTSU thresholds
    if namespace.otsu and (ntiffs > 0):
        print "Computing OTSU thresholds..."
        thresholds = get_OTSU(tiff_files, outputdir=outputdir)


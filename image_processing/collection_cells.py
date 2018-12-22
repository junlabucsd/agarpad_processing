#################### imports ####################
# standard
import sys
import os
import numpy as np
import scipy.sparse as ssp
import yaml
import argparse
import shutil
import tifffile as ti
import cv2
import cPickle as pkl

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

#################### function ####################
def default_parameters():
    """Generate a default parameter dictionary."""

    print "Loading default parameters"
    params={}
    # filtering - select only a subset
    params['collection']={}
    mydict = params['collection']
    mydict['px2um'] = None

    return params

def check_segmentation(sdir):
    """
    Function checking that segmentation exists and is in order.
    INPUT:
      * sdir: segmentation directory
    OUTPUT:
      * boolean
    """
    print "{:<20s}{:<s}".format("segmentation dir:", sdir)

    # check directory existence
    if not os.path.isdir(sdir):
        print "Segmentation directory doesn\'t exist."
        return False

    # check index
    indexname='index_tiffs.txt'
    pathtoindex=os.path.join(sdir,indexname)
    if not os.path.isfile(pathtoindex):
        print "Index file doesn\'t exist: {:s}".format(indexname)
        return False

    # check existence of all files in the index
    index = load_index(pathtoindex)
    nfiles = len(index)
    for n in range(nfiles):
        f, ef, mf, lf = index[n]
        f = os.path.relpath(os.path.join(sdir, f))
        ef = os.path.relpath(os.path.join(sdir, ef))
        mf = os.path.relpath(os.path.join(sdir, mf))
        lf = os.path.relpath(os.path.join(sdir, lf))

        # check tiff
        if not check_tiff_file(f):
            print "Problem with file {:s}".format(f)
            return False

        # check estimator file
        if not check_estimator_file(ef):
            print "Problem with file {:s}".format(ef)
            return False

        # check mask file
        if not check_mask_file(mf):
            print "Problem with file {:s}".format(mf)
            return False

        # check labels file
        if not check_labels_file(lf):
            print "Problem with file {:s}".format(lf)
            return False

    return True

def write_crop(img, mask, points, bname, tiff_dir='.', mask_dir='.', pad_x=5, pad_y=5, debug=False):
    """
    Return a cropped image where the clipping mask is the bounding box of the input points.
    """
    shape= img.shape
    if len(shape) == 2:
        nchannel = None
        height,width = shape
    elif len(shape) == 3:
        nchannel,height,width = shape
    else:
        raise ValueError("Shape not implemented")
    filled = np.zeros((height,width),dtype=np.uint8)

    # upright rectangles
    bb = cv2.boundingRect(points)
    x,y,w,h=bb
    x0 = max(x-pad_x,0)
    y0 = max(y-pad_y,0)
    x1 = min(x + w + pad_x,width-1)
    y1 = min(y + h + pad_y,height-1)

    # crop
    submask= mask[y0:y1+1]
    submask = submask[:, x0:x1+1]
    if nchannel is None:
        subimg = img[:, y0:y1+1]
        subimg = subimg[:, x0:x1+1]
    else:
        subimg = img[:, y0:y1+1]
        subimg = subimg[:, :, x0:x1+1]

    if debug:
        debugdir = os.path.join(tiff_dir,'debug')
        if not os.path.isdir(debugdir):
            os.makedirs(debugdir)
        import matplotlib.pyplot as plt
        plt.subplot(151)
        plt.imshow((img[3]-np.min(img[3]))/(np.max(img[3])-np.min(img[3])), cmap='gray')
        plt.xticks([]),plt.yticks([])

        plt.subplot(152)
        plt.imshow(mask, cmap='gray')
        plt.xticks([]),plt.yticks([])

        plt.subplot(153)
        plt.imshow(filled, cmap='gray')
        plt.xticks([]),plt.yticks([])

        plt.subplot(154)
        if nchannel is None:
            plt.imshow(subimg, cmap='gray')
        else:
            plt.imshow(subimg[-1], cmap='gray')

        plt.subplot(155)
        plt.imshow(submask, cmap='gray')
        plt.xticks([]),plt.yticks([])

        plt.xticks([]),plt.yticks([])
        fileout = os.path.join(debugdir,bname+".png")
        plt.savefig(fileout, dpi=300, bbox_inches='tight', pad_inches=0)
        print "{:<20s}{:<s}".format('fileout',fileout)
        plt.close('all')

    # write tiff
    fileout = os.path.join(tiff_dir,bname+'.tif')
    ti.imwrite(fileout, subimg, imagej=True, photometric='minisblack')
    print "{:<20s}{:<s}".format('fileout',fileout)

    # write
    fileout = os.path.join(mask_dir,bname+'.tif')
    #fileout = os.path.join(mask_dir,bname+'.txt')
    #np.savetxt(fileout,submask)
    ti.imwrite(fileout, np.array(255*submask,dtype=np.uint8), imagej=True, photometric='minisblack')
    print "{:<20s}{:<s}".format('fileout',fileout)

    return

#################### main ####################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Segmentation tool -- cells.")
    parser.add_argument('-f', '--paramfile',  type=file, required=False, help='Yaml file containing parameters.')
    parser.add_argument('-d', '--outputdir',  type=str, required=False, help='Output directory')
    parser.add_argument('--lean',  action='store_true', required=False, help='Do not write crops for collection.')
    parser.add_argument('--debug',  action='store_true', required=False, help='Enable debug mode')

    # INITIALIZATION
    # load arguments
    namespace = parser.parse_args(sys.argv[1:])

    # output directory
    outputdir = namespace.outputdir
    if (outputdir is None):
        outputdir = os.path.dirname(os.path.realpath())
    else:
        outputdir = os.path.relpath(outputdir, os.getcwd())
    rootdir = outputdir
    outputdir = os.path.join(outputdir,'cells','collection')
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
    print "{:<20s}{:<s}".format("outputdir", outputdir)

    # parameter file
    if namespace.paramfile is None:
        allparams = default_parameters()
        paramfile = "collection_cells.yml"
        with open(paramfile,'w') as fout:
            yaml.dump(allparams,fout)
    else:
        paramfile = namespace.paramfile.name
        allparams = yaml.load(namespace.paramfile)

    dest = os.path.join(outputdir, os.path.basename(paramfile))
    if (os.path.realpath(dest) != os.path.realpath(paramfile)):
        shutil.copy(paramfile,dest)
    paramfile = dest

    params=allparams['collection']

    # debug or not
    if namespace.debug:
        print "!! Debug mode !!"

    # GET METADATA
    pathtometa = os.path.join(rootdir,'metadata.txt')
    if not os.path.isfile(pathtometa):
        raise ValueError("Metadata missing at {:s}".format(pathtometa,rootdir))
    meta = load_json2dict(pathtometa)

    # CHECK EXISTING SEGMENTATION
    seg_dir = os.path.relpath(os.path.join(outputdir, '..', 'segmentation'))
    has_segmentation = check_segmentation(seg_dir)
    if not has_segmentation:
        raise ValueError("Problems with segmentation! Run segmentation again.")
    print "Segmentation detected"

    # parameters
    mpp = params['px2um']
    nfovs = meta['sequence_count']
    t_height = meta['tile_height']
    t_width = meta['tile_width']
    fmtdict={}
    fmt="f{{fov:0{:d}d}}".format(int(np.log10(nfovs))+1)
    fmtdict['fov']=fmt
    fmt="y{{y:0{:d}d}}".format(int(np.log10(t_height))+1)
    fmtdict['y']=fmt
    fmt="x{{x:0{:d}d}}".format(int(np.log10(t_width))+1)
    fmtdict['x']=fmt
    cell_id_fmt = fmtdict['fov'] + fmtdict['y'] + fmtdict['x']

    # MAKE CELL COLLECTION
    cells = []
    tiff_dir = os.path.join(outputdir,'tiffs')
    if not os.path.isdir(tiff_dir):
        os.makedirs(tiff_dir)
    print "{:<20s}{:<s}".format("tiff_dir", tiff_dir)
    mask_dir = os.path.join(outputdir,'masks')
    if not os.path.isdir(mask_dir):
        os.makedirs(mask_dir)
    print "{:<20s}{:<s}".format("mask_dir", mask_dir)
    ## tiff list
    indexname='index_tiffs.txt'
    pathtoindex=os.path.join(seg_dir,indexname)
    index = load_index(pathtoindex)
    nfiles = len(index)
    for n in range(nfiles):
        # load tiff
        f, ef, mf, lf = index[n]
        print "Processing file {:d} / {:d}".format(n,nfiles)
        f = os.path.relpath(os.path.join(seg_dir,f))
        print f
        with ti.TiffFile(f) as tif:
            img = tif.asarray()
            meta = tif.imagej_metadata
        if mpp is None:
            try:
                mpp = float(meta['mpp'])
                print "mpp = {:.6f}".format(mpp)
            except KeyError, ValueError:
                print "Unit is lacking: mpp = 1!"
                mpp = 1.
        fov = meta['m']
        nchannels, height, width = img.shape
        print img.dtype
        print img.shape

        # make meshgrid
        Y,X = np.mgrid[0:height,0:width]

        # load labels
        lf = os.path.relpath(os.path.join(seg_dir,lf))
        print lf
        labels = ssp.load_npz(lf).todense()
        nlabels = np.max(labels)
        if nlabels == 0:
            print "No labels detected"
            continue

        # compute background
        mask_bg = (labels == 0)
        val_bg = img[:,mask_bg]
        channel_bg = np.median(val_bg, axis=1)


        # iterate over segmented objects
        for n in range(1, nlabels):
            cell = {}
            print "Getting cell {:d} / {:d} for FOV {:d}".format(n,nlabels,fov)

            # make index
            mask = (labels == n)

            # fov
            cell['fov']=fov
            cell['mpp']=mpp

            # get points
            points = np.transpose([X[mask],Y[mask]])
            P = len(points)
            cell['pixels']={}
            cell['pixels']['xcoord']=points[:,0]
            cell['pixels']['ycoord']=points[:,1]

            # rotated bounding box
            bb = cv2.minAreaRect(points)
            xymid,wh,angle=bb
            w,h=wh
            cell['bounding_box_rotated']={}
            cell['bounding_box_rotated']['xcoord_center']=xymid[0]
            cell['bounding_box_rotated']['ycoord_center']=xymid[1]
            cell['bounding_box_rotated']['width']=w
            cell['bounding_box_rotated']['height']=h
            cell['bounding_box_rotated']['angle']=h

            # dimensions
            if w > h:
                wtp=w
                w=h
                h=wtp
            cell['width']=w
            cell['height']=h
            cell['area']=P
            cell['volume']=np.pi/4.* w**2*h - np.pi/12.*w**3 # cylinder with hemispherical caps of length h and width w
            cell['area_rect']=w*h
            cell['width_um']=cell['width']*mpp
            cell['height_um']=cell['height']*mpp
            cell['area_um2']=cell['area']*mpp*mpp
            cell['area_rect_um2']=cell['area_rect']*mpp*mpp
            cell['volume_um3']=cell['volume']*mpp*mpp*mpp

            # fluorescence
            val = img[:,mask]
            cell['fluorescence']={}
            cell['fluorescence']['background_px']=channel_bg
            cell['fluorescence']['background_cell']=channel_bg*len(points)
            cell['fluorescence']['total']=np.sum(val,axis=1)

            # id
            cell_id = cell_id_fmt.format(fov=fov,y=int(xymid[1]),x=int(xymid[0]))
            cell['id']=cell_id

            # add to list
            cells.append(cell)

            # write tiff
            if not namespace.lean:
                write_crop(img, mask, points, bname=cell_id, tiff_dir=tiff_dir, mask_dir=mask_dir,debug=namespace.debug, **params['crops'])

    ncells = len(cells)
    print "ncells = {:d} collected".format(ncells)

# write down the cell dictionary
    celldict = {cell['id']: cell for cell in cells}
    celldict = make_dict_serializable(celldict)
    pathtocells = os.path.join(outputdir, 'collection.js')
    write_dict2json(pathtocells,celldict)
    print "{:<20s}{:<s}".format('fileout', pathtocells)

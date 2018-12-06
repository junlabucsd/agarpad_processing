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
import cPickle as pkl
import scipy.ndimage as simg

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
    params['preprocess_images']={}
    mydict = params['preprocess_images']
    mydict['invert'] = None
    mydict['bg_subtract'] = True
    mydict['bg_size'] = 400

    return params

def get_background_checkerboard(img, size=65):
    """
    Compute background of an image and return it
    """
    bg=np.zeros(img.shape,dtype=np.float_)
    dtype=img.dtype
    height,width = img.shape

    L = np.min([size,height,width])

    hedges = np.arange(0,height+1,L)
    if hedges[-1] < height:
        hedges[-1] = height
    wedges = np.arange(0,width+1,L)
    if wedges[-1] < width:
        wedges[-1] = width
    nh = len(hedges)-1
    nw = len(wedges)-1
    img_coarse = np.zeros((nh,nw),dtype=np.uint8)
    Ycoarse,Xcoarse = np.mgrid[0:nh,0:nw]
    for j in range(nh):
        h0 = hedges[j]
        h1 = hedges[j+1]
#        print "h0 = {:d}    h1 = {:d}".format(h0,h1)
        for i in range(nw):
            w0 = wedges[i]
            w1 = wedges[i+1]
#            print "w0 = {:d}    w1 = {:d}".format(w0,w1)
            rect = img[h0:h1:,w0:w1]

            val = np.median(rect)
#            print "median = {:d}".format(val)
            bg[h0:h1,w0:w1] = val
            img_coarse[j,i] = val

    #bsize=5
    #kernel = np.ones((bsize,bsize))/(bsize*bsize)
    #itermax = min(max(size/bsize,1),10)
    #bg = simg.filters.convolve(bg,kernel)
    #bg = simg.uniform_filter(bg,size=size)
    #bg = np.array(bg,dtype=dtype)
    bsize = 2*int(size/2)+1
    bg = cv2.blur(bg,ksize=(bsize,bsize))
    # TEST
#    import matplotlib.pyplot as plt
#    fig = plt.figure()
#    ax = fig.add_subplot(131)
#    ax.imshow(img_coarse, cmap='gray')
#    ax.set_xticks([]), ax.set_yticks([])
#    ax = fig.add_subplot(132)
#    img_coarse_blur = cv2.blur(img_coarse,ksize=(3,3))
#    #img_coarse_blur = cv2.GaussianBlur(img_coarse,ksize=(3,3), sigmaX=0, sigmaY=0)
#    ax.imshow(img_coarse_blur, cmap='gray')
#    ax.set_xticks([]), ax.set_yticks([])
#    ax = fig.add_subplot(133)
#    img_mblur = cv2.medianBlur(img, ksize=201)
#    ax.imshow(img_mblur,cmap='gray')
#    ax.set_xticks([]), ax.set_yticks([])
#    plt.savefig("test.png")
#    sys.exit()
    # TEST

    # totally inefficient
#    if not (np.mod(L,2) == 1):
#        raise ValueError("L must be odd")
#    p = int(float(L)/2)
#    for j in range(height):
#        print "j = {:d}".format(j)
#        if (j-p < 0):
#            y0 = 0
#            y1 = min(y0 + 2*p + 1,height-1)
#        elif (j+p > height-1):
#            y1 = height-1
#            y0 = max(y1 - (2*p + 1),0)
#        else:
#            y0 = j-p
#            y1 = j+p
#        subimg = img[y0:y1+1,:]
#        for i in range(width):
#            if (i-p < 0):
#                x0 = 0
#                x1 = min(x0 + 2*p + 1, width-1)
#            elif (i+p > width-1):
#                x1 = width-1
#                x0 = max(x1 - (2*p + 1),0)
#            else:
#                x0 = i-p
#                x1 = i+p
#            rect = subimg[:,x0:x1+1]
#            bg[j,i] = np.median(rect)
#        # end i loop
#    # end j loop
    return bg

def get_background_medianblur(img, size=201):
    """
    Compute background of an image and return it

    !!Something is wrong with this method!!
    """
    shape = img.shape
    if len(shape) != 2:
        raise ValueError("img must be a 2D array")
    dtype = img.dtype
    norm = get_img_norm(dtype)

    if (norm != 255):
        img8_bg = cv2.medianBlur(np.array(255 * (np.float_(img) / norm), dtype=np.uint8), ksize=size)
        bg = np.array(norm*(np.float_(img8_bg) / 255), dtype=dtype)
    else:
        bg = cv2.medianBlur(img, ksize=size)
    bg = cv2.blur(bg,ksize=(size,size)) # maybe not essential, to smooth background values

    return bg

def preprocess_image(tiff_file, outputdir='.', invert=None, bg_subtract=True, bg_size=200, debug=False):
    """
    INPUT:
      * file to a tiff image.
    OUTPUT:
      * file with a preprocessed tiff image.

    The preprocessing consists in:
      * inverting some channels (namely for phase contrast)
      * subtracting the background.

    USEFUL DOCUMENTATION:
      * https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html
      * https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    """

    # method for background subtraction
    #get_background = get_background_medianblur # does not work well
    get_background = get_background_checkerboard

    # pre-processing
    bname = os.path.splitext(os.path.basename(tiff_file))[0]

    # read the input tiff_file
    img, meta = get_tiff2ndarray(tiff_file, channel=None,metadata=True,normalize=False) # read all channels

    ## adjust format
    shape = img.shape
    if (len(shape) == 2):
        img = np.array([img], dtype=img.dtype)  # make an axis for channel
    elif (len(shape) == 3):
        pass
    else:
        raise ValueError("Handling for this format of image is not implemented yet.")
    nchannel, height, width = img.shape
    dtype = img.dtype

    # get norm
    norm = get_img_norm(dtype)

    # inversions
    if invert == None:
        invert = []
    elif invert == 'all':
        invert = range(nchannels)

    # adjust background sliding window size
    bg_size = 2*int(bg_size/2) + 1 # make odd
    print "bg_size = {:d}".format(bg_size)

    img0 = np.copy(img)
    # process channels
    img_bg = np.zeros(shape, dtype=dtype)
    img_subtracted = np.copy(img0)
    blur = np.zeros((nchannel, height, width), dtype=dtype)
    for c in range(nchannel):
        # select source
        arr = img[c]

        # invert if needed
        if c in invert:
            arr = np.array(norm - arr, dtype=dtype)
            img0[c]=np.copy(arr)

        # background subtraction
        if bg_subtract:
#            print "Background subtraction..."
            # method for background subtraction using sliding window
            img_bg[c] = get_background(arr, size=bg_size)
            idx = arr > img_bg[c]
            arr[idx] = arr[idx] - img_bg[c][idx]
            arr[~idx] = 0
            img_subtracted[c] = np.copy(arr)
        else:
            img_subtracted[c] = np.zeros((height,width), dtype=dtype)

        # preprocess the imaging for connected components finding
        ## gaussian blur
        scale=5
        arr = cv2.GaussianBlur(arr,(scale,scale),sigmaX=0,sigmaY=0)
        blur[c] = np.copy(arr)

        # save
        img[c]=arr

    # write tiff
    dirname = os.path.dirname(tiff_file)
    if dirname == outputdir:
        raise ValueError("Output dir must be different from TIFF dir: {}".format(dirname))
    fname = os.path.basename(tiff_file)
    fileout = os.path.join(outputdir,fname)
    ti.imwrite(fileout, img, imagej=True, photometric='minisblack', metadata=meta)
    print "{:<20s}{:<s}".format('fileout',fileout)

    # debug
    if debug:
        print "Debug for preprocess images"
        img_bg_post = np.zeros(shape, dtype=dtype)
        for c in range(nchannel):
#            print "c = {:d}".format(c)
            img_bg_post[c] = get_background(img[c], size=bg_size)
#            print np.unique(img[c])
#            print np.unique(img_bg_post[c])
        debugdir = os.path.join(outputdir,'debug')
        if not os.path.isdir(debugdir):
            os.makedirs(debugdir)
        # plots
        import matplotlib.pyplot as plt
        import matplotlib.patches
        from matplotlib.path import Path
        import matplotlib.collections
        from matplotlib.gridspec import GridSpec
        ncolors=(20-1)
        images = [img0, img_bg, img_subtracted, blur, img_bg_post]
        titles = ['original','background (size = {:d})'.format(bg_size), 'Bg. subtr.','gaussian blur (scale = {:d})'.format(scale), 'background (post)']
        cmaps=['gray','gray','gray','gray','gray']
        nimg=len(images)
        nrow = nchannel
        ncol = nimg
        ratio = float(height)/float(width)
        fig = plt.figure(num=None,figsize=(ncol*4,nrow*4*ratio))
        gs = GridSpec(nrow,ncol,figure=fig)
        axes=[]
        for r in range(nrow):
            for c in range(ncol):
                image = images[c][r]
                ax=fig.add_subplot(gs[r,c])
                axes.append(ax)
                cf=ax.imshow(image, cmap=cmaps[c])
                ax.set_xticks([]), ax.set_yticks([])
                if (r == 0):
                    ax.set_title(titles[c].upper())

        fname = "{}".format(bname)
        fileout = os.path.join(debugdir,fname + '.png')
        gs.tight_layout(fig,w_pad=0)
        plt.savefig(fileout,dpi=300)
        print "{:<20s}{:<s}".format('debug file', fileout)
        plt.close('all')

    # exit
    return fileout

#################### main ####################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Preprocessing tool.")
    parser.add_argument('-i', '--images',  type=str, nargs='+', required=False, help='tiff files to open.')
    parser.add_argument('-f', '--paramfile',  type=file, required=False, help='Yaml file containing parameters.')
    parser.add_argument('-d', '--outputdir',  type=str, required=False, help='Output directory')
    parser.add_argument('--debug',  action='store_true', required=False, help='Enable debug mode')

    # INITIALIZATION
    # load arguments
    namespace = parser.parse_args(sys.argv[1:])

    # output directory
    rootdir = os.getcwd()
    outputdir = namespace.outputdir
    if (outputdir is None):
        outputdir = os.getcwd()
    else:
        outputdir = os.path.relpath(outputdir, os.getcwd())
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
    print "{:<20s}{:<s}".format("outputdir", outputdir)

    # parameter file
    if namespace.paramfile is None:
        allparams = default_parameters()
        paramfile = "preprocess_images.yml"
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
    if namespace.images is None:
        namespace.images=[]
    for f in namespace.images:
        test = check_tiff_file(f)
        if test:
            tiff_files.append(f)
    ntiffs = len(tiff_files)
    if ntiffs == 0:
        raise ValueError("No tiff detected!")

    # copy metadata
    tiffdir=os.path.dirname(os.path.relpath(tiff_files[0], rootdir))
    metadata=os.path.join(tiffdir,'metadata.txt')
    dest = os.path.join(outputdir,os.path.basename(metadata))
    if (os.path.realpath(metadata) != os.path.realpath(dest)):
        shutil.copy(metadata,dest)

    # debug or not
    if namespace.debug:
        print "!! Debug mode !!"

    params=allparams['preprocess_images']

    for f in tiff_files:
        pf=preprocess_image(f, outputdir=outputdir, debug=namespace.debug, **params)
        pf = os.path.relpath(pf,rootdir)


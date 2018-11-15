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
import cv2

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
    Open a tiff_file and return a numpy array normalized between 0 and 1.
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
    arr = np.array(arr, dtype=np.float_) / norm

    return arr

def get_estimator_contours(img, w0=1, w1=100, h0=10, h1=1000, acut=0.9):
    """
    INPUT:
      * 2D matrix (image).
      * (w0,w1): minimum and maximum width for contours in pixels.
      * (l0,l1): minimum and maximum length for contours in pixels.
      * acut: minimum area/rectangle bounding box ratio.
    OUTPUT:
      * 2D matrix of weights corresponding to the probability that a pixel belongs to a cell.

    USEFUL DOCUMENTATION:
      * https://docs.opencv.org/3.4.3/dd/d49/tutorial_py_contour_features.html
      * https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
    """

    #""" from agarpad code: start
    # find all the cells using contouring
    kernel = np.ones((3,3), np.uint8)
#    img_preprocess = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # preprocess the imaging for connected components finding
    ## gaussian blur
    scale=5
    blur = cv2.GaussianBlur(img,(scale,scale),sigmaX=0,sigmaY=0)
    #th3 = cv2.adaptiveThreshold(blur, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,121,2)
    #th3 = cv2.adaptiveThreshold(blur, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,211,2)
    #ret, th3 = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY)
    #ret2,global_otsu_inv = cv2.threshold(th3, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    ## OTSU thresholding to binary mask
    ret2,global_otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    ## opening/closing operations
    opening = cv2.morphologyEx(global_otsu_inv, cv2.MORPH_OPEN, kernel)
    closing = cv2.erode(opening, kernel, iterations = 1)
    img_inv = cv2.dilate(closing, kernel, iterations = 1)

    ## find connected components
    ncomp, labels = cv2.connectedComponents(img_inv)
    print "Found {:d} objects".format(ncomp)

    ## compute the bounding box for the identified labels
    #for n in range(ncomp):
    height,width = img_inv.shape
    Y,X = np.mgrid[0:height,0:width]
    np.random.seed(123)
    boundingboxes=[]
    pointsperbox=[]
    upright=False
    #for n in np.random.permutation(np.arange(ncomp))[:10]:
    for n in np.arange(ncomp):
        idx = (labels == n)
        points = np.transpose([X[idx],Y[idx]])
        pointsperbox.append(len(points))
        if upright:
            # upright rectangles
            bb = cv2.boundingRect(points)
            boundingboxes.append(bb)
            x,y,w,h=bb
            #cv2.rectangle(img_inv, (x,y), (x+w,y+h),(255,0,0),2) # to draw the rectangles
        else:
            # rotated rectangles
            bb = cv2.minAreaRect(points)
            boundingboxes.append(bb)

    # estimator matrix
    ## compute scores
    scores = []
    for n in np.arange(ncomp):
        bb = boundingboxes[n]
        area = pointsperbox[n]
        if upright:
            xlo,ylo,w,h = bb
        else:
            xymid,wh,angle=bb
            w,h=wh
        if w > h:
            wtp=w
            w=h
            h=wtp
        area_rect = w*h
        #print "w={:.1f}    h={:.1f}".format(w,h)
        aval = area/area_rect
        score = 1.
        score *= min(1,np.exp((w-w0)))                  # penalize w < w0
        score *= min(1,np.exp(w1-w))               # penalize w > w1
        score *= min(1,np.exp((h-h0)))                  # penalize w < w0
        score *= min(1,np.exp(h1-h))               # penalize w > w1
        score *= min(1,np.exp(aval-acut))   # penalize area/rect < acut
        scores.append(score)

    # estimator matrix
    eimg = np.zeros(img.shape, dtype=np.float_)
    for n in np.arange(ncomp):
        idx = (labels == n)
        eimg[idx] = scores[n]

    # plots
    import matplotlib.pyplot as plt
    import matplotlib.patches
    from matplotlib.path import Path
    import matplotlib.collections
    from matplotlib.gridspec import GridSpec
    ncolors=(20-1)
    labels_iterated = np.uint8(labels - np.int_(labels / ncolors) * ncolors) + 1
    images = [img, blur, global_otsu_inv, img_inv, labels_iterated,eimg]
    titles = ['original','gaussian blur','otsu','closing/opening','labels','estimator']
    cmaps=['gray','gray','gray','gray','tab20c','viridis']
    nfig=len(images)
    nrow = int(np.sqrt(nfig))
    ncol = nfig/nrow
    if (ncol*nrow < nfig): ncol+=1
    fig = plt.figure(num=None,figsize=(ncol*4,nrow*3))
    gs = GridSpec(nrow,ncol,figure=fig)
    axes=[]
    for r in range(nrow):
        for c in range(ncol):
            ind = r*ncol+c
            print ind
            if not (ind < nfig):
                break
            ax=fig.add_subplot(gs[r,c])
            axes.append(ax)
            ax.set_title(titles[ind].upper())
            cf=ax.imshow(images[ind], cmap=cmaps[ind])
            ax.set_xticks([]), ax.set_yticks([])

            if titles[ind] == 'labels':
                # draw bounding boxes
                rects = []
                codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
                for bb in boundingboxes:
                    if upright:
                        xlo,ylo,w,h = bb
                        rect = matplotlib.patches.Rectangle((xlo,ylo), width=w, height=h, fill=False)
                    else:
                        verts=cv2.boxPoints(bb)
                        verts=np.concatenate((verts,[verts[0]]))
                        path = Path(verts,codes)
                        rect = matplotlib.patches.PathPatch(path)
                    rects.append(rect)
                col = matplotlib.collections.PatchCollection(rects, edgecolors='k', facecolors='none', linewidths=0.5)
                ax.add_collection(col)

            if titles[ind] == 'estimator':
                fig.colorbar(cf,ax=ax)

    fileout = os.path.join(os.getcwd(),'test_get_estimator_agarpadoriginal.png')
    gs.tight_layout(fig,w_pad=0)
    plt.savefig(fileout,dpi=300)
    print "writing: ", fileout
    plt.close('all')
    # from agarpad code: end """

    """ canny: start
    #test canny filtering
    print np.min(img), np.max(img)
    maxthres=2500
    minthres=100
    edges = cv2.Canny(img,threshold1=minthres,threshold2=maxthres,apertureSize=5)
    print np.unique(edges)
    print "START TEST"
    import matplotlib.pyplot as plt
    fig = plt.figure(num=None,figsize=(8,4))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.xticks([]),plt.yticks([])
    fileout = os.path.join(os.getcwd(),'test_get_estimator_contours.png')
    fig.tight_layout()
    plt.savefig(fileout,dpi=300)
    print "writing: ", fileout
    plt.close('all')

    fig = plt.figure(num=None,figsize=(4,4))
    hist,hedges = np.histogram(np.ravel(img), bins='auto')
    plt.bar(hedges[:-1], hist, np.diff(hedges), facecolor='blue', lw=0)
    fileout = os.path.join(os.getcwd(),'test_get_estimator_contours_histogram.png')
    fig.tight_layout()
    plt.savefig(fileout,dpi=300)
    print "writing: ", fileout
    plt.close('all')
    sys.exit()
    print "END TEST"
    #test
    # canny: start """


    return

def get_estimator(tiff_file, method='contours', outputdir='.', channel=0, method_params=dict(w0=1, w1=100, l0=10, l1=1000, acut=0.9),crop=dict(xlo=None, xhi=None, ylo=None, yhi=None)):
    """
    Compute the estimator for a given images. The estimator is a real matrix where each entry is the estimation that the corresponding pixel belongs to the segmented class.
    INPUT:
        * path to a tiff image.
    OUTPUT:
        * path to a matrix (written in a file) of same size of the orignal image.
    """
    # read the input tiff_file
    img = get_tiff2ndarray(tiff_file, channel=channel)
    xlo=crop['xlo']
    xhi=crop['xhi']
    ylo=crop['ylo']
    yhi=crop['yhi']
    img = img[ylo:yhi,xlo:xhi]

    # test
#    import matplotlib.pyplot as plt
#    plt.imshow(img)
#    fname = os.path.splitext(os.path.basename(tiff_file))[0]
#    fileout = os.path.join(outputdir,fname+'.png')
#    print fileout
#    plt.savefig(fileout)
    # test

    # perform the segmentation
    if method == 'contours':
        # pass a 8-bit image
        img = (img - np.min(img))/(np.max(img)-np.min(img))
        estimator = get_estimator_contours(np.array(255*img,np.uint8), **method_params)
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
    pathtoindex = os.path.join(outputdir,"index_tiffs.txt")

    ## make tiff index
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
        print f
        ef=get_estimator(f, method=segmentation_method, outputdir=estimator_dir, method_params=params['method_params'][segmentation_method], crop=params['crop'])
        est_files.append(os.path.relpath(ef,outputdir))
    paths = np.asarray([os.path.relpath(ef,outputdir) for ef in est_files])
    with open(pathtoindex,'r') as fin:
        index = np.loadtxt(pathtoindex, dtype=np.string_, ndmin=1)
    if len(index.shape) == 1:
        index = np.array([index])
    est_files = np.array(est_files, dtype=np.string_)
    index = np.concatenate([index,[est_files]])
    with open(pathtoindex,'w') as fout:
        np.savetxt(fout,np.transpose(index),fmt='%s',delimiter=',')
    print "{:<20s}{:<s}".format("fileout", pathtoindex)



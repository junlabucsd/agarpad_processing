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
    params['segmentation']={}
    mydict = params['segmentation']
    mydict['method'] = 'bounding_box'
    mydict['estimator_params'] = {\
            'bounding_box': dict(w0=1, w1=100, l0=10, l1=1000, acut=0.9)\
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

def get_estimator_boundingbox(tiff_file, channel=0, outputdir='.', w0=1, w1=100, h0=10, h1=1000, acut=0.9, emin=1.0e-4, phase_contrast=False, debug=False):
    """
    INPUT:
      * file to a tiff image.
      * (w0,w1): minimum and maximum width for bounding box in pixels.
      * (l0,l1): minimum and maximum length for bounding box in pixels.
      * acut: minimum area/rectangle bounding box ratio.
    OUTPUT:
      * 2D matrix of weights corresponding to the probability that a pixel belongs to a cell.

    USEFUL DOCUMENTATION:
      * https://docs.opencv.org/3.4.3/dd/d49/tutorial_py_contour_features.html
      * https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
    """

    # pre-processing
    bname = os.path.splitext(os.path.basename(tiff_file))[0]
    ## read the input tiff_file
    img = get_tiff2ndarray(tiff_file, channel=channel)
    ## rescale dynamic range linearly
    img = (img - np.min(img))/(np.max(img)-np.min(img))
    # pass a 8-bit image
    img = np.array(255*img,np.uint8)

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
    if (phase_contrast):
        ret2,global_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        ret2,global_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    ## opening/closing operations
    opening = cv2.morphologyEx(global_otsu, cv2.MORPH_OPEN, kernel)
    closing = cv2.erode(opening, kernel, iterations = 1)
    img_inv = cv2.dilate(closing, kernel, iterations = 1)

    ## find connected components
    ncomp, labels = cv2.connectedComponents(img_inv)
    print "Found {:d} objects".format(ncomp)

    ## compute the bounding box for the identified labels
    #for n in range(ncomp):
    height,width = img_inv.shape
    Y,X = np.mgrid[0:height,0:width]
    boundingboxes=[]
    pointsperbox=[]
    upright=False
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

        # discard small values
        if score < emin:
            score = 0.
        scores.append(score)

    # estimator matrix
    eimg = np.zeros(img.shape, dtype=np.float_)
    for n in np.arange(ncomp):
        idx = (labels == n)
        eimg[idx] = scores[n]
    nz = len(eimg > 0.)
    print "nz = {:d}    sparcity index = {:.2e}".format(nz, float(nz)/len(np.ravel(eimg)))
    efname = bname
    #efile = os.path.join(outputdir,efname+'.txt')
    #efile = os.path.join(outputdir,efname+'.pkl')
    efile = os.path.join(outputdir,efname+'.npz')
    with open(efile,'w') as fout:
        #np.savetxt(fout, eimg)
        #pkl.dump(eimg,fout)
        ssp.save_npz(efile, ssp.coo_matrix(eimg), compressed=False)
    print "{:<20s}{:<s}".format('est. file', efile)

    if debug:
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
        labels_iterated = np.uint8(labels - np.int_(labels / ncolors) * ncolors) + 1
        images = [img, blur, global_otsu, img_inv, labels_iterated,eimg]
        titles = ['original','gaussian blur','otsu','closing/opening','bounding box','estimator']
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
                if not (ind < nfig):
                    break
                ax=fig.add_subplot(gs[r,c])
                axes.append(ax)
                ax.set_title(titles[ind].upper())
                cf=ax.imshow(images[ind], cmap=cmaps[ind])
                ax.set_xticks([]), ax.set_yticks([])

                if titles[ind] == 'bounding box':
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

        fname = "{}_estimator_debug".format(bname)
        fileout = os.path.join(debugdir,fname + '.png')
        gs.tight_layout(fig,w_pad=0)
        plt.savefig(fileout,dpi=300)
        print "{:<20s}{:<s}".format('debug file', fileout)
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


    return os.path.realpath(efile)

def get_estimator(tiff_file, method='bounding_box', outputdir='.', channel=0, estimator_params=dict(w0=1, w1=100, l0=10, l1=1000, acut=0.9), emin=1.0e-4, debug=False):
    """
    Compute the estimator for a given images. The estimator is a real matrix where each entry is the estimation that the corresponding pixel belongs to the segmented class.
    INPUT:
        * path to a tiff image.
    OUTPUT:
        * path to a matrix (written in a file) of same size of the orignal image.
    """
    # perform the segmentation
    if method == 'bounding_box':
        efile = get_estimator_bounding_box(tiff_file, channel=channel, outputdir=outputdir, debug=debug, emin=emin, **estimator_params)
    else:
        raise ValueError("Segmentation method not implemented.")

    return efile

def get_mask(f, ef, threshold=0., outputdir='.', debug=False):
    """
    Make a binary mask by applying a threshold to the input estimator file.
    INPUT:
        * ef: estimator file as a sparse matrix
    OUTPUT:
        * mf: binary mask
    """
    bname = os.path.splitext(os.path.basename(f))[0]
    # read and convert input to dense matrix
    with open(ef, 'r') as fin:
        emat = ssp.load_npz(fin).todense()

    # apply threshold
    mask = np.array(emat > threshold, dtype=np.uint8)

    # write the mask
    mfname = bname
    #mfile = os.path.join(outputdir,mfname+'.txt')
    #mfile = os.path.join(outputdir,mfname+'.pkl')
    mfile = os.path.join(outputdir,mfname+'.npz')
    with open(mfile,'w') as fout:
        #np.savetxt(fout, mask)
        #pkl.dump(mask,fout)
        ssp.save_npz(mfile, ssp.coo_matrix(mask), compressed=False)
    print "{:<20s}{:<s}".format('mask file', mfile)

    # debug
    if debug:
        debugdir = os.path.join(outputdir,'debug')
        if not os.path.isdir(debugdir):
            os.makedirs(debugdir)
        # plots
        import matplotlib.pyplot as plt
        height, width = mask.shape
        ratio = float(height/width)
        fig = plt.figure(num=None,figsize=(4,4*ratio))
        ax=fig.gca()
        ax.imshow(mask,cmap='gray')
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title('MASK')
        fname = "{}_mask_debug".format(bname)
        fileout = os.path.join(debugdir,fname + '.png')
        fig.tight_layout()
        plt.savefig(fileout,dpi=300, bbox_inches='tight', pad_inches=0)
        print "{:<20s}{:<s}".format('debug file', fileout)
        plt.close('all')

    return os.path.realpath(mfile)

def get_label(f, mf, outputdir='.', debug=False):
    """
    Make a matrix containing labels for each connected component.
    INPUT:
        * binary mask
    OUTPUT:
        * label matrix
    """
    bname = os.path.splitext(os.path.basename(f))[0]
    # read and convert input to dense matrix
    with open(mf, 'r') as fin:
        mmat = ssp.load_npz(fin).todense()

    # find connected components
    n, labels = cv2.connectedComponents(mmat)

    # write the labels
    lfname = bname
    #lfile = os.path.join(outputdir,lfname+'.txt')
    #lfile = os.path.join(outputdir,lfname+'.pkl')
    lfile = os.path.join(outputdir,lfname+'.npz')
    with open(lfile,'w') as fout:
        #np.savetxt(fout, labels)
        #pkl.dump(labels,fout)
        ssp.save_npz(lfile, ssp.coo_matrix(labels), compressed=False)
    print "{:<20s}{:<s}".format('labels file', lfile)

    # debug
    if debug:
        debugdir = os.path.join(outputdir,'debug')
        if not os.path.isdir(debugdir):
            os.makedirs(debugdir)
        ncolors=20-1
        labels_mod = np.uint8(labels - np.int_(labels / ncolors) * ncolors) + 1 # between 1 and 19
        # plots
        import matplotlib.pyplot as plt
        height, width = labels.shape
        ratio = float(height/width)
        fig = plt.figure(num=None,figsize=(4,4*ratio))
        ax=fig.gca()
        ax.imshow(labels_mod,cmap='tab20c')
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title('LABELS')
        fname = "{}_labels_debug".format(bname)
        fileout = os.path.join(debugdir,fname + '.png')
        fig.tight_layout()
        plt.savefig(fileout,dpi=300, bbox_inches='tight', pad_inches=0)
        print "{:<20s}{:<s}".format('debug file', fileout)
        plt.close('all')


    return os.path.realpath(lfile)

#################### main ####################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Segmentation tool -- cells.")
    parser.add_argument('-i', '--images',  type=str, nargs='+', required=False, help='tiff files to open.')
    parser.add_argument('-f', '--paramfile',  type=file, required=False, help='Yaml file containing parameters.')
    parser.add_argument('-d', '--outputdir',  type=str, required=False, help='Output directory')
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
    rootdir=outputdir
    outputdir = os.path.join(outputdir,'cells','segmentation')
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
    if namespace.images is None:
        namespace.images=[]
    for f in namespace.images:
        test = check_tiff_file(f)
        if test:
            tiff_files.append(f)
    ntiffs = len(tiff_files)

    # debug or not
    if namespace.debug:
        print "!! Debug mode !!"

    params=allparams['segmentation']
    segmentation_method=params['method']

    # BUILD INDEX IF NECESSARY
    pathtoindex = os.path.join(outputdir,"index_tiffs.txt")
    if ntiffs == 0:
        index = load_index(pathtoindex)
        index = np.array(index, dtype=np.string_)
        tiff_files = [os.path.join(outputdir,f) for f in index[:,0]]
    else:
        if os.path.isfile(pathtoindex):
            os.remove(pathtoindex)

    ntiffs = len(tiff_files)

    if ntiffs == 0:
        raise ValueError("Number of tiff files is zero!")
    else:
        print "{:<20s}{:<d}".format("ntiffs", ntiffs)

    if not os.path.isfile(pathtoindex):
        index = np.asarray([[os.path.relpath(f,outputdir)] for f in tiff_files])
        write_index(pathtoindex,index)
        print "{:<20s}{:<s}".format("fileout", pathtoindex)

    # move metadata if found
    index = load_index(pathtoindex)
    index = np.array(index,dtype=np.string_)
    tiffdir=os.path.dirname(os.path.join(outputdir,index[0][0]))
    metadata=os.path.join(tiffdir,'metadata.txt')
    dest = os.path.join(rootdir,os.path.basename(metadata))
    if (os.path.realpath(metadata) != os.path.realpath(dest)):
        shutil.copy(metadata,dest)

    # BUILD ESTIMATOR MATRICES
    build_estimator=False
    index = load_index(pathtoindex)
    index = np.array(index,dtype=np.string_)
    if (index.shape[1] < 2):
        build_estimator=True
    if build_estimator:
        estimator_dir=os.path.join(outputdir,'estimators')
        if not os.path.isdir(estimator_dir):
            os.makedirs(estimator_dir)
        print "{:<20s}{:<s}".format("est. dir.", estimator_dir)
        est_files = []
        for f in tiff_files:
            ef=get_estimator(f, method=segmentation_method, outputdir=estimator_dir, estimator_params=params['estimator_params'][segmentation_method], channel=params['channel'], debug=namespace.debug)
            est_files.append(os.path.relpath(ef,outputdir))
        est_files = np.array(est_files, dtype=np.string_)
        index = np.concatenate([index,np.transpose([est_files])], axis=1)

        write_index(pathtoindex,index)
        print "{:<20s}{:<s}".format("fileout", pathtoindex)

    # BUILD MASKS
    build_mask=False
    index = load_index(pathtoindex)
    index = np.array(index, dtype=np.string_)
    if (index.shape[1] < 3):
        build_mask=True
    if build_mask:
        mask_dir=os.path.join(outputdir,'masks')
        if not os.path.isdir(mask_dir):
            os.makedirs(mask_dir)
        print "{:<20s}{:<s}".format("mask dir.", mask_dir)
        mask_files = []
        nfile = len(index)
        for n in range(nfile):
            f, ef = index[n]
            f = os.path.join(outputdir,f)
            ef = os.path.join(outputdir,ef)
            mf=get_mask(f, ef, outputdir=mask_dir, debug=namespace.debug, **params['mask_params'])
            mask_files.append(os.path.relpath(mf,outputdir))

        mask_files = np.array(mask_files, dtype=np.string_)
        index = np.concatenate([index,np.transpose([mask_files])], axis=1)

        write_index(pathtoindex,index)
        print "{:<20s}{:<s}".format("fileout", pathtoindex)

    # BUILD LABELS
    build_label=False
    index = load_index(pathtoindex)
    index = np.array(index, dtype=np.string_)
    if (index.shape[1] < 4):
        build_label=True
    if build_label:
        label_dir=os.path.join(outputdir,'labels')
        if not os.path.isdir(label_dir):
            os.makedirs(label_dir)
        print "{:<20s}{:<s}".format("label dir.", label_dir)
        label_files = []
        nfile = len(index)
        for n in range(nfile):
            f, ef, mf = index[n]
            f = os.path.join(outputdir,f)
            ef = os.path.join(outputdir,ef)
            mf = os.path.join(outputdir,mf)
            lf=get_label(f, mf, outputdir=label_dir, debug=namespace.debug)
            label_files.append(os.path.relpath(lf,outputdir))

        label_files = np.array(label_files, dtype=np.string_)
        index = np.concatenate([index,np.transpose([label_files])], axis=1)

        write_index(pathtoindex,index)
        print "{:<20s}{:<s}".format("fileout", pathtoindex)

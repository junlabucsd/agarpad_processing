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
            'bounding_box': dict(w0=1, w1=100, h0=10, h1=1000, acut=0.9, aratio_min=2., aratio_max=100., border_pad=5, emin=1.0e-4, threshold=None)\
            }
    mydict['channel'] = 0
    mydict['mask_params']={'threshold': 0.95}

    return params

def get_background_img(img, size=65):
    """
    Compute background of an image and return it
    """
    bg=np.zeros(img.shape,dtype=img.dtype)
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

            val = np.uint8(np.median(rect))
#            print "median = {:d}".format(val)
            bg[h0:h1,w0:w1] = val
            img_coarse[j,i] = val

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

def get_estimator_boundingbox(tiff_file, channel=0, outputdir='.', w0=1, w1=100, h0=10, h1=1000, acut=0.9, aratio_min=2., aratio_max=100., border_pad=5, emin=1.0e-4, debug=False, threshold=None):
    """
    INPUT:
      * file to a tiff image.
      * (w0,w1): minimum and maximum width for bounding box in pixels.
      * (l0,l1): minimum and maximum length for bounding box in pixels.
      * acut: minimum area/rectangle bounding box ratio.
      * threshold is a lower threshold (everything below is set to zero). Value must be a float between 0 and 1. 1 is the maximum, eg. 255 or 65535.
    OUTPUT:
      * 2D matrix of weights corresponding to the probability that a pixel belongs to a cell.

    USEFUL DOCUMENTATION:
      * https://docs.opencv.org/3.4.3/dd/d49/tutorial_py_contour_features.html
      * https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=blur#blur
      * https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
      * https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
      * https://docs.opencv.org/3.4/db/d5c/tutorial_py_bg_subtraction.html
    """

    bname = os.path.splitext(os.path.basename(tiff_file))[0]

    ## read the input tiff_file
    img = get_tiff2ndarray(tiff_file, channel=channel)
    #img0 = np.copy(img)

    # rescale dynamic range linearly (important for OTSU)
    amin = np.min(img)
    amax = np.max(img)
    print "amin = {:.1g}    amax = {:.1g}".format(amin,amax)
    img = (img - amin)/(amax-amin)

    ## thresholding to binary mask
    if threshold is None:
        print "OTSU thresholding"
        # convert to 8-bit image (Open CV requirement for OTSU)
        img = np.array(255*img,np.uint8)
        img8 = np.copy(img)
        # OTSU threshold
        ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        th = max((threshold - amin)/(amax-amin),0) # value in rescaled DNR
        th = np.uint8(255*th) # uint8
        print "threshold = {:.1g}    th = {:d}".format(threshold,th)
        # convert to 8-bit image (Open CV requirement for OTSU)
        img = np.array(255*img,np.uint8)
        img8 = np.copy(img)
        # input threshold
        ret,img = cv2.threshold(img,th,255,cv2.THRESH_BINARY)
    thres = np.int_(ret)
    print "thres = {:d}".format(thres)
    img_bin = np.copy(img)

    ## opening/closing operations
    kernel = np.ones((3,3), np.uint8) # smoothing kernel
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #img_opening = np.copy(img)
    img = cv2.erode(img, kernel, iterations = 1)
    #img_closing = np.copy(img)
    img = cv2.dilate(img, kernel, iterations = 1)
    img_morph = np.copy(img)

    ## find connected components
    ncomp, labels = cv2.connectedComponents(img)
    print "Found {:d} objects".format(ncomp)

    ## compute the bounding box for the identified labels
    #for n in range(ncomp):
    height,width = img.shape
    Y,X = np.mgrid[0:height,0:width]
    boundingboxes=[]
    boundingboxes_upright=[]
    pointsperbox=[]
    for n in np.arange(ncomp):
        idx = (labels == n)
        # pixels coordinates for the label
        points = np.transpose([X[idx],Y[idx]])
        pointsperbox.append(len(points))

        # upright rectangles
        bb = cv2.boundingRect(points)
        boundingboxes_upright.append(bb)

        # rotated rectangles
        bb = cv2.minAreaRect(points)
        boundingboxes.append(bb)

    # estimator matrix
    ## compute scores
    scores = []
    for n in np.arange(ncomp):
        bb = boundingboxes[n]
        bb_upright = boundingboxes_upright[n]
        area = pointsperbox[n]

        # get bounding box width and height
        xymid,wh,angle=bb
        w,h=wh
        if w > h:
            wtp=w
            w=h
            h=wtp

        area_rect = w*h
        aval = area/area_rect
        aratio = h/w
#        print "w={:.1f}    h={:.1f}  aval={:.2e}  aratio={:.1f}".format(w,h,aval,aratio)

        # computing score
        score = 1.
        score *= min(1,np.exp((w-w0)))              # penalize w < w0
        score *= min(1,np.exp(w1-w))                # penalize w > w1
        score *= min(1,np.exp((h-h0)))              # penalize w < w0
        score *= min(1,np.exp(h1-h))                # penalize w > w1
        score *= min(1,np.exp(aval-acut))           # penalize area/rect < acut
        score *= min(1,np.exp(aratio-aratio_min))   # penalize aratio < aratio_min
        score *= min(1,np.exp(aratio_max-aratio))   # penalize aratio > aratio_max

        # check that box corners of an upright bounding box are within the image plus some pad
        x,y,ww,hh = bb_upright
        x0 = x-border_pad
        y0 = y-border_pad
        x1 = x + ww + border_pad
        y1 = y + hh + border_pad
#        print "a0 = {:d}  b0 = {:d}".format(x,y)
#        print "a1 = {:d}  b1 = {:d}".format(x+ww,y+hh)
#        print "x0 = {:d}  y0 = {:d}".format(x0,y0)
#        print "x1 = {:d}  y1 = {:d}".format(x1,y1)
        if not (x0 >= 0 and x1 < width and y0 >=0 and y1 < height):
            score = 0.

        # discard small values
        if score < emin:
            score = 0.
        scores.append(score)

    # estimator matrix
    eimg = np.zeros(img.shape, dtype=np.float_)
    for n in np.arange(ncomp):
        idx = (labels == n)
        eimg[idx] = scores[n]
    nz = np.sum(eimg > 0.)
    ntot = len(np.ravel(eimg))
    print "nz = {:d} / {:d}    sparcity index = {:.2e}".format(nz, ntot, float(nz)/float(ntot))
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

        ## rescale dynamic range linearly
        img_base = np.array(img8, dtype=np.float_)/255.
        img_base = (img_base - np.min(img_base))/(np.max(img_base)-np.min(img_base))
        img_base = np.array(img_base*255, dtype=np.uint8)
        ncolors=(20-1)
        labels_iterated = np.uint8(labels - np.int_(labels / ncolors) * ncolors) + 1
        images = [img_base, img_bin, img_morph, labels_iterated, eimg]
        titles = ['original','binary (thres = {:d})'.format(thres),'closing/opening','bounding box','estimator']
        cmaps=['gray','gray','gray','tab20c','viridis']
        nfig=len(images)
        nrow = int(np.ceil(np.sqrt(nfig)))
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
                    rects_upright = []
                    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
                    for n in range(ncomp):
                        bb_upright = boundingboxes_upright[n]
                        bb = boundingboxes[n]
                        # upright rect
                        xlo,ylo,w,h = bb_upright
                        rect = matplotlib.patches.Rectangle((xlo,ylo), width=w, height=h, fill=False)
                        rects_upright.append(rect)

                        # rect
                        verts=cv2.boxPoints(bb)
                        verts=np.concatenate((verts,[verts[0]]))
                        path = Path(verts,codes)
                        rect = matplotlib.patches.PathPatch(path)
                        rects.append(rect)

                    col = matplotlib.collections.PatchCollection(rects_upright, edgecolors='k', facecolors='none', linewidths=0.5)
                    ax.add_collection(col)
                    col = matplotlib.collections.PatchCollection(rects, edgecolors='r', facecolors='none', linewidths=0.5)
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

def get_estimator(tiff_file, method='bounding_box', outputdir='.', channel=0, estimator_params=dict(w0=1, w1=100, l0=10, l1=1000, acut=0.9, aratio_min=2., aratio_max=100.), emin=1.0e-4, debug=False):
    """
    Compute the estimator for a given images. The estimator is a real matrix where each entry is the estimation that the corresponding pixel belongs to the segmented class.
    INPUT:
        * path to a tiff image.
    OUTPUT:
        * path to a matrix (written in a file) of same size of the orignal image.
    """
    # perform the segmentation
    if method == 'bounding_box':
        efile = get_estimator_boundingbox(tiff_file, channel=channel, outputdir=outputdir, debug=debug, emin=emin, **estimator_params)
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
        ratio = float(height)/width
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
        ratio = float(height)/width
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

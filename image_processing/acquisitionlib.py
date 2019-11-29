#################### imports ####################
import sys
import os
import numpy as np
#from nd2reader import ND2Reader # does not work properly
from pims_nd2 import ND2_Reader as ND2Reader
#from pims import FramesSequenceND
import tifffile as ti


#class ImageReaderND(FramesSequenceND):
#   @property
#   def pixel_type(self):
#       return 'uint16'
#
#   def __init__(self, **axes):
#       super(ImageReaderND, self).__init__()  # properly initialize
#       for name in axes:
#           self._init_axis(name, axes[name])
#       self._register_get_frame(self.get_frame, 'yx')
#       self.bundle_axes = 'yx'  # set default value
#       if 't' in axes:
#           self.iter_axes = 't'  # set default value
#           self.default_coords['t']=0
#       elif 'm' in axes:
#           self.iter_axes = 'm'  # set default value
#           self.default_coords['m']=0
#       elif 'c' in axes:
#           self.iter_axes = 'c'  # set default value
#           self.default_coords['c']=0
#
#   def get_frame(self, c, t, m):
#       return np.array([[c, t, m]], dtype=np.uint8)
#
#   def get_frame(self, **ind):
#       return np.array((self.sizes['y'], self.sizes['x']), dtype=self.pixel_type)

#class IndexReturningReader(FramesSequenceND):
#   @property
#   def pixel_type(self):
#       return np.uint8  # the pixel datatype
#
#   def __init__(self):
#       # first call the baseclass initialization
#       super(IndexReturningReader, self).__init__()
#       self._init_axis('x', 4)
#       self._init_axis('y', 3)
#       self._init_axis('c', 2)
#       self._init_axis('m', 1)
#       self._init_axis('t', 0)
#       # register the get_frame function
#       self._register_get_frame(self.get_frame_func, 'yx')
#
#   def get_frame_func(self, c, t, m):
#       return np.array([[c, t, m]], dtype=np.uint8)

#################### methods ####################

def process_nd2_tiff(nd2file, tstart=None, tend=None, fovs=None, colors=None, xcrop=None, ycrop=None, tiffdir='TIFF'):
    """
    Write tiff images contained in an ND2 file and return the meta data.
    """
    # check file existence
    if not os.path.isfile(nd2file):
        sys.exit("File does not exist: {}".format(nd2file))

    #  load iterator
    nd2_iterator = ND2Reader(nd2file)
    axes = nd2_iterator.axes  # list of char ['t', 'm', 'c', 'z',] 'y' or 'x'
    sizes = nd2_iterator.sizes  # dict with keys in axes list and value is the corresponding dimension
    print vars(nd2_iterator).keys()
    bname = os.path.splitext(os.path.basename(nd2_iterator.filename))[0]
    print "{:<20s}{:<s}".format('bname',bname)

    print "Axes:", axes
    print "Sizes:"
    for ax in axes:
        print "{ax:<5s}{dim:<10d}".format(ax=ax,dim=sizes[ax])

    # build filtering indexes
    idx = {}
    for ax in axes:
        idx[ax]=None

    # make format dictionary
    fmtdict={}

    ## time
    if ('t' in axes):
        nt = sizes['t']
        fmt="t{{t:0{:d}d}}".format(int(np.log10(nt))+1)
        fmtdict['t']=fmt
        if (tstart is None):
            tstart = 0
        if (tend is None):
            tend = nt-1
        tstart = max(tstart,0)
        tend = min(tend,nt-1)
        print "tstart = {:d}  tend = {:d}".format(tstart,tend)
        idx['t']=np.arange(tstart,tend+1)

    ## fov
    if ('m' in axes):
        nm = sizes['m']
        fmt="f{{fov:0{:d}d}}".format(int(np.log10(nm))+1)
        fmtdict['m']=fmt
        if (fovs is None):
            fovs = np.arange(nm)
        print "FOVs"
        for i in fovs:
            print "{:<2s}FOV {:d} selected".format("",i)
        idx['m']=fovs

    ## colors
    if ('c' in axes):
        nc = sizes['c']
        fmt="c{{c:0{:d}d}}".format(int(np.log10(nc))+1)
        fmtdict['c']=fmt
        if (colors is None):
            colors = np.arange(nc)
        for i in colors:
            print "Channel {:d} selected".format(i)
        idx['c']=colors

    ## cropping
    if not ('x' in axes) and ('y' in axes):
        raise ValueError("\'y\' and \'x\' must be in the axes!")
    ### y
    ny = sizes['y']
    if not (ycrop is None):
        ylo,yhi = np.asarray(ycrop,dtype=np.uint)
        ylo = max(ylo,0)
        yhi = min(yhi,ny-1)
    else:
        ylo=0
        yhi=ny-1
    if not (ylo < yhi):
        print "Problem with y-cropping: ylo = {:d}    yhi = {:d}".format(ylo,yhi)
    if np.mod(yhi+1-ylo,2) == 1:
        yhi = yhi - 1
    idx['y']=np.arange(ylo,yhi+1, dtype=np.uint)

    ### x
    nx = sizes['x']
    if not (xcrop is None):
        nx = sizes['x']
        xlo,xhi = np.asarray(xcrop,dtype=np.uint)
        xlo = max(xlo,0)
        xhi = min(xhi,nx-1)
    else:
        xlo=0
        xhi=nx-1
    if not (xlo < xhi):
        print "Problem with x-cropping: xlo = {:d}    xhi = {:d}".format(xlo,xhi)
    if np.mod(xhi+1-xlo,2) == 1:
        xhi = xhi - 1
    idx['x']=np.arange(xlo,xhi+1,dtype=np.uint)


#    if 't' in axes:
#        nd2_iterator.iter_axes='t'
#        print len(nd2_iterator)
#        nd2_iterator = nd2_iterator[idx['t']]
#        print len(nd2_iterator)
#        nd2_iterator.bundle_axes='tcyx'
#    else:
    nd2_iterator.bundle_axes='cyx'

#    print nd2_iterator.axes
#    print nd2_iterator.sizes
    # format for file out

    if 'm' in axes:
        nd2_iterator.iter_axes='m'
        print "Starting per-FOV writing"
        for fov in idx['m']:
            print "FOV {:d}".format(fov)
            frame = nd2_iterator[fov]
            fov_no = frame.frame_no
            tiff_meta = frame.metadata
            if (fov_no != fov):
                print "Inconsistency for fov {:d}: fov_no={:d}".format(fov,fov_no)
            if 't' in axes:
                raise ValueError("Time axis handling not implemented yet.")
                frame.iter_axes='t'
            else:
                fmt=fmtdict['m']
                img = np.array(frame)
                # filtering
                ## color
                img = img[idx['c']]
                ## y cropping
                img = img[:, idx['y']]
                ## x cropping
                img = img[:,:,idx['x']]
                # write tiff as a stack
                fname = "{}_{}".format(bname,fmt)
                fname = fname.format(fov=fov)
                fileout = os.path.join(tiffdir,fname+'.tif')
                ti.imwrite(fileout, img, imagej=True, photometric='minisblack',metadata=tiff_meta)
                print "{:<20s}{:<s}".format('fileout', fileout)
    else:
        frame = nd2_iterator
        tiff_meta = frame.metadata
        if 't' in axes:
            frame.iter_axes='t'
            nt = sizes['t']
            fmt = fmtdict['t']
            fname = bname + "_" + fmt
            for t in idx['t']:
                img = np.array(frame[t])
                # filtering
                ## color
                img = img[idx['c']]
                ## y cropping
                img = img[:, idx['y']]
                ## x cropping
                img = img[:,:,idx['x']]
                # write tiff as a stack
                fileout = os.path.join(tiffdir,fname.format(t=t) +'.tif')
                ti.imwrite(fileout, img, imagej=True, photometric='minisblack',metadata=tiff_meta)
                print "{:<20s}{:<s}".format('fileout', fileout)
            sys.exit()
        else:
            img = np.array(frame[0])
            # filtering
            ## color
            img = img[idx['c']]
            ## y cropping
            img = img[:, idx['y']]
            ## x cropping
            img = img[:,:,idx['x']]
            # write tiff as a stack
            fname = bname
            fileout = os.path.join(tiffdir,fname+'.tif')
            ti.imwrite(fileout, img, imagej=True, photometric='minisblack',metadata=tiff_meta)
            print "{:<20s}{:<s}".format('fileout', fileout)

    return nd2_iterator.metadata

#    # get the color names out. Kinda roundabout way.
#    planes = [nd2f.metadata[md]['name'] for md in nd2f.metadata if md[0:6] == u'plane_' and not md == u'plane_count']
#
#    # this insures all colors will be saved when saving tiff
#    if len(planes) > 1:
#        nd2f.bundle_axes = [u'c', u'y', u'x']
#
#    # extraction range is the time points that will be taken out. Note the indexing,
#    # it is zero indexed to grab from nd2, but TIFF naming starts at 1.
#    # if there is more than one FOV (len(nd2f) != 1), make sure the user input
#    # last time index is before the actual time index. Ignore it.
#    if (p['image_start'] < 1):
#        p['image_start']=1
#    if len(nd2f) > 1 and len(nd2f) < p['image_end']:
#        p['image_end'] = len(nd2f)
#    extraction_range = range(p['image_start'], p['image_end']+1)
#
#    # loop through time points
#    for t in extraction_range:
#        # timepoint output name (1 indexed rather than 0 indexed)
#        t_id = t - 1
#        # set counter for FOV output name
#        #fov = fov_naming_start
#
#        for fov_id in range(0, nd2f.sizes[u'm']): # for every FOV
#            # fov_id is the fov index according to elements, fov is the output fov ID
#            fov = fov_id + 1
#
#            # skip FOVs as specified above
#            if len(specify_fovs) > 0 and not (fov in specify_fovs):
#                continue
#            if start_fov > -1 and (fov < start_fov):
#                continue
#
#            # set the FOV we are working on in the nd2 file object
#            nd2f.default_coords[u'm'] = fov_id
#
#            # get time picture was taken
#            seconds = copy.deepcopy(nd2f[t_id].metadata['t_ms']) / 1000.
#            minutes = seconds / 60.
#            hours = minutes / 60.
#            days = hours / 24.
#            acq_time = starttime + days
#
#            # get physical location FOV on stage
#            x_um = nd2f[t_id].metadata['x_um']
#            y_um = nd2f[t_id].metadata['y_um']
#
#            # make dictionary which will be the metdata for this TIFF
#            metadata_t = { 'fov': fov,
#                           't' : t,
#                           'jd': acq_time,
#                           'x': x_um,
#                           'y': y_um,
#                           'planes': planes}
#            metadata_json = json.dumps(metadata_t)
#
#            # get the pixel information
#            image_data = nd2f[t_id]

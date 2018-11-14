#################### imports ####################
import sys
import os
import numpy as np
#from nd2reader import ND2Reader # does not work properly
from pims_nd2 import ND2_Reader as ND2Reader
#from pims import FramesSequenceND


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

def get_nd2_images(nd2file, tstart=None, tend=None, fovs=None, colors=None, xcrop=None, ycrop=None, tiffdir='TIFF'):
    """
    Return the iterator over images contained in an ND2 file.
    """
    # check file existence
    if not os.path.isfile(nd2file):
        sys.exit("File does not exist: {}".format(nd2file))

    #  load iterator
    nd2_iterator = ND2Reader(nd2file)
    axes = nd2_iterator.axes  # list of char ['t', 'm', 'c', 'z',] 'y' or 'x'
    sizes = nd2_iterator.sizes  # dict with keys in axes list and value is the corresponding dimension
    print "Axes:", axes
    print "Sizes:"
    for ax in axes:
        print "{ax:<5s}{dim:<10d}".format(ax=ax,dim=sizes[ax])

    # build filtering indexes
    idx = {}
    for ax in axes:
        idx[ax]=None

    ## time
    if ('t' in axes):
        nt = sizes['t']
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
        if (fovs is None):
            fovs = np.arange(nm)
        print "FOVs"
        for i in fovs:
            print "{:<2s}FOV {:d} selected".format("",i)
        idx['m']=fovs

    ## colors
    if ('c' in axes):
        nc = sizes['c']
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
    idx['y']=np.arange(ylo,yhi+1)

    ### x
    nx = sizes['x']
    if not (xcrop is None):
        nx = sizes['x']
        xlo,xhi = np.asarray(xcrop,dtype=np.uint)
        xlo = max(xlo,0)
        xhi = min(xhi,ny-1)
    else:
        xlo=0
        xhi=nx-1
    if not (xlo < xhi):
        print "Problem with x-cropping: xlo = {:d}    xhi = {:d}".format(xlo,xhi)
    idx['x']=np.arange(xlo,xhi+1)

    print idx

    if 't' in axes:
        nd2_iterator.iter_axes='t'
        print len(nd2_iterator)
        nd2_iterator = nd2_iterator[idx['t']]
        print len(nd2_iterator)
        nd2_iterator.iter_axes='m'
        nd2_iterator.bundle_axes='tcyx'
    else:
        nd2_iterator.iter_axes='m'
        nd2_iterator.bundle_axes='cyx'
    print nd2_iterator.axes
    print nd2_iterator.sizes
    sys.exit()

    for fov in idx['m']:
        print "FOV {:d}".format(fov)
        subnd2 = nd2_iterator[fov]
        print subnd2[0]
        sys.exit()
        if 't' in axes:
            subnd2.iter_axes='t'
            subnd2.bundle_axes='cyx'
            print len(subnd2)
            sys.exit()
            print subnd2[idx['t']]
        else:
            print subnd2
    sys.exit()

    # metadata
    metadata = nd2_iterator.metadata
    nd2_iterator.bundle_axes='tvcyx'
    nframes,nfovs,ncolors,height,width=nd2_iterator.frame_shape
    #print nframes, nfovs, ncolors, height, width

    # filtering
    ## frames
    print "nframes = {:d}".format(nframes)
    if nstart is None:
        nstart = 0
    if nend is None:
        nend = nframes-1
    nstart = max(0,nstart)
    print "nstart={:d}".format(nstart)
    nend = min(nframes-1,nend)
    print "nend={:d}".format(nend)
    nd2_iterator=nd2_iterator[nstart:nend+1]
    nframes=len(nd2_iterator)
    print "nframes = {:d}".format(nframes)

    ## colors
    print "ncolors = {:d}".format(ncolors)
    if (colors is None):
        colors = range(ncolors)
    print colors
    #nd2_iterator=nd2_iterator[:,:,colors,:,:]

    ## HERE ##
    # there is a bug here. I cannot select the color axes.
    # maybe try to use directly pims_nd2 instead of ND2Reader
    #nd2_iterator.iter_axes='c'
    print "TEST"
    print "Starting shape \'tvcyx\':"
    print nd2_iterator.frame_shape
    nd2_iterator.bundle_axes='ctvyx'
    print "End shape \'ctvyx\':"
    print nd2_iterator.frame_shape
    nd2_iterator = nd2_iterator[colors]
    print len(nd2_iterator)
    print "TEST"
    ## HERE ##

    print nd2_iterator.frame_shape
    sys.exit()
    # conversions

    return nd2_iterator

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

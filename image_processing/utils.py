#################### imports ####################
import sys,os
import numpy as np
import scipy.sparse as ssp
import json
import datetime

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


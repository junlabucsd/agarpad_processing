#################### imports ####################
import sys,os
import numpy as np

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

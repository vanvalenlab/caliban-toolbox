from __future__ import division
import numpy as np
import json
import tempfile
import urllib
import shutil
from os.path import join
import os
from util import imread
#from utils.util import imread
from glob import glob
import tifffile as tiff


def retrieve_ff_ref(refpath, darkrefpath):
    """
    refpath: 'http://archive.simtk.org/ktrprotocol/temp/ffref_20x3bin.npz'
    darkrefpath: 'http://archive.simtk.org/ktrprotocol/temp/ffdarkref_20x3bin.npz'
    """
    try:
        temp_dir = tempfile.mkdtemp()
        urllib.urlretrieve(refpath, join(temp_dir, 'ref.npz'))
        ref = np.load(join(temp_dir, 'ref.npz'))
        urllib.urlretrieve(darkrefpath, join(temp_dir, 'darkref.npz'))
        darkref = np.load(join(temp_dir, 'darkref.npz'))
    finally:
        shutil.rmtree(temp_dir)  # delete directory
    return ref, darkref


def correct_shade(img, ref, darkref, ch):
    img = img.astype(np.float)
    d0 = img.astype(np.float) - darkref[ch]
    d1 = ref[ch] - darkref[ch]
    return d1.mean() * d0/d1


def shading_correction_folder(inputfolder, outputfolder, binning=3, magnification=20):
    """
    Covert lab specific.
    Not to be called by preprocess_operation. Use it in separate from a pipeline.
    """
    refpath = 'http://archive.simtk.org/ktrprotocol/temp/ffref_{0}x{1}bin.npz'.format(magnification, binning)
    darkrefpath = 'http://archive.simtk.org/ktrprotocol/temp/ffdarkref_{0}x{1}bin.npz'.format(magnification, binning)
    ref, darkref = retrieve_ff_ref(refpath, darkrefpath)
    parentfolder = inputfolder
    for dirname, subdirlist, filelist in os.walk(parentfolder):
        if 'metadata.txt' in filelist:
            outputdir = join(outputfolder, dirname.split(parentfolder)[-1])
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            with open(join(dirname, 'metadata.txt')) as mfile:
                data = json.load(mfile)
                channels = data['Summary']['ChNames']
            for chnum, ch in enumerate(channels):
                pathlist = glob(join(dirname, '*channel{0:03d}*'.format(chnum)))
                for path in pathlist:
                    try:
                        img = correct_shade(imread(path), ref, darkref, ch)
                    except:
                        print "ch might not exist as a reference."
                        img = imread(path)
                    tiff.imsave(join(outputdir, os.path.basename(path)), img.astype(np.float32))

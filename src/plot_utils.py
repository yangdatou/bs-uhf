import numpy
import scipy
from numpy import load
import matplotlib as mpl
from matplotlib import pyplot as plt

def read_data_from_h5(filename):
    import h5py

    data_dict = {}

    with h5py.File(filename, "r") as f:
        data = {}
        keys = f.keys()
        

        for key in keys:
            
            r_dict  = f[key]
            
            for k, v in r_dict.items():
                if k not in data_dict:
                    data_dict[k] = []
                data_dict[k].append((float(key), float(v[()])))

    for k, v in data_dict.items():
        data_dict[k] = numpy.array(v)

    return data_dict

params = {
        "font.size":       20,
        "axes.titlesize":  16,
        "axes.labelsize":  24,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.subplot.wspace":0.0,
        "figure.subplot.hspace":0.0,
        "axes.spines.right": True,
        "axes.spines.top":   True,
        "xtick.direction":'in',
        "ytick.direction":'in',
        "text.usetex": True,
        "font.family": "serif",
        'text.latex.preamble': r"\usepackage{amsmath}"
}
mpl.rcParams.update(params)

colors = ["#016b7e", "#099da0", "#9cd8c7", "#ecdcae", "#f1a502", "#d17203", "#c44703", "#b82513", "#a6272d"]
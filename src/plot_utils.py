import numpy
import scipy
from numpy import load
import matplotlib as mpl
from matplotlib import pyplot as plt

params = {
        "font.size":       18,
        "axes.titlesize":  20,
        "axes.labelsize":  20,
        "legend.fontsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
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

colors  = ["f94144","f3722c","f8961e","f9844a","f9c74f","90be6d","43aa8b","4d908e","577590","277da1"]
colors  = [f"#{color}" for color in colors]

def get_plot_data(filename):
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

def get_plot_style(label_list):
    style_dict = {}

    for i, label in enumerate(label_list):
        color = colors[3 * i % len(colors)]
        label_split = label.split("_")

        if len(label_split) == 2:
            plot_label = label_split[1].upper()

        else:
            plot_label = "-".join(label_split[1:]).upper()

        if plot_label == "FCI":
            style_dict[label] = {
                "color": color,
                "marker": "None",
                "linestyle": "-",
                "linewidth": 2.0,
                "markersize": 4.0,
                "label": plot_label,
            }

        elif plot_label.split("-")[0] == "NOCI":
            style_dict[label] = {
                "color": color,
                "marker": ">",
                "linestyle": "-",
                "linewidth": 2.0,
                "markersize": 4.0,
                "label": plot_label,
            }
        
        elif plot_label.split("-")[0] == "BS":
            style_dict[label] = {
                "color": color,
                "marker": "None",
                "linestyle": "dotted",
                "linewidth": 2.0,
                "markersize": 4.0,
                "label": plot_label,
            }

        else:
            style_dict[label] = {
                "color": color,
                "marker": "None",
                "linestyle": "-.",
                "linewidth": 2.0,
                "markersize": 4.0,
                "label": plot_label,
            }

    return style_dict

def plot_data(data_dict, style_dict, label_list):
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey="row")

    ax = axs[0]
    for i, label in enumerate(label_list):
        x = data_dict[label][:,0]
        y = data_dict[label][:,1] + data_dict["ene_nuc"][:,1]
        ax.plot(x, y, **style_dict[label])

    ax.legend(loc=(0.0, 1.05), fancybox=False, framealpha=1.0,
            edgecolor='silver', frameon=False, fontsize=15, ncol=3)

    ax.set_ylabel(r"$E$ (Hartree)")

    ax = axs[1]
    for i, label in enumerate(label_list):
        x = data_dict[label][:,0]
        y = data_dict[label][:,1] - data_dict["ene_fci"][:,1]
        ax.plot(x, y, **style_dict[label])

    ax.set_ylabel(r"$E - E_{\rm FCI}$ (Hartree)")
    ax.set_xlabel(r"$R$ (Bohr)")
    fig.tight_layout(w_pad=0.8, h_pad=0.8)

    return fig, axs
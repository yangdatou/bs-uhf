import numpy
import scipy
from numpy import load
import matplotlib as mpl
from matplotlib import pyplot as plt

from pyscf.lib import chkfile

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

# colors  = ["f94144","f3722c","f8961e","f9844a","f9c74f","90be6d","43aa8b","4d908e","577590","277da1"]
# colors  = [f"#{color}" for color in colors]
colors  = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:pink", "tab:olive", "tab:cyan"]

def get_plot_data(filename):
    import h5py

    data_dict = {}

    with h5py.File(filename, "r") as f:
        keys = f.keys()
        for key in keys:
            r_dict  = f[key]
            for k, v in r_dict.items():
                if k not in data_dict:
                    data_dict[k] = []

                data_dict[k].append((float(key), float(v)))

    for k, v in data_dict.items():
        data_dict[k] = numpy.array(v)

    return data_dict

def dump_h5_data(filename, dump_key_list=None, outfile=None):
    import h5py

    nrow     = len(dump_key_list)
    key_list = []
    with h5py.File(filename, "r") as f:
        keys = f.keys()
        for key in keys:
            key_list.append(key)

    title_list = []
    data_str   = ""
    data_dict  = {k: [] for k in dump_key_list}
    data_dict["x"] = []

    for k1 in key_list:
        x = float(k1)
        data_dict["x"].append(x)
        data_list = []

        for k2 in dump_key_list:
            k2_bra = k2.find("[")
            k2_ket = k2.find("]")
            data = None

            if k2_bra == -1 and k2_ket == -1:
                print(f"Reading {k1}/{k2} from {filename}")
                data = chkfile.load(filename, f"{k1}/{k2}")

            if k2_bra != -1 and k2_ket != -1:
                print(f"Reading {k1}/{k2} from {filename}")
                k2_key, k2_idx = k2[:k2_bra], int(k2[k2_bra+1:k2_ket])
                data = chkfile.load(filename, f"{k1}/{k2_key}")[k2_idx]

            if isinstance(data, float):
                if k2 not in title_list:
                    title_list.append(k2)
                data_list.append(data)

            assert data is not None, "data is None"
            data_dict[k2].append(data)
        
        if len(data_list) > 0:
            data_str += ",".join([f"{data: 16.8f}" for data in [x] + data_list]) + "\n"

    for k in data_dict:
        data_dict[k] = numpy.array(data_dict[k])
    
    if len(title_list) > 0:
        title_str = f"# {'x':>15s} " + " ".join([f"{title:>16s}" for title in title_list])

        print("Data from file: ", filename)
        print(title_str)
        print(data_str)

        if outfile is not None:
            with open(outfile, "w") as f:
                f.write(title_str + "\n" + data_str)

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

if __name__ == "__main__":
    dump_h5_data(
        "../work/data/h4-td-sto-3g.h5", outfile="../work/data/h4-td-sto-3g.csv",
        dump_key_list=["ene-uhf-bs[0]", "ene-uhf-bs[1]", "ene-uhf-bs[2]", "ene-uhf-bs"]
        )
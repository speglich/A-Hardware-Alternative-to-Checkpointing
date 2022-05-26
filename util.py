import h5py
import numpy as np
import socket
import os.path
import csv
import skimage.measure
from examples.seismic.model import Model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa


def from_hdf5(filename, **kwargs):
    f = h5py.File(filename, 'r')
    origin = kwargs.pop('origin', None)
    if origin is None:
        origin_key = kwargs.pop('origin_key', 'o')
        origin = f[origin_key][()]

    spacing = kwargs.pop('spacing', None)
    if spacing is None:
        spacing_key = kwargs.pop('spacing_key', 'd')
        spacing = f[spacing_key][()]
    nbpml = kwargs.pop('nbpml', 20)
    datakey = kwargs.pop('datakey', None)
    if datakey is None:
        raise ValueError("datakey must be known - what is the name of the data in the file?")  # noqa
    space_order = kwargs.pop('space_order', None)
    dtype = kwargs.pop('dtype', None)
    data_m = f[datakey][()]
    data_vp = np.sqrt(1/data_m).astype(dtype)

    if len(data_vp.shape) > 2:
        data_vp = np.transpose(data_vp, (1, 2, 0))
    else:
        data_vp = np.transpose(data_vp, (1, 0))
    shape = data_vp.shape
    return Model(space_order=space_order, vp=data_vp, origin=origin,
                 shape=shape, dtype=dtype, spacing=spacing, nbl=nbpml,
                 bcs="damp")


def to_hdf5(data, filename, datakey='data', additional=None):
    with h5py.File(filename, 'w') as f:
        f.create_dataset(datakey, data=data, dtype=data.dtype)
        if additional is not None:
            for k, v in additional.items():
                f.create_dataset(k, data=v, dtype=v.dtype)


def error_norm(original, decompressed, ord=2):
    error_field = original - decompressed
    return np.linalg.norm(np.ravel(error_field), ord)


def error_L0(original, decompressed):
    return error_norm(original, decompressed, 0)


def error_L1(original, decompressed):
    return error_norm(original, decompressed, 1)


def error_L2(original, decompressed):
    return error_norm(original, decompressed, 2)


def error_Linf(original, decompressed):
    return error_norm(original, decompressed, np.inf)


def write_results(data, results_file):
    hostname = socket.gethostname()
    if not os.path.isfile(results_file):
        write_header = True
    else:
        write_header = False

    data['hostname'] = hostname
    fieldnames = list(data.keys())
    with open(results_file, 'a') as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(data)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def error_angle(original, decompressed):
    return angle_between(np.ravel(original), np.ravel(decompressed))


def error_psnr(original, decompressed):
    range = np.max(original) - np.min(original)
    return skimage.metrics.peak_signal_noise_ratio(original, decompressed, data_range=range)


def read_csv(filename):
    results = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k, v in row.items():
                results_list = results.get(k, [])
                try:
                    v = float(v)
                except ValueError:
                    pass
                results_list.append(v)
                results[k] = results_list
    return results


def plot_field(data, output_file, basepath='figs/'):
    shape = data.shape
    print(shape)
    vmax = np.max(data)
    slice_loc = 440
    if len(shape) > 2:
        data = data[slice_loc]

    plt.imshow(np.transpose(data), vmax=vmax, vmin=-vmax, cmap="seismic",
               extent=[0, 20, 0.001*(shape[-1]-1)*25, 0])

    plt.xlabel("X (km)")
    plt.ylabel("Depth (km)")
    cb = plt.colorbar(shrink=.3, pad=.01, aspect=10)
    for i in cb.ax.yaxis.get_ticklabels():
        i.set_fontsize(12)

        cb.set_label('Pressure')

    plt.savefig(basepath+output_file, bbox_inches='tight')
    plt.clf()

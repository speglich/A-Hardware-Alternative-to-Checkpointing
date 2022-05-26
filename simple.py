from argparse import ArgumentParser
from examples.seismic import (Receiver, TimeAxis, RickerSource,
                              AcquisitionGeometry)
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.tti import AnisotropicWaveSolver
import numpy as np
from util import from_hdf5

from devito import TimeFunction
from devito.data.allocators import ExternalAllocator


def overthrust_setup(filename, kernel='OT2', tn=4000, src_coordinates=None,
                     space_order=2, datakey='m0', nbpml=40, dtype=np.float32,
                     **kwargs):
    model = from_hdf5(filename, space_order=space_order, nbpml=nbpml,
                      datakey=datakey, dtype=dtype)

    shape = model.shape
    spacing = model.spacing
    nrec = shape[0]

    if src_coordinates is None:
        src_coordinates = np.empty((1, len(spacing)))
        src_coordinates[0, :] = np.array(model.domain_size) * .5
        if len(shape) > 1:
            src_coordinates[0, -1] = model.origin[-1] + 2 * spacing[-1]

    rec_coordinates = np.empty((nrec, len(spacing)))
    rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    if len(shape) > 1:
        rec_coordinates[:, 1] = np.array(model.domain_size)[1] * .5
        rec_coordinates[:, -1] = model.origin[-1] + 2 * spacing[-1]

    # Create solver object to provide relevant operator
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=0.0, tn=tn, src_type='Ricker', f0=0.008)

    solver = AcousticWaveSolver(model, geometry, kernel=kernel,
                                space_order=space_order, **kwargs)
    return solver


def overthrust_setup_tti(filename, tn=4000, space_order=2, nbpml=40,
                         **kwargs):
    model = from_hdf5(filename, space_order=space_order, nbpml=nbpml,
                      datakey='m0', dtype=np.float32)
    shape = model.vp.shape
    spacing = model.shape
    nrec = shape[0]

    # Derive timestepping from model spacing
    dt = model.critical_dt
    t0 = 0.0
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.015,
                       time_range=time_range)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    if len(shape) > 1:
        src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]

    # Define receiver geometry (spread across x, just below surface)
    rec = Receiver(name='rec', grid=model.grid, time_range=time_range,
                   npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0],
                                             num=nrec)
    if len(shape) > 1:
        rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # Create solver object to provide relevant operators
    return AnisotropicWaveSolver(model, source=src, receiver=rec,
                                 space_order=space_order, **kwargs)


def run(space_order=4, kernel='OT4', nbpml=40, filename='', **kwargs):

    if kernel in ['OT2', 'OT4']:
        solver = overthrust_setup(filename=filename, nbpml=nbpml,
                                  space_order=space_order, kernel=kernel,
                                  **kwargs)
    elif kernel == 'TTI':
        solver = overthrust_setup_tti(filename=filename, nbpml=nbpml,
                                      space_order=space_order, kernel=kernel,
                                      **kwargs)
    else:
        raise ValueError()

    grid = solver.model.grid

    rec = solver.geometry.rec

    dt = solver.model.critical_dt

    numpy_array = np.memmap("/scr01/array.dat", mode='w+',  shape=(solver.geometry.nt,893,893,299), dtype=np.float32)

    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=space_order,
        allocator=ExternalAllocator(numpy_array),
        initializer=lambda x: None,
        save=solver.geometry.nt)

    fw_op = solver.op_fwd(save=True)
    rev_op = solver.op_grad(save=True)

    fw_op.apply(rec=rec, src=solver.geometry.src, u=u, dt=dt)
    rev_op.apply(u=u, dt=dt, rec=rec)

    return

if __name__ == "__main__":

    description = ("Example script for a set of acoustic operators.")

    parser = ArgumentParser(description=description)

    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("-k", dest="kernel", default='OT2',
                        choices=['OT2', 'OT4', 'TTI'],
                        help="Choice of finite-difference kernel")

    args = parser.parse_args()

    run(nbpml=args.nbpml,
        space_order=args.space_order,
        kernel=args.kernel,
        filename='overthrust_3D_initial_model.h5')

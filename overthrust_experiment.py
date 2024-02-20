from argparse import ArgumentParser
from examples.seismic import (Receiver, TimeAxis, RickerSource,
                              AcquisitionGeometry)
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.tti import AnisotropicWaveSolver
import numpy as np

from util import from_hdf5
from shutil import copyfile
from devito import TimeFunction, Function
from devito.data.allocators import ExternalAllocator
from devito import configuration, compiler_registry
from devito.arch.compiler import GNUCompiler

from examples.seismic import demo_model

solvers = {
    'acoustic' : {
        'disk' : {
            'single':{
                'forward' : 'src/non-mpi/forward.c',
                'gradient' : 'src/non-mpi/gradient.c'
            },
            'mpi': {
                'forward' : 'src/mpi/forward.c',
                'gradient' : 'src/mpi/gradient.c'
            }
        },
        'ram' : {
            'single':{
                'forward' : 'src/ram/non-mpi/forward.c',
                'gradient' : 'src/ram/non-mpi/gradient.c'
            },
            'mpi': {
                'forward' : 'src/ram/mpi/forward.c',
                'gradient' : 'src/ram/mpi/gradient.c'
            }
        },
        'compression' : {
            'single':{
                'forward' : 'src/compression/non-mpi/forward.c',
                'gradient' : 'src/compression/non-mpi/gradient.c'
            },
        }
    },
    'tti' : {
        'disk' : {
            'single':{
                'forward' : 'src/tti/disk/single/forward.c',
                'gradient' : 'src/tti/disk/single/gradient.c'
            },
        },
        'ram' : {
            'single':{
                'forward' : 'src/tti/ram/single/forward.c',
                'gradient' : 'src/tti/ram/single/gradient.c'
            }
        }
    }
}

def operatorInjector(op, payload):

    configuration['jit-backdoor'] = True
    configuration.add('payload', payload)

    # Force compilation *and* loading upon the next `op.apply`

    op._lib = None
    op._cfunction = None

    if op._soname:
        del op._soname

    cfile = "%s.c" % str(op._compiler.get_jit_dir().joinpath(op._soname))

    copyfile(payload, cfile)

    return

def overthrust_setup(filename, kernel='OT2', tn=10, src_coordinates=None,
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


def overthrust_setup_tti(filename, kernel='staggered', tn=10, src_coordinates=None,
                         space_order=2, datakey='m0', nbpml=40, dtype=np.float32,
                         **kwargs):

    shape = (100, 100, 100)
    spacing = (10., 10., 10.)
    nrec = shape[0]

    # Use demo model instead of overthrust model for now
    model = demo_model('layers-tti', spacing=spacing, space_order=8,
                    shape=shape, nbl=nbpml, nlayers=2)

    # initialize Thomsem parameters to those used in Mu et al., (2020)
    model.update('vp', np.ones(shape)*3.6) # km/s
    model.update('epsilon', np.ones(shape)*0.23)
    model.update('delta', np.ones(shape)*0.17)
    model.update('theta', np.ones(shape)*(45.*(np.pi/180.))) # radians

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

    # Create solver object to provide relevant operators
    solver =  AnisotropicWaveSolver(model, geometry=geometry, kernel=kernel,
                                    space_order=space_order, **kwargs)
    return solver


def run(space_order=4, kernel='OT4', nbpml=40, filename='', to_disk=True, compression=False, mpi=False, **kwargs):

    if kernel in ['OT2', 'OT4']:
        solver = overthrust_setup(filename=filename, nbpml=nbpml,
                                  space_order=space_order, kernel=kernel,
                                  **kwargs)
    elif kernel == 'TTI':
        solver = overthrust_setup_tti(filename=filename, nbpml=nbpml,
                                      space_order=space_order, kernel='centered',
                                      **kwargs)
    else:
        raise ValueError("Unknown kernel")

    grid = solver.model.grid
    rec = solver.geometry.rec
    dt = solver.model.critical_dt
    grad = Function(name='grad', grid=solver.model.grid)

    kname = 'acoustic' if kernel in ['OT2', 'OT4'] else 'tti'
    mode = 'compression' if compression else 'disk' if to_disk else 'ram'
    executor = 'mpi' if mpi else 'single'

    print("Running %s %s %s" % (kname, mode, executor))

    if kname == 'acoustic':
        if to_disk:

            u = TimeFunction(name='u', grid=grid, time_order=2, space_order=space_order)
            fw_op = solver.op_fwd(save=False)
            rev_op = solver.op_grad(save=False)

            operatorInjector(fw_op, solvers[kname][mode][executor]['forward'])
            operatorInjector(rev_op, solvers[kname][mode][executor]['gradient'])

        else:

            u = TimeFunction(name='u', grid=grid, time_order=2, space_order=space_order, save=solver.geometry.nt)
            fw_op = solver.op_fwd(save=True)
            rev_op = solver.op_grad(save=True)

            operatorInjector(fw_op, solvers[kname][mode][executor]['forward'])
            operatorInjector(rev_op, solvers[kname][mode][executor]['gradient'])

        fw_op.apply(rec=rec, src=solver.geometry.src, u=u, dt=dt)
        rev_op.apply(u=u, dt=dt, rec=rec, grad=grad)

    elif kname == 'tti':

        if to_disk:
            u = TimeFunction(name='u', grid=grid, time_order=2, space_order=space_order)
            v = TimeFunction(name='v', grid=grid, time_order=2, space_order=space_order)
            fw_op = solver.op_fwd(save=False)
            rev_op = solver.op_jacadj(save=False)

            operatorInjector(fw_op, solvers[kname][mode][executor]['forward'])
            operatorInjector(rev_op, solvers[kname][mode][executor]['gradient'])

        else:
            u = TimeFunction(name='u', grid=grid, time_order=2, space_order=space_order, save=solver.geometry.nt)
            v = TimeFunction(name='v', grid=grid, time_order=2, space_order=space_order, save=solver.geometry.nt)
            fw_op = solver.op_fwd(save=True)
            rev_op = solver.op_jacadj(save=True)

            operatorInjector(fw_op, solvers[kname][mode][executor]['forward'])
            operatorInjector(rev_op, solvers[kname][mode][executor]['gradient'])

        fw_op.apply(u=u, v=v, rec=rec, src=solver.geometry.src, dt=dt)
        rev_op.apply(u0=u, v0=v, rec=rec, dm=grad, dt=dt)

    # get norm, max and min of the gradient
    norm = np.linalg.norm(grad.data.flatten())
    max_val = np.max(grad.data.flatten())
    min_val = np.min(grad.data.flatten())

    print("Norm: %f" % norm)
    print("Max: %f" % max_val)
    print("Min: %f" % min_val)

    return grad

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

    parser.add_argument("--mpi", default=False, action="store_true",
                        help="Use MPI on experiments")

    parser.add_argument("--compression", default=False, action="store_true",
                        help="Use Compression on experiments")

    parser.add_argument("--rate", default=16,
                        type=int, help="Set the Compression Rate to compression")

    parser.add_argument("--disks", default=8, type=int,
                        help="Number of PML layers around the domain")

    parser.add_argument("--ram", default=False, action="store_true",
                        help="Use MPI on experiments")

    parser.add_argument("--cache", default=False, action="store_true",
                        help="Disable O_DIRECT on experiments")

    args = parser.parse_args()

    class ZFPCompiler(GNUCompiler):
        def __init__(self, *c_args, **kwargs):

            super(ZFPCompiler, self).__init__(*c_args, **kwargs)

            #self.libraries.append("zfp")
            if args.cache:
                d_cache = "CACHE=1"
                self.defines.append(d_cache)
            d_ndisks = "NDISKS=%d" % args.disks
            d_rate = "RATE=%d" % args.rate

            self.defines.append(d_ndisks)
            self.defines.append(d_rate)

    compiler_registry['zfpcompile'] = ZFPCompiler
    configuration.add("compiler", "custom", list(compiler_registry), callback=lambda i: compiler_registry[i]())
    configuration['compiler'] = 'zfpcompile'

    to_disk = not args.ram
    grad = run(nbpml=args.nbpml,
        space_order=args.space_order,
        kernel=args.kernel,
        filename='overthrust_3D_initial_model.h5',
        to_disk=to_disk,
        compression=args.compression,
        mpi=args.mpi)

    # Write u to disk

    # file name based on arguments
    if to_disk:
        file_name = "grad_%s_%s_%s_%s_%s_%s_%s" % (args.space_order, args.kernel, args.nbpml, args.compression, args.mpi, args.rate, args.disks)
    else:
        file_name = "grad_ram_%s_%s_%s" % (args.space_order, args.kernel, args.nbpml)

    grad.data.tofile(file_name)
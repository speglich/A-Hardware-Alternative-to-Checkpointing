import numpy as np
from devito import configuration
from examples.seismic import demo_model
from devito import gaussian_smooth

from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver

from devito import TimeFunction, Operator, Eq, solve, Function
from examples.seismic import PointSource

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shutil import copyfile

mpl.rc('font', size=16)
mpl.rc('figure', figsize=(8, 6))

files = {
    'forward' : 'src/RTM/non-mpi/forward.c',
    'gradient' : 'src/RTM/non-mpi/kernel.c',
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

def plot_image(data, vmin=None, vmax=None, colorbar=True, cmap="gray"):
    """
    Plot image data, such as RTM images or FWI gradients.
    Parameters
    ----------
    data : ndarray
        Image data to plot.
    cmap : str
        Choice of colormap. Defaults to gray scale for images as a
        seismic convention.
    """
    plot = plt.imshow(np.transpose(data),
                      vmin=vmin or 0.9 * np.min(data),
                      vmax=vmax or 1.1 * np.max(data),
                      cmap=cmap)

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    plt.savefig('foo.png')

def ImagingOperator(model, image):
    # Define the wavefield with the size of the model and the time dimension
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)

    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=4,
                     save=geometry.nt)

    # Define the wave equation, but with a negated damping term
    eqn = model.m * v.dt2 - v.laplace + model.damp * v.dt.T

    # Use `solve` to rearrange the equation into a stencil expression
    stencil = Eq(v.backward, solve(eqn, v.backward))

    # Define residual injection at the location of the forward receivers
    dt = model.critical_dt
    residual = PointSource(name='residual', grid=model.grid,
                           time_range=geometry.time_axis,
                           coordinates=geometry.rec_positions)
    res_term = residual.inject(field=v.backward, expr=residual * dt**2 / model.m)

    # Correlate u and v for the current time step and add it to the image
    image_update = Eq(image, image - u * v)

    return Operator([stencil] + res_term + [image_update],
                    subs=model.spacing_map)

if __name__ == "__main__":

    #preset = 'layers-isotropic'
    preset = 'marmousi2d-isotropic'

    # Standard preset with a simple two-layer model
    if preset == 'layers-isotropic':
        def create_model(grid=None):
            return demo_model('layers-isotropic', origin=(0., 0.), shape=(101, 101),
                            spacing=(10., 10.), nbl=20, grid=grid, nlayers=2)
        filter_sigma = (1, 1)
        nshots = 21
        nreceivers = 101
        t0 = 0.
        tn = 1000.  # Simulation last 1 second (1000 ms)
        f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)


    # A more computationally demanding preset based on the 2D Marmousi model
    if preset == 'marmousi2d-isotropic':
        def create_model(grid=None):
            return demo_model('marmousi2d-isotropic', data_path='../data/',
                            grid=grid, nbl=20)
        filter_sigma = (6, 6)
        nshots = 301  # Need good covergae in shots, one every two grid points
        nreceivers = 601  # One recevier every grid point
        t0 = 0.
        tn = 3500.  # Simulation last 3.5 second (3500 ms)
        f0 = 0.025  # Source peak frequency is 25Hz (0.025 kHz

    # Create true model from a preset
    model = create_model()

    # Create initial model and smooth the boundaries
    model0 = create_model(grid=model.grid)
    gaussian_smooth(model0.vp, sigma=filter_sigma)

    # First, position source centrally in all dimensions, then set depth
    src_coordinates = np.empty((1, 2))
    src_coordinates[0, :] = np.array(model.domain_size) * .5
    src_coordinates[0, -1] = 20.  # Depth is 20m

    # Define acquisition geometry: receivers

    # Initialize receivers for synthetic and imaging data
    rec_coordinates = np.empty((nreceivers, 2))
    rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nreceivers)
    rec_coordinates[:, 1] = 30.

    # Geometry
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=.010, src_type='Ricker')

    solver = AcousticWaveSolver(model, geometry, space_order=4)
    true_d , _, _ = solver.forward(vp=model.vp)
    smooth_d, _, _ = solver.forward(vp=model0.vp)

    # Prepare the varying source locations
    source_locations = np.empty((nshots, 2), dtype=np.float32)
    source_locations[:, 0] = np.linspace(0., 1000, num=nshots)
    source_locations[:, 1] = 30.

    # Create image symbol and instantiate the previously defined imaging operator
    image = Function(name='image', grid=model.grid)

    op_fwd = solver.op_fwd(save=True)
    operatorInjector(op_fwd, files ['forward'])

    op_imaging = ImagingOperator(model, image)
    operatorInjector(op_imaging, files ['gradient'])

    for i in range(nshots):
        print('Imaging source %d out of %d' % (i+1, nshots))

        # Update source location
        geometry.src_positions[0, :] = source_locations[i, :]

        # Generate synthetic data from true model
        true_d, _, _ = solver.forward(vp=model.vp)

        u0 = TimeFunction(name='u', grid=solver.model.grid, save=solver.geometry.nt,
                              time_order=2, space_order=solver.space_order)

        op_fwd.apply(src=solver.geometry.src, rec=solver.geometry.rec, u=u0, dt=solver.dt)

        # Compute gradient from the data residual
        v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)
        residual = smooth_d.data - true_d.data
        op_imaging(v=v, vp=model0.vp, dt=model0.critical_dt,
                residual=residual)

    plot_image(np.diff(image.data, axis=1))

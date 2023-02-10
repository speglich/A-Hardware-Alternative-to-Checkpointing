import numpy as np
from numpy import linalg
from devito.tools import memoized_meth
from devito import Function, info, smooth
from examples.seismic.tti import tti_setup

from examples.seismic import demo_model, setup_geometry, Receiver
from devito import Function, TimeFunction, DevitoCheckpoint, CheckpointOperator

from examples.seismic.acoustic.operators import (
    ForwardOperator, AdjointOperator, GradientOperator, BornOperator
)
from pyrevolve import Revolver

from devito import configuration
from shutil import copyfile

files = {
    'forward' : 'src/gradient/forward.c',
    'gradient' : 'src/gradient/gradient.c',
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

class AcousticWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.
    Parameters
    ----------
    model : Model
        Physical model with domain parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    kernel : str, optional
        Type of discretization, centered or shifted.
    space_order: int, optional
        Order of the spatial stencil discretisation. Defaults to 4.
    """
    def __init__(self, model, geometry, kernel='OT2', space_order=4, **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="damp")
        self.geometry = geometry

        assert self.model.grid == geometry.grid

        self.space_order = space_order
        self.kernel = kernel

        # Cache compiler options
        self._kwargs = kwargs

    @property
    def dt(self):
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        if self.kernel == 'OT4':
            return self.model.dtype(1.73 * self.model.critical_dt)
        return self.model.critical_dt

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_adj(self):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, save=None, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_grad(self, save=True):
        """Cached operator for gradient runs"""
        return GradientOperator(self.model, save=save, geometry=self.geometry,
                                kernel=self.kernel, space_order=self.space_order,
                                **self._kwargs)

    @memoized_meth
    def op_born(self):
        """Cached operator for born runs"""
        return BornOperator(self.model, save=None, geometry=self.geometry,
                            kernel=self.kernel, space_order=self.space_order,
                            **self._kwargs)

    def forward(self, src=None, rec=None, u=None, model=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.
        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        u : TimeFunction, optional
            Stores the computed wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        save : bool, optional
            Whether or not to save the entire (unrolled) wavefield.
        Returns
        -------
        Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefield if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              save=self.geometry.nt if save else None,
                              time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        fw_op = self.op_fwd(save)

        if save:
            operatorInjector(fw_op, files ['forward'])

        summary = fw_op.apply(src=src, rec=rec, u=u,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)

        return rec, u, summary

    def adjoint(self, rec, srca=None, v=None, model=None, **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.
        Parameters
        ----------
        rec : SparseTimeFunction or array-like
            The receiver data. Please note that
            these act as the source term in the adjoint run.
        srca : SparseTimeFunction or array-like
            The resulting data for the interpolated at the
            original source location.
        v: TimeFunction, optional
            The computed wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        Returns
        -------
        Adjoint source, wavefield and performance summary.
        """
        # Create a new adjoint source and receiver symbol
        srca = srca or self.geometry.new_src(name='srca', src_type=None)

        # Create the adjoint wavefield if not provided
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        summary = self.op_adj().apply(srca=srca, rec=rec, v=v,
                                      dt=kwargs.pop('dt', self.dt), **kwargs)
        return srca, v, summary

    def jacobian_adjoint(self, rec, u, src=None, v=None, grad=None, model=None,
                         checkpointing=False, **kwargs):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modelling function, ie. the action of the
        Jacobian adjoint on an input data.
        Parameters
        ----------
        rec : SparseTimeFunction
            Receiver data.
        u : TimeFunction
            Full wavefield `u` (created with save=True).
        v : TimeFunction, optional
            Stores the computed wavefield.
        grad : Function, optional
            Stores the gradient field.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        Returns
        -------
        Gradient field and performance summary.
        """
        dt = kwargs.pop('dt', self.dt)
        # Gradient symbol
        grad = grad or Function(name='grad', grid=self.model.grid)

        # Create the forward wavefield
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        if checkpointing:
            u = TimeFunction(name='u', grid=self.model.grid,
                             time_order=2, space_order=self.space_order)
            cp = DevitoCheckpoint([u])
            n_checkpoints = None
            wrap_fw = CheckpointOperator(self.op_fwd(save=False),
                                         src=src or self.geometry.src,
                                         u=u, dt=dt, **kwargs)
            wrap_rev = CheckpointOperator(self.op_grad(save=False), u=u, v=v,
                                          rec=rec, dt=dt, grad=grad, **kwargs)

            # Run forward
            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, rec.data.shape[0]-2)
            wrp.apply_forward()
            summary = wrp.apply_reverse()
        else:
            op_grad = self.op_grad()
            operatorInjector(op_grad, files ['gradient'])
            summary = op_grad.apply(rec=rec, grad=grad, v=v, u=u, dt=dt,
                                           **kwargs)
        return grad, summary

    def jacobian(self, dmin, src=None, rec=None, u=None, U=None, model=None, **kwargs):
        """
        Linearized Born modelling function that creates the necessary
        data objects for running an adjoint modelling operator.
        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        u : TimeFunction, optional
            The forward wavefield.
        U : TimeFunction, optional
            The linearized wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefields u and U if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)
        U = U or TimeFunction(name='U', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        summary = self.op_born().apply(dm=dmin, u=u, U=U, src=src, rec=rec,
                                       dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, u, U, summary

    # Backward compatibility
    born = jacobian
    gradient = jacobian_adjoint

def iso_setup(shape=(50, 50, 50), spacing=(15.0, 15.0, 15.0),
                   tn=500., kernel='OT2', space_order=4, nbl=10,
                   preset='layers-isotropic', fs=False, **kwargs):
    model = demo_model(preset, space_order=space_order, shape=shape, nbl=nbl,
                       dtype=kwargs.pop('dtype', np.float32), spacing=spacing,
                       fs=fs, **kwargs)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, geometry, kernel=kernel,
                                space_order=space_order, **kwargs)
    return solver

def test_gradientFWI(dtype, space_order, kernel, shape, ckp, setup_func,
                         time_order):
        """
        This test ensures that the FWI gradient computed with devito
        satisfies the Taylor expansion property:
        .. math::
            \Phi(m0 + h dm) = \Phi(m0) + \O(h) \\
            \Phi(m0 + h dm) = \Phi(m0) + h \nabla \Phi(m0) + \O(h^2) \\
            \Phi(m0) = .5* || F(m0 + h dm) - D ||_2^2

        where
        .. math::
            \nabla \Phi(m0) = <J^T \delta d, dm> \\
            \delta d = F(m0+ h dm) - D \\

        with F the Forward modelling operator.
        """
        spacing = tuple(10. for _ in shape)
        wave = setup_func(shape=shape, spacing=spacing, dtype=dtype, kernel=kernel,
                          tn=400.0, space_order=space_order, nbl=40,
                          time_order=time_order)

        vel0 = Function(name='vel0', grid=wave.model.grid, space_order=space_order)
        smooth(vel0, wave.model.vp)
        v = wave.model.vp.data
        dm = dtype(wave.model.vp.data**(-2) - vel0.data**(-2))
        # Compute receiver data for the true velocity
        rec = wave.forward()[0]

        # Compute receiver data and full wavefield for the smooth velocity
        if setup_func is tti_setup:
            rec0, u0, v0, _ = wave.forward(vp=vel0, save=True)
        else:
            rec0, u0 = wave.forward(vp=vel0, save=True)[0:2]

        # Objective function value
        F0 = .5*linalg.norm(rec0.data - rec.data)**2

        # Gradient: <J^T \delta d, dm>
        residual = Receiver(name='rec', grid=wave.model.grid, data=rec0.data - rec.data,
                            time_range=wave.geometry.time_axis,
                            coordinates=wave.geometry.rec_positions)

        if setup_func is tti_setup:
            gradient, _ = wave.jacobian_adjoint(residual, u0, v0, vp=vel0,
                                                checkpointing=ckp)
        else:
            gradient, _ = wave.jacobian_adjoint(residual, u0, vp=vel0,
                                                checkpointing=ckp)

        G = np.dot(gradient.data.reshape(-1), dm.reshape(-1))

        # FWI Gradient test
        H = [0.5, 0.25, .125, 0.0625, 0.0312, 0.015625, 0.0078125]
        error1 = np.zeros(7)
        error2 = np.zeros(7)
        for i in range(0, 7):
            # Add the perturbation to the model
            def initializer(data):
                data[:] = np.sqrt(vel0.data**2 * v**2 /
                                  ((1 - H[i]) * v**2 + H[i] * vel0.data**2))
            vloc = Function(name='vloc', grid=wave.model.grid, space_order=space_order,
                            initializer=initializer)
            # Data for the new model
            d = wave.forward(vp=vloc)[0]
            # First order error Phi(m0+dm) - Phi(m0)
            F_i = .5*linalg.norm((d.data - rec.data).reshape(-1))**2
            error1[i] = np.absolute(F_i - F0)
            # Second order term r Phi(m0+dm) - Phi(m0) - <J(m0)^T \delta d, dm>
            error2[i] = np.absolute(F_i - F0 - H[i] * G)

        # Test slope of the  tests
        p1 = np.polyfit(np.log10(H), np.log10(error1), 1)
        p2 = np.polyfit(np.log10(H), np.log10(error2), 1)
        info('1st order error, Phi(m0+dm)-Phi(m0): %s' % (p1))
        info(r'2nd order error, Phi(m0+dm)-Phi(m0) - <J(m0)^T \delta d, dm>: %s' % (p2))
        assert np.isclose(p1[0], 1.0, rtol=0.1)
        assert np.isclose(p2[0], 2.0, rtol=0.1)
        print("Working")

if __name__ == "__main__":
    test_gradientFWI(kernel='OT2', shape=(50, 60), ckp=False, setup_func=iso_setup, time_order=2, space_order=4, dtype=np.float32)

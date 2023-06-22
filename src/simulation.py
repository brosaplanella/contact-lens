#
# Simulation class
#
import pickle
import pybamm
import numpy as np
import copy
import warnings
import sys
from functools import lru_cache


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # pragma: no cover
            # Jupyter notebook or qtconsole
            cfg = get_ipython().config
            nb = len(cfg["InteractiveShell"].keys()) == 0
            return nb
        elif shell == "TerminalInteractiveShell":  # pragma: no cover
            return False  # Terminal running IPython
        elif shell == "Shell":  # pragma: no cover
            return True  # Google Colab notebook
        else:  # pragma: no cover
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class Simulation:
    """A Simulation class for easy building and running of PyBaMM simulations.

    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        The model to be simulated
    experiment : :class:`pybamm.Experiment` (optional)
        The experimental conditions under which to solve the model
    geometry: :class:`pybamm.Geometry` (optional)
        The geometry upon which to solve the model
    parameter_values: :class:`pybamm.ParameterValues` (optional)
        Parameters and their corresponding numerical values.
    submesh_types: dict (optional)
        A dictionary of the types of submesh to use on each subdomain
    var_pts: dict (optional)
        A dictionary of the number of points used by each spatial variable
    spatial_methods: dict (optional)
        A dictionary of the types of spatial method to use on each
        domain (e.g. pybamm.FiniteVolume)
    solver: :class:`pybamm.BaseSolver` (optional)
        The solver to use to solve the model.
    output_variables: list (optional)
        A list of variables to plot automatically
    C_rate: float (optional)
        The C-rate at which you would like to run a constant current (dis)charge.
    """

    def __init__(
        self,
        model,
        geometry=None,
        parameter_values=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        output_variables=None,
    ):
        self.parameter_values = parameter_values or model.default_parameter_values
        self._unprocessed_parameter_values = self.parameter_values

        self._unprocessed_model = model
        self.model = model

        self.geometry = geometry or self.model.default_geometry
        self.submesh_types = submesh_types or self.model.default_submesh_types
        self.var_pts = var_pts or self.model.default_var_pts
        self.spatial_methods = spatial_methods or self.model.default_spatial_methods
        self.solver = solver or self.model.default_solver
        self.output_variables = output_variables

        # Initialize empty built states
        self._model_with_set_params = None
        self._built_model = None
        self._mesh = None
        self._disc = None
        self._solution = None
        self.quick_plot = None

        # ignore runtime warnings in notebooks
        if is_notebook():  # pragma: no cover
            import warnings

            warnings.filterwarnings("ignore")

    def set_parameters(self):
        """
        A method to set the parameters in the model and the associated geometry.
        """

        if self.model_with_set_params:
            return

        if self._parameter_values._dict_items == {}:
            # Don't process if parameter values is empty
            self._model_with_set_params = self._unprocessed_model
        else:
            self._model_with_set_params = self._parameter_values.process_model(
                self._unprocessed_model, inplace=False
            )
            self._parameter_values.process_geometry(self.geometry)
        self.model = self._model_with_set_params

    def build(self, check_model=True):
        """
        A method to build the model into a system of matrices and vectors suitable for
        performing numerical computations. If the model has already been built or
        solved then this function will have no effect.
        This method will automatically set the parameters
        if they have not already been set.

        Parameters
        ----------
        check_model : bool, optional
            If True, model checks are performed after discretisation (see
            :meth:`pybamm.Discretisation.process_model`). Default is True.
        """

        if self.built_model:
            return
        elif self.model.is_discretised:
            self._model_with_set_params = self.model
            self._built_model = self.model
        else:
            self.set_parameters()
            self._mesh = pybamm.Mesh(self._geometry, self._submesh_types, self._var_pts)
            self._disc = pybamm.Discretisation(self._mesh, self._spatial_methods)
            self._built_model = self._disc.process_model(
                self._model_with_set_params, inplace=False, check_model=check_model
            )
            # rebuilt model so clear solver setup
            self._solver._model_set_up = {}

    def solve(
        self,
        t_eval,
        solver=None,
        check_model=True,
        callbacks=None,
        **kwargs,
    ):
        """
        A method to solve the model. This method will automatically build
        and set the model parameters if not already done so.

        Parameters
        ----------
        t_eval : numeric type
            The times (in seconds) at which to compute the solution. Can be
            provided as an array of times at which to return the solution, or as a
            list `[t0, tf]` where `t0` is the initial time and `tf` is the final time.
            If provided as a list the solution is returned at 100 points within the
            interval `[t0, tf]`.
        solver : :class:`pybamm.BaseSolver`, optional
            The solver to use to solve the model. If None, Simulation.solver is used
        check_model : bool, optional
            If True, model checks are performed after discretisation (see
            :meth:`pybamm.Discretisation.process_model`). Default is True.
        callbacks : list of callbacks, optional
            A list of callbacks to be called at each time step. Each callback must
            implement all the methods defined in :class:`pybamm.callbacks.BaseCallback`.
        **kwargs
            Additional key-word arguments passed to `solver.solve`.
            See :meth:`pybamm.BaseSolver.solve`.
        """
        # Setup
        if solver is None:
            solver = self.solver

        callbacks = pybamm.callbacks.setup_callbacks(callbacks)

        self.build(check_model=check_model)
        self._solution = solver.solve(self.built_model, t_eval, **kwargs)

        return self.solution

    def step(
        self, dt, solver=None, npts=2, save=True, starting_solution=None, **kwargs
    ):
        """
        A method to step the model forward one timestep. This method will
        automatically build and set the model parameters if not already done so.

        Parameters
        ----------
        dt : numeric type
            The timestep over which to step the solution
        solver : :class:`pybamm.BaseSolver`
            The solver to use to solve the model.
        npts : int, optional
            The number of points at which the solution will be returned during
            the step dt. Default is 2 (returns the solution at t0 and t0 + dt).
        save : bool
            Turn on to store the solution of all previous timesteps
        starting_solution : :class:`pybamm.Solution`
            The solution to start stepping from. If None (default), then self._solution
            is used
        **kwargs
            Additional key-word arguments passed to `solver.solve`.
            See :meth:`pybamm.BaseSolver.step`.
        """
        self.build()

        if solver is None:
            solver = self.solver

        if starting_solution is None:
            starting_solution = self._solution

        self._solution = solver.step(
            starting_solution, self.built_model, dt, npts=npts, save=save, **kwargs
        )

        return self.solution

    def plot(self, output_variables=None, **kwargs):
        """
        A method to quickly plot the outputs of the simulation. Creates a
        :class:`pybamm.QuickPlot` object (with keyword arguments 'kwargs') and
        then calls :meth:`pybamm.QuickPlot.dynamic_plot`.

        Parameters
        ----------
        output_variables: list, optional
            A list of the variables to plot.
        **kwargs
            Additional keyword arguments passed to
            :meth:`pybamm.QuickPlot.dynamic_plot`.
            For a list of all possible keyword arguments see :class:`pybamm.QuickPlot`.
        """

        if self._solution is None:
            raise ValueError(
                "Model has not been solved, please solve the model before plotting."
            )

        if output_variables is None:
            output_variables = self.output_variables

        self.quick_plot = pybamm.dynamic_plot(
            self._solution, output_variables=output_variables, **kwargs
        )

        return self.quick_plot

    def create_gif(self, number_of_images=80, duration=0.1, output_filename="plot.gif"):
        """
        Generates x plots over a time span of t_eval and compiles them to create
        a GIF. For more information see :meth:`pybamm.QuickPlot.create_gif`

        Parameters
        ----------
        number_of_images : int (optional)
            Number of images/plots to be compiled for a GIF.
        duration : float (optional)
            Duration of visibility of a single image/plot in the created GIF.
        output_filename : str (optional)
            Name of the generated GIF file.

        """
        if self.quick_plot is None:
            self.quick_plot = pybamm.QuickPlot(self._solution)

        self.quick_plot.create_gif(
            number_of_images=number_of_images,
            duration=duration,
            output_filename=output_filename,
        )

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = copy.copy(model)

    @property
    def model_with_set_params(self):
        return self._model_with_set_params

    @property
    def built_model(self):
        return self._built_model

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        self._geometry = geometry.copy()

    @property
    def parameter_values(self):
        return self._parameter_values

    @parameter_values.setter
    def parameter_values(self, parameter_values):
        self._parameter_values = parameter_values.copy()

    @property
    def submesh_types(self):
        return self._submesh_types

    @submesh_types.setter
    def submesh_types(self, submesh_types):
        self._submesh_types = submesh_types.copy()

    @property
    def mesh(self):
        return self._mesh

    @property
    def var_pts(self):
        return self._var_pts

    @var_pts.setter
    def var_pts(self, var_pts):
        self._var_pts = var_pts.copy()

    @property
    def spatial_methods(self):
        return self._spatial_methods

    @spatial_methods.setter
    def spatial_methods(self, spatial_methods):
        self._spatial_methods = spatial_methods.copy()

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver.copy()

    @property
    def output_variables(self):
        return self._output_variables

    @output_variables.setter
    def output_variables(self, output_variables):
        self._output_variables = copy.copy(output_variables)

    @property
    def solution(self):
        return self._solution

    def save(self, filename):
        """Save simulation using pickle"""
        if self.model.convert_to_format == "python":
            # We currently cannot save models in the 'python' format
            raise NotImplementedError(
                """
                Cannot save simulation if model format is python.
                Set model.convert_to_format = 'casadi' instead.
                """
            )
        # Clear solver problem (not pickle-able, will automatically be recomputed)
        if (
            isinstance(self._solver, pybamm.CasadiSolver)
            and self._solver.integrator_specs != {}
        ):
            self._solver.integrator_specs = {}
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def load_sim(filename):
    """Load a saved simulation"""
    return pybamm.load(filename)

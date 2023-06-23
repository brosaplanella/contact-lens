#
# Contact lens curing temperature dependent model
#
import pybamm


class ContactLens(pybamm.models.base_model.BaseModel):
    def __init__(self, name="contact lens"):
        super().__init__(name=name)

        ######################
        # Parameters
        ######################
        rho = pybamm.Parameter("Density [kg.m-3]")
        c_p = pybamm.Parameter("Specific heat capacity [J.kg-1.K-1]")
        k_T = pybamm.Parameter("Thermal conductivity [W.m-1.K-1]")
        beta = pybamm.Parameter("Enthalpy of polimerisation [J.m-3]")
        I0 = pybamm.Parameter("UV boundary intensity [W.m-2]")
        T0 = pybamm.Parameter("Initial temperature [K]")
        T_amb = pybamm.Parameter("Ambient temperature [K]")
        h = pybamm.Parameter("Heat transfer coefficient [W.m-2.K-1]")
        L = pybamm.Parameter("Lens thickness [m]")

        def R_p(alpha, I, T):
            return pybamm.FunctionParameter(
                "Rate of polimerisation [s-1]",
                {
                    "Degree of polimerisation": alpha,
                    "UV intensity [W.m-2]": I,
                    "Temperature [K]": T,
                },
            )

        def k_I(alpha, T):
            return pybamm.FunctionParameter(
                "Radiation absorbance [m-1]",
                {
                    "Degree of polimerisation": alpha,
                    "Temperature [K]": T,
                },
            )

        ######################
        # Variables
        ######################
        domains = {"primary": "lens"}
        T = pybamm.Variable("Temperature [K]", domains=domains)
        I = pybamm.Variable("UV intensity [W.m-2]", domains=domains)
        alpha = pybamm.Variable("Degree of polimerisation", domains=domains)
        self.z = pybamm.SpatialVariable("z", domain="lens", coord_sys="cartesian")

        ######################
        # Governing equations
        ######################
        dTdt = (
            pybamm.div(k_T * pybamm.grad(T))
            + k_I(alpha, T) * I
            + beta * R_p(alpha, I, T)
        ) / (rho * c_p)
        dalphadt = R_p(alpha, I, T)
        v = pybamm.PrimaryBroadcastToEdges(1, "lens")
        I_eq = pybamm.div(pybamm.upwind(I) * v) + k_I(alpha, T) * I
        self.rhs = {T: dTdt, alpha: dalphadt}
        self.algebraic = {I: I_eq}

        lbc = h * (pybamm.BoundaryValue(T, "left") - T_amb) / k_T
        rbc = -h * (pybamm.BoundaryValue(T, "right") - T_amb) / k_T
        self.boundary_conditions = {
            T: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")},
            I: {"left": (I0, "Dirichlet")},
        }

        alpha0 = pybamm.Scalar(0)
        self.initial_conditions = {T: T0, alpha: alpha0, I: I0}

        T_av = pybamm.Integral(T, self.z) / L
        alpha_av = pybamm.Integral(alpha, self.z) / L
        R_p_av = pybamm.Integral(R_p(alpha, I, T), self.z) / L
        alpha_bottom = pybamm.BoundaryValue(alpha, "right")
        alpha_top = pybamm.BoundaryValue(alpha, "left")

        ######################
        # (Some) variables
        ######################
        self.variables = {
            "Temperature [K]": T,
            "Averaged temperature [K]": T_av,
            "Degree of polimerisation": alpha,
            "Top boundary degree of polimerisation": alpha_top,
            "Bottom boundary degree of polimerisation": alpha_top,
            "UV intensity [W.m-2]": I,
            "Averaged degree of polimerisation": alpha_av,
            "Rate of polimerisation [s-1]": R_p(alpha, I, T),
            "Averaged rate of polimerisation [s-1]": R_p_av,
            "Time [s]": pybamm.t,
            "Time [min]": pybamm.t / 60,
            "z [m]": self.z,
            "z [um]": self.z * 1e6,
        }

    @property
    def default_geometry(self):
        L = pybamm.Parameter("Lens thickness [m]")
        return pybamm.Geometry({"lens": {self.z: {"min": pybamm.Scalar(0), "max": L}}})

    @property
    def default_submesh_types(self):
        return {"lens": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)}

    @property
    def default_var_pts(self):
        return {self.z: 50}

    @property
    def default_spatial_methods(self):
        return {"lens": pybamm.FiniteVolume()}

    @property
    def default_solver(self):
        return pybamm.IDAKLUSolver()
        # return pybamm.CasadiSolver("fast")

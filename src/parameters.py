#
# Parameter values for the polymerisation model
#
import pybamm
import numpy as np


def R_p_Gao(alpha, I, T):
    R = 8.314
    E_R_p = pybamm.Parameter("Activation energy polimerisation [J.mol-1]")
    C0 = pybamm.Parameter("Reaction rate [m.s-1.W.-0.5]")
    return C0 * pybamm.exp(-E_R_p / (R * T)) * (1 - alpha) * I ** (1 / 2)


def get_parameters(parameter_set="Gao2018"):
    """
    Returns the parameters for the polymerisation model

    Parameters
    ----------
    parameter_set : str, optional
        The name of the parameter set to use (default is "Gao2018")

    Returns
    -------
    parameters : parameter_values.ParameterValues
        The parameters for the polymerisation model
    """
    if parameter_set == "Gao2018":
        parameter_values = {
            "Density [kg.m-3]": 1128,
            "Specific heat capacity [J.kg-1.K-1]": 1190,
            "Thermal conductivity [W.m-1.K-1]": 0.2,
            "Radiation absorbance [m-1]": 1 / 77.55e-6,
            "Enthalpy of polimerisation [J.mol-1]": 79950,
            "Initial monomer concentration [mol.m-3]": 8200,
            "UV boundary intensity [W.m-2]": 80,
            "Initial temperature [K]": 353.15,
            "Ambient temperature [K]": 353.15,
            "Heat transfer coefficient [W.m-2.K-1]": 10,
            "Lens thickness [m]": 200e-6,
            "Rate of polimerisation [s-1]": R_p_Gao,
            "Activation energy polimerisation [J.mol-1]": 17792,
            "Reaction rate [m.s-1.W.-0.5]": 1.304,
        }
    elif parameter_set == "Gao2018 isothermal":
        parameter_values = {
            "Density [kg.m-3]": 1128,
            "Specific heat capacity [J.kg-1.K-1]": 1190,
            "Thermal conductivity [W.m-1.K-1]": 0.2,
            "Radiation absorbance [m-1]": 1 / 77.55e-6,
            "Enthalpy of polimerisation [J.mol-1]": 79950,
            "Initial monomer concentration [mol.m-3]": 8200,
            "UV boundary intensity [W.m-2]": 80,
            "Initial temperature [K]": 353.15,
            "Ambient temperature [K]": 353.15,
            "Heat transfer coefficient [W.m-2.K-1]": 10,
            "Lens thickness [m]": 200e-6,
            "Rate of polimerisation [s-1]": R_p_Gao,
            "Activation energy polimerisation [J.mol-1]": 0,
            "Reaction rate [m.s-1.W.-0.5]": 1.304 * np.exp(-17792 / (8.314 * 353.15)),
        }
    else:
        raise ValueError("Parameter set '{}' not recognised".format(parameter_set))

    return pybamm.ParameterValues(parameter_values)

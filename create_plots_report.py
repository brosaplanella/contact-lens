import pybamm
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from src import ContactLens, Simulation, get_parameters

# Assemble and solve the model
model = ContactLens()

param = get_parameters()
param2 = get_parameters("Gao2018 isothermal")

parameters = [param2, param]

solutions = []
for param in parameters:
    sim = Simulation(model, parameter_values=param)
    t = np.linspace(0, 400, 1000)
    solution = sim.solve(t)
    solutions.append(solution)

# Generate plot for averaged quantities
plot = pybamm.QuickPlot(
    solutions,
    output_variables=[
        "Averaged temperature [K]",
        "Averaged degree of polimerisation",
        "Averaged rate of polimerisation [s-1]"
    ],
    labels=[
        "temperature dependent",
        "isothermal",
    ],
)

plot.plot(0)

plot.fig.savefig(join("fig", "averaged_variables.png"), dpi=300)

# Generate plot for spatially distributed quantities
from  matplotlib.gridspec import GridSpec
gs = GridSpec(3, 3, width_ratios=[1, 1, 0.05])
fig = plt.figure(figsize=(15, 12))
variables = ["Temperature [K]", "Degree of polimerisation", "Rate of polimerisation [s-1]"]

limits = {}

for var in variables:
    vmax = max([solution[var].entries.max() for solution in solutions])
    vmin = min([solution[var].entries.min() for solution in solutions])
    limits[var] = (vmin, vmax)

for j, var in enumerate(variables):
    for i, solution in enumerate(solutions):
        t = solution["Time [s]"].entries
        z = solution["z [um]"].entries[:, 0]
        data = solution[var].entries
        ax = fig.add_subplot(gs[j, i])
        vmin, vmax = limits[var]
        cs = ax.contourf(t, z, data, vmin=vmin, vmax=vmax, cmap="viridis")
        

    cbar_ax = fig.add_subplot(gs[j, 2])
    cbar = fig.colorbar(cs, cax=cbar_ax)
    cbar.ax.set_ylabel(var)

fig.axes[0].set_title("Isothermal")
fig.axes[1].set_title("Temperature dependent")
for i in [0, 3, 6]:
    fig.axes[i].set_ylabel("Thickness [$\mu$m]")
for i in [6, 7]:
    fig.axes[i].set_xlabel("Time [min]")

fig.tight_layout()

fig.savefig(join("fig", "spatially_distributed.png"), dpi=300)
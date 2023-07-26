import pybamm
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from src import ContactLens, Simulation, get_parameters

model = ContactLens()

param = get_parameters()
param2 = get_parameters("Gao2018 isothermal")

parameters = [param, param2]

solutions = []
for param in parameters:
    sim = Simulation(model, parameter_values=param)
    t = np.linspace(0, 400, 1000)
    solution = sim.solve(t)
    solutions.append(solution)

plot = pybamm.QuickPlot(
    solutions,
    output_variables=[
        "Temperature [K]",
        "Degree of polimerisation",
        "Rate of polimerisation [s-1]",
        "Averaged temperature [K]",
        "Averaged degree of polimerisation",
        "Averaged rate of polimerisation [s-1]"
        # "UV intensity [W.m-2]",
    ],
    labels=[
        "temperature dependent",
        "isothermal",
    ],
)

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import gc

# time stamps at which the images/plots will be created
time_array = np.linspace(0, solutions[0]["Time [s]"].entries[-1], num=80)
images = []

# create images/plots
for val in time_array:
    print(val)
    plot.plot(val)
    images.append("plot" + str(val) + ".png")
    plot.fig.savefig("plot" + str(val) + ".png", dpi=300)
    plt.close()
    gc.collect()

images = images + images[::-1]

# compile the images/plots to create a GIF
with imageio.get_writer(join("fig", "my_gif.gif"), mode="I", duration=10) as writer:
    for image in images:
        print(image)
        writer.append_data(imageio.imread(image))
        gc.collect()

print("GIF created!")
# remove the generated images
for image in images:
    os.remove(image)

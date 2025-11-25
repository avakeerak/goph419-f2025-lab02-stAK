import numpy as np
import matplotlib.pyplot as plt

from linalg_interp import spline_function

# load data for splines
air_file = "air_density_vs_temp_eng_toolbox.txt"
water_file   = "water_density_vs_temp_usgs.txt"

water_data = np.loadtxt(water_file)
air_data   = np.loadtxt(air_file)

Tw, rho_w = water_data[:, 0], water_data[:, 1]
Ta, rho_a = air_data[:, 0], air_data[:, 1]



orders = [1, 2, 3]

# compute splines
water_splines = {order: spline_function(Tw, rho_w, order) for order in orders}
air_splines   = {order: spline_function(Ta, rho_a, order) for order in orders}


# Evaluate each spline at 100 temperatures across the domain
Tw_eval = np.linspace(Tw.min(), Tw.max(), 100)
Ta_eval = np.linspace(Ta.min(), Ta.max(), 100)

water_interp = {order: water_splines[order](Tw_eval) for order in orders}
air_interp   = {order: air_splines[order](Ta_eval) for order in orders}


# plot the resulting spline functions
fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

orders = [1, 2, 3]

for i, order in enumerate(orders):

    # plot splines for water density
    ax = axes[i, 0]
    ax.scatter(Tw, rho_w, color="black", s=30, label="Data")
    ax.plot(Tw_eval, water_interp[order], lw=2, label=f"Order {order}")
    ax.set_title(f"Water Density (Order {order})", fontsize=12)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Density (g/cm³)")
    ax.grid(True)
    ax.legend(loc="best")

    # plot splines for air density
    ax = axes[i, 1]
    ax.scatter(Ta, rho_a, color="black", s=30, label="Data")
    ax.plot(Ta_eval, air_interp[order], lw=2, label=f"Order {order}")
    ax.set_title(f"Air Density (Order {order})", fontsize=12)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Density (kg/m³)")
    ax.grid(True)
    ax.legend(loc="best")

# save plots as a single file 
plt.savefig("spline_interpolations.png", dpi=300)
plt.show()
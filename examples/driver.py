import numpy as np
import matplotlib.pyplot as plt

from linalg_interp import spline_function


air_file = "air_density_vs_temp_eng_toolbox.txt"
water_file   = "water_density_vs_temp_usgs.txt"

water_data = np.loadtxt(water_file)
air_data   = np.loadtxt(air_file)

Tw, rho_w = water_data[:, 0], water_data[:, 1]
Ta, rho_a = air_data[:, 0], air_data[:, 1]



orders = [1, 2, 3]

water_splines = {order: spline_function(Tw, rho_w, order) for order in orders}
air_splines   = {order: spline_function(Ta, rho_a, order) for order in orders}



Tw_eval = np.linspace(Tw.min(), Tw.max(), 100)
Ta_eval = np.linspace(Ta.min(), Ta.max(), 100)

water_interp = {order: water_splines[order](Tw_eval) for order in orders}
air_interp   = {order: air_splines[order](Ta_eval) for order in orders}



fig, axes = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle("Spline Interpolation of Water and Air Density Data", fontsize=16)

for i, order in enumerate(orders):

    # ---- Water (left column) ----
    ax = axes[i, 0]
    ax.scatter(Tw, rho_w, color="black", s=30, label="Data")
    ax.plot(Tw_eval, water_interp[order], lw=2, label=f"Order {order}")
    ax.set_title(f"Water Density – Order {order}")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend()

    # ---- Air (right column) ----
    ax = axes[i, 1]
    ax.scatter(Ta, rho_a, color="black", s=30, label="Data")
    ax.plot(Ta_eval, air_interp[order], lw=2, label=f"Order {order}")
    ax.set_title(f"Air Density – Order {order}")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend()

# ==========================================================
# 5. Save figure to the same directory
# ==========================================================

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("spline_interpolations.png", dpi=300)

print("Figure saved as spline_interpolations.png")
plt.show()

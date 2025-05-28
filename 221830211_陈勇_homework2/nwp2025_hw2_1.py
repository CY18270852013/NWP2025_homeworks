# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 02:32:02 2025

@author: Chen Yong
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 6371  # Earth's radius in km
Omega = 7.2921e-5  # Earth's angular velocity in rad/s
phi0 = np.radians(60)  # Reference latitude in radians (60°N)

# Grid parameters
d = 500  # Grid spacing in km
In = -4  # I-coordinate of point P
Jn = 8   # J-coordinate of point P

# Calculate l_e
l_e = R * (1 + np.sin(phi0))

# Calculate l for point P
l = np.sqrt((In * d)**2 + (Jn * d)**2)

# Calculate the latitude phi of point P
numerator = 1 - (l / l_e)**2
denominator = 1 + (l / l_e)**2
phi = np.arcsin(numerator / denominator)

# Map magnification factor
m = (1 + np.sin(phi0)) / (1 + np.sin(phi))

# Coriolis parameter
f = 2 * Omega * np.sin(phi)

# Calculate l for grid center (fixed at (0, 4))
In_center = 0  # I-coordinate of grid center
Jn_center = 5  # J-coordinate of grid center
l_center = np.sqrt((In_center * d)**2 + (Jn_center * d)**2)

# Calculate latitude of grid center
numerator_center = 1 - (l_center / l_e)**2
denominator_center = 1 + (l_center / l_e)**2
phi_center = np.arcsin(numerator_center / denominator_center)

# Convert latitudes to degrees
phi_deg = np.degrees(phi)
phi_center_deg = np.degrees(phi_center)

# Output results
print(f"Map magnification factor: {m}")
print(f"Coriolis parameter: {f} rad/s")
print(f"Latitude of grid center: {phi_center_deg} degrees")

# Plotting the grid
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.grid(True)

# Plot grid lines
for i in range(-10, 11):
    ax.axhline(i * d, color='blue', linestyle='--', linewidth=0.5)
    ax.axvline(i * d, color='blue', linestyle='--', linewidth=0.5)

# Plot point P
ax.plot(In * d, Jn * d, 'ro', markersize=8)  # Point P
ax.text(In * d, Jn * d, ' P', color='red', fontsize=12, ha='left', va='bottom')

# Plot grid center
ax.plot(In_center * d, Jn_center * d, 'go', markersize=8)  # Grid center
ax.text(In_center * d, Jn_center * d, ' Center', color='green', fontsize=12, ha='left', va='bottom')

# Set plot title and labels
plt.title("Polar Stereographic Projection Grid", fontsize=14)
plt.xlabel("X (km)", fontsize=12)
plt.ylabel("Y (km)", fontsize=12)

# Show plot
plt.show()
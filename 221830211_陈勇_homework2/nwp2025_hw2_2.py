# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 02:29:49 2025

@author: Chen Yong
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 6371  # Earth's radius in km
Omega = 7.2921e-5  # Earth's angular velocity in rad/s
phi1 = np.radians(30)  # First standard parallel (30°N)
phi2 = np.radians(60)  # Second standard parallel (60°N)

# Grid parameters
d = 300  # Grid spacing in km
In = 5   # I-coordinate of point P
Jn = 15  # J-coordinate of point P

# Calculate projection constant n
numerator_n = np.log(np.cos(phi1) / np.cos(phi2))
denominator_n = np.log(np.tan(np.pi/4 + phi2/2) / np.tan(np.pi/4 + phi1/2))
n = numerator_n / denominator_n

# Calculate l for point P
l = np.sqrt((In * d)**2 + (Jn * d)**2)

# Calculate the latitude phi of point P
phi = np.arcsin(1 - (l / (R * n))**2)

# Map magnification factor
m = n / np.cos(phi)

# Coriolis parameter
f = 2 * Omega * np.sin(phi)

# Calculate l for grid center (fixed at (0, 0))
In_center = 0  # I-coordinate of grid center
Jn_center = 11  # J-coordinate of grid center
l_center = np.sqrt((In_center * d)**2 + (Jn_center * d)**2)

# Calculate latitude of grid center
phi_center = np.arcsin(1 - (l_center / (R * n))**2)

# Convert latitudes to degrees
phi_deg = np.degrees(phi)
phi_center_deg = np.degrees(phi_center)

# Output results
print(f"Projection constant n: {n}")
print(f"Map magnification factor: {m}")
print(f"Coriolis parameter: {f} rad/s")
print(f"Latitude of grid center: {phi_center_deg} degrees")

# Plotting the grid
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.grid(True)

# Plot grid lines
for i in range(-20, 21):
    ax.axhline(i * d, color='blue', linestyle='--', linewidth=0.5)
    ax.axvline(i * d, color='blue', linestyle='--', linewidth=0.5)

# Plot point P
ax.plot(In * d, Jn * d, 'ro', markersize=8)  # Point P
ax.text(In * d, Jn * d, ' P', color='red', fontsize=12, ha='left', va='bottom')

# Plot grid center
ax.plot(In_center * d, Jn_center * d, 'go', markersize=8)  # Grid center
ax.text(In_center * d, Jn_center * d, ' Center', color='green', fontsize=12, ha='left', va='bottom')

# Set plot title and labels
plt.title("Lambert Conformal Conic Projection Grid", fontsize=14)
plt.xlabel("X (km)", fontsize=12)
plt.ylabel("Y (km)", fontsize=12)

# Show plot
plt.show()
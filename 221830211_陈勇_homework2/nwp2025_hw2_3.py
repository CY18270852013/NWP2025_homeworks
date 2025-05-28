# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 02:36:27 2025

@author: Chen Yong
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 6371  # Earth's radius in km
Omega = 7.2921e-5  # Earth's angular velocity in rad/s

# Grid parameters
d = 200  # Grid spacing in km
je = 3   # J-coordinate of point P

# Calculate the latitude phi of point P
phi = np.radians(22.5)  # Since the projection is cut at 22.5°N and 22.5°S

# Map magnification factor
m = 1 / np.cos(phi)

# Coriolis parameter
f = 2 * Omega * np.sin(phi)

# Output results
print(f"Map magnification factor: {m}")
print(f"Coriolis parameter: {f} rad/s")

# Plotting the grid
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.grid(True)

# Plot grid lines
for i in range(-10, 11):
    ax.axhline(i * d, color='blue', linestyle='--', linewidth=0.5)
    ax.axvline(i * d, color='blue', linestyle='--', linewidth=0.5)

# Plot point P
ax.plot(0, je * d, 'ro', markersize=8)  # Point P
ax.text(0, je * d, ' P', color='red', fontsize=12, ha='left', va='bottom')

# Set plot title and labels
plt.title("Mercator Projection Grid", fontsize=14)
plt.xlabel("X (km)", fontsize=12)
plt.ylabel("Y (km)", fontsize=12)

# Show plot
plt.show()
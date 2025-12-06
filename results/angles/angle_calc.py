import pandas as pd
import numpy as np

# Read the CSV, specifying the separator is ';'
df = pd.read_csv("results/angles/angle_data.csv", sep=';')

# Extract columns
a = df["AC_norm"]
b = df["BC_norm"]
theta = df["angle_rad"]  # in radians

# Compute numerator and denominator
numerator = a * b * np.sin(theta)
denominator = 2 * (a**2 + b**2 - 2 * a * b * np.cos(theta))

# Compute scaled area
df["scaled_area"] = numerator / denominator

# Save updated file
df.to_csv("angle_data_with_scaled_area.csv", sep=';', index=False)

print("Done! New column 'scaled_area' added.")

import numpy as np
from pyDOE2 import lhs
import pandas as pd

# Define the number of factors and samples
factors = 10
samples = 980
x_scale = [-350,350] # X-position: 0 to 300, -350 to 350 for shell
y_scale = [-150,150] # Y-position: 0 to 100, -150 to 150 for shell
geo_scale = [0,700] # Geometrie vectors: 10 to 150, 0 to 700 for shell


def create_lhs(n_factors,n_samples):
    return lhs(n_factors, samples=n_samples, criterion='maximin')

if __name__ == "__main__":
    lhs_design = create_lhs(factors, samples)

    # Initialize an array to hold the scaled design
    lhs_scaled = np.zeros_like(lhs_design)

    # Scale Factor 1 (Y-position)
    lhs_scaled[:, 1] = lhs_design[:, 1] * (y_scale[1] - y_scale[0]) + y_scale[0]

    # Scale Factor 2 (X-position)
    lhs_scaled[:, 0] = lhs_design[:, 0] * (x_scale[1] - x_scale[0]) + x_scale[0]

    # Scale Factors 3 to 10 (Geometrie Vectors)
    lhs_scaled[:, 2:] = lhs_design[:, 2:] * (geo_scale[1] - geo_scale[0]) + geo_scale[0]

    columns = ['X_position', 'Y_position', 'Geom1', 'Geom2', 'Geom3', 'Geom4', 'Geom5', 'Geom6', 'Geom7', 'Geom8']
    lhs_df = pd.DataFrame(lhs_scaled, columns=columns)

    # Save the DataFrame to an Excel file
    lhs_df.to_excel('lhs_design_schale.xlsx', index=False)

    # Verify the scaling by printing the first few samples
    np.set_printoptions(precision=2)  # better readability
    print("First 5 samples of the scaled LHS design:")
    print(lhs_scaled[:5, :])

    # np.savetxt("lhs_design.csv", lhs_scaled, delimiter=",", header="Y_position,X_position,Geom1,Geom2,Geom3,Geom4,Geom5,Geom6,Geom7,Geom8", comments='')

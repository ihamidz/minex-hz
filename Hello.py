import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

# Step 3: Load your geochemical dataset
data = pd.read_csv('NDI2.csv')

# Step 4: Preprocess the data
# Assuming the first column is the sample id and the rest are elements
X = data.iloc[:, 1:].values  # Adjust the range to cover all 31 variables

# Step 5: Standardize the data
standardized_data = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Step 6: Define weight scores for each variable
weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5, 1.5, 1.5, 1.5,1.5, 1.5,1.5, 1.5,1.5]

# Step 7: Apply weights to the standardized variables
weighted_data = standardized_data * weights

# Step 8: Perform change point detection on weighted data with automatic breakpoint selection using BIC
model = "rbf"  # you can choose other models, e.g., "l2", "rbf", etc.
algo = rpt.Pelt(model=model).fit(weighted_data)
result = algo.predict(pen=1)  # adjust the penalty term as needed

# Step 9: Visualize the results with a legend
plt.figure(figsize=(10, 4))  # Adjust the figure size as needed

# Plot each standardized and weighted element separately and label with element names
for i in range(weighted_data.shape[1]):
    plt.plot(weighted_data[:, i], label=data.columns[i + 1])

plt.title('Multivariate Change Point Detection (Standardized and Weighted Data)')
plt.xlabel('Data Points')
plt.ylabel('Weighted Values')

# Mark the change points
for bkp in result:
    plt.axvline(bkp, linestyle='dashed', color='red')

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the figure as a PNG file
plt.savefig('standardized_weighted_change_point_detection_plot.png', bbox_inches='tight')

# Show the plot
plt.show()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

def main():
    st.title("Multivariate Change Point Detection App")

    # Step 1: Load your geochemical dataset
    data = pd.read_csv('NDI2.csv')

    # Step 2: Sidebar with user input
    st.sidebar.header("Parameters")
    pen_value = st.sidebar.slider("Penalty Term", min_value=1, max_value=100, value=50)

    # Step 3: Preprocess the data
    X = data.iloc[:, 1:].values

    # Step 4: Standardize the data
    standardized_data = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Step 5: Define weight scores for each variable
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]

    # Step 6: Apply weights to the standardized variables
    weighted_data = standardized_data * weights

    # Step 7: Perform change point detection on weighted data with automatic breakpoint selection using BIC
    model = "rbf"
    algo = rpt.Pelt(model=model).fit(weighted_data)
    result = algo.predict(pen=pen_value)

    # Step 8: Visualize the results
    plot_change_points(weighted_data, result, data.columns)

def plot_change_points(weighted_data, result, column_names):
    plt.figure(figsize=(10, 4))
    for i in range(weighted_data.shape[1]):
        plt.plot(weighted_data[:, i], label=column_names[i + 1])

    plt.title('Multivariate Change Point Detection (Standardized and Weighted Data)')
    plt.xlabel('Data Points')
    plt.ylabel('Weighted Values')

    for bkp in result:
        plt.axvline(bkp, linestyle='dashed', color='red')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot()

if __name__ == "__main__":
    main()

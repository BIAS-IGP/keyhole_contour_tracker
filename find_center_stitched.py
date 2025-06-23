import pandas as pd
import matplotlib.pyplot as plt

# Replace 'data.csv' with the path to your actual CSV file
csv_file = '11_depth_data.csv'
x = 75
# Read the CSV file
try:
    data = pd.read_csv(csv_file)

    # Check if expected columns exist
    if 'Length [mm]' not in data.columns or 'Keyhole depth [mm]' not in data.columns:
        raise ValueError("CSV must contain 'Length [mm]' and 'Keyhole depth [mm]' columns.")

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.scatter(data['Length [mm]'], data['Keyhole depth [mm]'], marker='o', linestyle='-', s = 1)
    plt.axvline(x)

    plt.title('Keyhole Depth vs. Length')
    plt.xlabel('Length [mm]')
    plt.ylabel('Keyhole depth [mm]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{csv_file}' was not found.")
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except Exception as e:
    print(f"An error occurred: {e}")

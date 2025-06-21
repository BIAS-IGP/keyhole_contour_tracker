import pandas as pd
import matplotlib.pyplot as plt
import os

# Lists of file stems
one_list = [
    f"{i}-1.png" if i != 15 else "X-1.png"
    for i in range(1, 21)
]
two_list = [
    f"{i}-2.png" if i != 15 else "X-2.png"
    for i in range(1, 21)
]

# Loop over the paired image names
for one_img, two_img in zip(one_list, two_list):
    # Extract base name (e.g., "1" from "1-1.png" or "X" from "X-1.png")
    base_name = one_img.split("-")[0]

    # Construct corresponding CSV filenames
    if base_name == "6" or base_name == "3":
        csv1, csv2 = f"processed/{base_name}-2_depth_data.csv", f"processed/{base_name}-1_depth_data.csv"
    else:
        csv1 = f"processed/{base_name}-1_depth_data.csv"
        csv2 = f"processed/{base_name}-2_depth_data.csv"
    output_csv = f"stitched/{base_name}_depth_data.csv"
    output_plot = f"stitched/{base_name}_depth_plot.png"

    # Check if both files exist
    if not os.path.exists(csv1) or not os.path.exists(csv2):
        print(f"Skipping {base_name}: missing CSV files.")
        continue

    # Load both CSVs
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Calculate scaling step and offset
    scaling_step = df1["Length [mm]"].iloc[1] - df1["Length [mm]"].iloc[0]
    last_length = df1["Length [mm]"].iloc[-1]
    df2["Length [mm]"] += last_length + scaling_step

    # Combine and save
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.to_csv(output_csv, index=False)

    # Plot and save
    plt.figure(figsize=(10, 5))
    plt.plot(combined_df["Length [mm]"], combined_df["Keyhole depth [mm]"], label="Keyhole Depth", color="blue")
    plt.xlabel("Length [mm]")
    plt.ylabel("Keyhole depth [mm]")
    plt.title(f"{base_name} Keyhole Depth over Length")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    print(f"Saved: {output_csv}, {output_plot}")

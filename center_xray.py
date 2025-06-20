import os
import pandas as pd

# === Parameters ===
root_dir = "xray/"
time_factor = 1 / 5000

# Provide one center index per subfolder (in order they appear in os.walk)
center_indices = [0, 4378, 4383, 4378, 
                4368, 4377, 4367, 
                4382, 4223, 
                0,0,0,0,0]

# === Traverse folders ===
folder_count = 0

for dirpath, dirnames, filenames in os.walk(root_dir):
    if dirpath == root_dir:
        continue  # Skip root, only process subfolders

    # Look for *_analysis_complete.csv and *_index.csv
    analysis_file = None
    index_file = None
    base_name = None

    for fname in filenames:
        if fname.endswith("_analysis_complete.csv"):
            analysis_file = fname
            base_name = fname.replace("_analysis_complete.csv", "")
        elif fname.endswith("_index.csv"):
            index_file = fname

    if analysis_file and index_file and base_name:
        if folder_count >= len(center_indices):
            print(f"⚠️ Not enough center indices for folder: {dirpath}. Skipping.")
            continue

        center_old_index = center_indices[folder_count]
        folder_count += 1

        try:
            print(f"\n📂 Processing: {base_name} in {dirpath}")
            print(f"   → Using center_old_index = {center_old_index}")

            # Load CSVs
            main_path = os.path.join(dirpath, analysis_file)
            index_path = os.path.join(dirpath, index_file)

            main_df = pd.read_csv(main_path, sep=';', skiprows=[1])
            main_df = main_df.rename(columns={"dex": "index"})

            index_df = pd.read_csv(index_path, sep=';')
            index_df.columns = ['old_name', 'new_name', 'old_index', 'new_index']

            # Map new_index → old_index
            new_to_old = dict(zip(index_df["new_index"], index_df["old_index"]))
            main_df["old_index"] = main_df["index"].map(new_to_old)

            main_df = main_df.dropna(subset=["old_index"])
            main_df["old_index"] = main_df["old_index"].astype(int)

            main_df["time"] = main_df["old_index"] * time_factor
            center_time = center_old_index * time_factor
            main_df["centered_time"] = main_df["time"] - center_time

            # Export result
            out_df = main_df[["centered_time", "depth"]].rename(columns={"centered_time": "time"})
            out_name = f"{base_name}_centered_depth.csv"
            out_path = os.path.join(dirpath, out_name)
            out_df.to_csv(out_path, index=False, sep=';')

            print(f"✅ Saved: {out_path}")

        except Exception as e:
            print(f"❌ Error processing {dirpath}: {e}")

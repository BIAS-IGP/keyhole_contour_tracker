import pandas as pd

"""
For Versuchsnummer 2 the values are reversed!!!
"""

def center_and_convert_to_time(csv_path, x_value, rate_mm_per_min, output_path=None):
    """
    Replace 'Length [mm]' column with 'Time [s]' by centering around x_value and converting using mm/min rate.

    Parameters:
    - csv_path (str): Path to the input CSV file.
    - x_value (float): Target x-value in mm to center around.
    - rate_mm_per_min (float): Speed in mm/min for conversion to time (output will be in seconds).
    - output_path (str, optional): Path to save the modified CSV. If None, file is not saved.

    Returns:
    - pd.DataFrame: Modified DataFrame with 'Time [s]' replacing 'Length [mm]'.
    """
    df = pd.read_csv(csv_path)
    
    if "Length [mm]" not in df.columns:
        raise ValueError("CSV file must contain a 'Length [mm]' column.")
    
    # Find closest value in 'Length [mm]'
    closest_value = df["Length [mm]"].iloc[(df["Length [mm]"] - x_value).abs().argmin()]
    
    # Center and convert to time in seconds
    df["Length [mm]"] = (df["Length [mm]"] - closest_value) / rate_mm_per_min * 60
    df["Length [mm]"] = df["Length [mm]"][::-1]
    # Rename column to 'Time [s]'
    df = df.rename(columns={"Length [mm]": "Time [s]"})
    
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df

    
x_vals = [
    77.5,
    7,
    75,
    78.5,
    78,
    77,
    79.5,
    76.5,
    78.5,
    80,
    75
]

speed = [3000,
3000,
3000,
3000,
3000,
3000,
3000,
3000,
3000,
3000,
3000,
]
for idx, x_val in enumerate(x_vals):

# Example usage:
# idx = 1
# x_val = 77


    df_centered = center_and_convert_to_time(f"{idx+1}_depth_data.csv", 
                                        x_value=x_val, 
                                        rate_mm_per_min = speed[idx],
                                        output_path=f"{idx+1}_depth_data_centered.csv"
                                        )
    if idx == 1:
        df_centered["Time [s]"] = -df_centered["Time [s]"]
        df_centered.to_csv(f"{idx+1}_depth_data_centered.csv", index=False)
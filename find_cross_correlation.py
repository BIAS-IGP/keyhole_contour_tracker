import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, correlation_lags
from scipy.interpolate import interp1d

idx = 2


df1 = pd.read_csv(f"xray/IGP-H-IFSW-{idx}/IGP-H-IFSW-{idx}_centered_depth.csv", delimiter=";")
df2 = pd.read_csv(f"OCT-Auswertungen/IGP-H-IFSW-{idx}/IGP-H-IFSW-{idx}_depth_analysis_centered.csv", delimiter=";")
df3 = pd.read_csv(f"stitched/{idx}_depth_data_centered.csv", delimiter=",")


xray_time = df1["time"].values
xray_depth = df1["depth"].values

oct_time = df2["time"].values
oct_depth = df2["OCT-depth"].values / 1000

stitched_time = df3["Time [s]"].values
stitched_depth = df3["Keyhole depth [mm]"].values

# Optional: interpolate all to same time base
common_time = np.linspace(max(xray_time.min(), oct_time.min(), stitched_time.min()),
                          min(xray_time.max(), oct_time.max(), stitched_time.max()), 1000)


xray_interp = interp1d(xray_time, xray_depth, bounds_error=False, fill_value=np.nan)
oct_interp = interp1d(oct_time, oct_depth, bounds_error=False, fill_value=np.nan)
stitched_interp = interp1d(stitched_time, stitched_depth, bounds_error=False, fill_value=np.nan)

xray_resampled = xray_interp(common_time)
oct_resampled = oct_interp(common_time)
stitched_resampled = stitched_interp(common_time)


valid = ~np.isnan(xray_resampled) & ~np.isnan(oct_resampled)
xray_resampled = xray_resampled[valid]
oct_resampled_crop = oct_resampled[valid]
common_time_crop = common_time[valid]


corr_xray = correlate(xray_resampled - np.mean(xray_resampled),
                      oct_resampled_crop - np.mean(oct_resampled_crop), mode="full")
lags_xray = correlation_lags(len(xray_resampled), len(oct_resampled_crop), mode="full")
lag_xray = lags_xray[np.argmax(corr_xray)]
shift_xray = (common_time_crop[1] - common_time_crop[0]) * lag_xray


valid = ~np.isnan(stitched_resampled) & ~np.isnan(oct_resampled)
stitched_resampled = stitched_resampled[valid]
oct_resampled_crop = oct_resampled[valid]
common_time_crop = common_time[valid]

corr_stitched = correlate(stitched_resampled - np.mean(stitched_resampled),
                          oct_resampled_crop - np.mean(oct_resampled_crop), mode="full")
lags_stitched = correlation_lags(len(stitched_resampled), len(oct_resampled_crop), mode="full")
lag_stitched = lags_stitched[np.argmax(corr_stitched)]
shift_stitched = (common_time_crop[1] - common_time_crop[0]) * lag_stitched

print(f"X-ray to OCT time shift: {shift_xray:.4f} s")
print(f"Stitched to OCT time shift: {shift_stitched:.4f} s")


xray_time_aligned = xray_time - shift_xray
stitched_time_aligned = stitched_time - shift_stitched

# Plotting
plt.figure(figsize=(6, 5))
plt.scatter(xray_time_aligned, xray_depth, label="X-Ray (aligned)", marker='o', s=1)
plt.scatter(oct_time, oct_depth, label="OCT", marker='s', s=1)
plt.scatter(stitched_time_aligned, stitched_depth, label="Längsschliff (aligned)", marker='^', s=1)

plt.xlabel("Time (s)")
plt.ylabel("Depth (mm)")
plt.title("Aligned Depth over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

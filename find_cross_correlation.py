import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, correlation_lags
from scipy.interpolate import interp1d

def add_units_to_ticks(ax, axis='x', unit='mm'):
    """
    Add units to the second-to-last tick on the given axis.

    Parameters:
      - ax: The axes object.
      - axis: 'x' or 'y', the axis to modify.
      - unit: The unit to append to the tick label (default 'mm').
    """
    if axis == 'x':
        old_xticks = [t for t in ax.get_xticks() if t >= ax.get_xlim()[
            0] and t <= ax.get_xlim()[1]]
        ax.set_xticks(old_xticks)
        old_xticklabels = ax.get_xticklabels()
        old_xticklabels[-2] = unit
        ax.set_xticklabels(old_xticklabels)

    elif axis == 'y':
        old_yticks = [t for t in ax.get_yticks() if t >= ax.get_ylim()[
            0] and t <= ax.get_ylim()[1]]
        ax.set_yticks(old_yticks)
        old_yticklabels = ax.get_yticklabels()
        old_yticklabels[-2] = unit
        ax.set_yticklabels(old_yticklabels)

for idx in range(0, 12):
    try:
        # Load data
        df1 = pd.read_csv(f"xray/IGP-H-IFSW-{idx}/IGP-H-IFSW-{idx}_centered_depth.csv", delimiter=";")
        df2 = pd.read_csv(f"OCT-Auswertungen/IGP-H-IFSW-{idx}/IGP-H-IFSW-{idx}_depth_analysis_centered.csv", delimiter=";")
        df3 = pd.read_csv(f"stitched/{idx}_depth_data_centered.csv", delimiter=",")

        xray_time = df1["time"].values
        xray_depth = df1["depth"].values

        oct_time = df2["time"].values
        oct_depth = df2["OCT-depth"].values / 1000  # Convert to mm

        stitched_time = df3["Time [s]"].values
        stitched_depth = df3["Keyhole depth [mm]"].values

        # Interpolate to common time base
        common_time = np.linspace(
            max(xray_time.min(), oct_time.min(), stitched_time.min()),
            min(xray_time.max(), oct_time.max(), stitched_time.max()),
            100000
        )

        xray_interp = interp1d(xray_time, xray_depth, bounds_error=False, fill_value=np.nan)
        oct_interp = interp1d(oct_time, oct_depth, bounds_error=False, fill_value=np.nan)
        stitched_interp = interp1d(stitched_time, stitched_depth, bounds_error=False, fill_value=np.nan)

        xray_resampled = xray_interp(common_time)
        oct_resampled = oct_interp(common_time)
        stitched_resampled = stitched_interp(common_time)

        # Align x-ray to OCT
        valid = ~np.isnan(xray_resampled) & ~np.isnan(oct_resampled)
        corr_xray = correlate(
            xray_resampled[valid] - np.mean(xray_resampled[valid]),
            oct_resampled[valid] - np.mean(oct_resampled[valid]),
            mode="full"
        )
        lags_xray = correlation_lags(len(valid.nonzero()[0]), len(valid.nonzero()[0]), mode="full")
        lag_xray = lags_xray[np.argmax(corr_xray)]
        dt = common_time[1] - common_time[0]
        shift_xray = dt * lag_xray

        # Align stitched to OCT
        valid = ~np.isnan(stitched_resampled) & ~np.isnan(oct_resampled)
        corr_stitched = correlate(
            stitched_resampled[valid] - np.mean(stitched_resampled[valid]),
            oct_resampled[valid] - np.mean(oct_resampled[valid]),
            mode="full"
        )
        lags_stitched = correlation_lags(len(valid.nonzero()[0]), len(valid.nonzero()[0]), mode="full")
        lag_stitched = lags_stitched[np.argmax(corr_stitched)]
        shift_stitched = dt * lag_stitched

        print(f"[{idx}] X-ray to OCT shift: {shift_xray:.4f}s | Stitched to OCT shift: {shift_stitched:.4f}s")

        # Apply alignment shifts
        xray_time_aligned = xray_time - shift_xray
        stitched_time_aligned = stitched_time - shift_stitched

        # Plot all in one figure
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.scatter(xray_time_aligned, xray_depth, s=1, label="Röntgen", color="blue")
        ax.scatter(oct_time, oct_depth, s=1, label="OCT", color="green")
        ax.scatter(stitched_time_aligned, stitched_depth, s=1, label="Längsschliff", color="red")

        ax.set_title(f"Tiefenvergleich – ID {idx}")
        ax.set_xlabel("Zeit")
        ax.set_ylabel("Tiefe")
        ax.grid(True)
        ax.legend()

        add_units_to_ticks(ax, axis='x', unit='s')
        add_units_to_ticks(ax, axis='y', unit='mm')

        plt.tight_layout()
        plt.savefig(f"Cross_correlated/{idx}.png")
        plt.show()

    except Exception as e:
        # print(f"[{idx}] Fehler: {e}")
        pass

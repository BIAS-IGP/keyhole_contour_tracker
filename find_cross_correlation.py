import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

def find_shift_with_uncertainty(corr, lags, dt, uncertainty_seconds):
    zero_idx = np.where(lags == 0)[0][0]
    max_error_lags = int(uncertainty_seconds / dt)
    start = max(0, zero_idx - max_error_lags)
    end = min(len(corr), zero_idx + max_error_lags + 1)

    corr_window = corr[start:end]
    lags_window = lags[start:end]

    best_lag = lags_window[np.argmax(corr_window)]
    return best_lag * dt

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
            min(xray_time.min(), oct_time.min(), stitched_time.min()),
            max(xray_time.max(), oct_time.max(), stitched_time.max()),
            100000
        )
        # common_time = oct_time
        
        xray_interp = interp1d(xray_time, xray_depth, bounds_error=False, fill_value=0)
        oct_interp = interp1d(oct_time, oct_depth, bounds_error=False, fill_value=0)
        stitched_interp = interp1d(stitched_time, stitched_depth, bounds_error=False, fill_value=0)

        xray_resampled = xray_interp(common_time)
        oct_resampled = oct_interp(common_time)
        stitched_resampled = stitched_interp(common_time)

        # Align x-ray to OCT
        valid = ~np.isnan(xray_resampled) & ~np.isnan(oct_resampled)
        corr_xray = correlate(
            (xray_resampled[valid] - np.mean(xray_resampled[valid]))/np.std(xray_resampled[valid]),
            (oct_resampled[valid] - np.mean(oct_resampled[valid]))/np.std(oct_resampled[valid]),
            mode="full"
        )
        lags_xray = correlation_lags(len(valid.nonzero()[0]), len(valid.nonzero()[0]), mode="full")
        lag_xray = lags_xray[np.argmax(corr_xray)]
        dt = common_time[1] - common_time[0]
        shift_xray = dt * lag_xray

        # Align stitched to OCT
        valid = ~np.isnan(stitched_resampled) & ~np.isnan(oct_resampled)
        corr_stitched = correlate(
            (stitched_resampled[valid] - np.mean(stitched_resampled[valid]))/np.std(stitched_resampled[valid]),
            (oct_resampled[valid] - np.mean(oct_resampled[valid]))/np.std(oct_resampled[valid]),
            mode="full"
        )
        lags_stitched = correlation_lags(len(valid.nonzero()[0]), len(valid.nonzero()[0]), mode="full")
        lag_stitched = lags_stitched[np.argmax(corr_stitched)]
        shift_stitched = dt * lag_stitched

        print(f"[{idx}] X-ray to OCT shift: {shift_xray:.4f}s | Stitched to OCT shift: {shift_stitched:.4f}s")

        # shift_xray = find_shift_with_uncertainty(corr_xray, lags_xray, dt, 0.15)
        # shift_stitched = find_shift_with_uncertainty(corr_stitched, lags_stitched, dt, 0.15)
        # Apply alignment shifts
        xray_time_aligned = xray_time - shift_xray
        stitched_time_aligned = stitched_time - shift_stitched

        # Create figure with GridSpec layout
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], hspace=0.3, wspace=0.3)
        
        # Top left: Correlation X-ray vs OCT
        ax_corr1 = fig.add_subplot(gs[0, 0])
        ax_corr1.plot(lags_xray * dt, corr_xray, color='blue')
        ax_corr1.axvline(shift_xray, color='black', linestyle='--', label=f'Shift = {shift_xray:.4f}s')
        ax_corr1.set_title('Kreuzkorrelation: Röntgen vs OCT')
        ax_corr1.set_xlabel("Lag (s)")
        ax_corr1.set_ylabel("Korrelationswert")
        ax_corr1.grid(True)
        ax_corr1.legend()
        
        # Top right: Correlation Stitched vs OCT
        ax_corr2 = fig.add_subplot(gs[0, 1])
        ax_corr2.plot(lags_stitched * dt, corr_stitched, color='red')
        ax_corr2.axvline(shift_stitched, color='black', linestyle='--', label=f'Shift = {shift_stitched:.4f}s')
        ax_corr2.set_title('Kreuzkorrelation: Längsschliff vs OCT')
        ax_corr2.set_xlabel("Lag (s)")
        ax_corr2.set_ylabel("Korrelationswert")
        ax_corr2.grid(True)
        ax_corr2.legend()
        
        # Bottom: Final combined depth comparison (spans both columns)
        ax_final = fig.add_subplot(gs[1, :])
        
        # Original (unshifted) data in faded colors
        # ax_final.scatter(xray_time, xray_depth, s=1, label="Röntgen (original)", color="blue", alpha=0.05)
        # ax_final.scatter(stitched_time, stitched_depth, s=1, label="Längsschliff (original)", color="red", alpha=0.05)
        
        # Shifted (aligned) data
        ax_final.scatter(xray_time_aligned, xray_depth, s=1, label="Röntgen (aligned)", color="blue")
        ax_final.scatter(oct_time, oct_depth, s=1, label="OCT", color="green")
        ax_final.scatter(stitched_time_aligned, stitched_depth, s=1, label="Längsschliff (aligned)", color="red")
        
        # Arrows indicating shifts

        y_arrow = np.nanmax([np.nanmax(xray_depth), np.nanmax(oct_depth), np.nanmax(stitched_depth)]) * 0.95
        ax_final.annotate("", xy=(xray_time_aligned[0], y_arrow), xytext=(xray_time[0], y_arrow), arrowprops=dict(arrowstyle="->", color="blue", linewidth=1))
        ax_final.annotate("", xy=(stitched_time_aligned[0], y_arrow * 0.9), xytext=(stitched_time[0], y_arrow * 0.9), arrowprops=dict(arrowstyle="->", color="red", linewidth=1))
        ax_final.text((xray_time_aligned[0] + xray_time[0]) / 2, y_arrow * 1.02, f"{shift_xray:.2f}s", ha='center')
        ax_final.text((stitched_time_aligned[0] + stitched_time[0]) / 2, y_arrow * 0.92, f"{shift_stitched:.2f}s", ha='center')
        
        ax_final.set_title(f"Tiefenvergleich – ID {idx}")
        ax_final.set_xlabel("Zeit [s]")
        ax_final.set_ylabel("Tiefe [mm]")
        ax_final.grid(True)
        ax_final.legend()
        
        add_units_to_ticks(ax_final, axis='x', unit='s')
        add_units_to_ticks(ax_final, axis='y', unit='mm')
        
        # Save and show
        plt.tight_layout()
        plt.savefig(f"Cross_correlated/{idx}_combined_plot.png")
        plt.show()


    except Exception as e:
        # print(f"[{idx}] Fehler: {e}")
        if idx==2:
            raise e
        pass

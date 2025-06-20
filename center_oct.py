import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel
)
from PyQt5.QtCore import Qt

def read_custom_hist_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the line index for #end_of_header
    for i, line in enumerate(lines):
        if line.strip() == "#end_of_header":
            header_index = i
            break
    else:
        raise ValueError("Missing #end_of_header in file")

    # Line after header: column names
    column_line = lines[header_index + 2].strip()
    column_names = column_line.split(";")

    # Units line (optional, we can ignore or capture separately)
    data_start_index = header_index + 3

    # Read the rest of the data
    from io import StringIO
    data_str = ''.join(lines[data_start_index:])
    df = pd.read_csv(StringIO(data_str), sep=';', names=column_names)
    print(df[1:])
    return df


class PlotWindow(QMainWindow):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.current_line = None

        self.setWindowTitle("Depth Plot with Movable Vertical Line")
        self.setGeometry(100, 100, 800, 600)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        layout = QVBoxLayout(self.main_widget)

        # === Plot setup ===
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.ax.plot(df['time'], df['depth_1'], label='Depth')
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Depth (µm)")
        self.ax.legend()
        self.canvas.draw()
        layout.addWidget(self.canvas)

        # === Slider ===
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(df) - 1)
        self.slider.setValue(0)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.update_vline)
        layout.addWidget(self.slider)

        self.position_label = QLabel("Time: 0.0000 s")
        layout.addWidget(self.position_label)

    def update_vline(self, index):
        time_value = self.df['time'].iloc[index]

        # Remove previous line
        if self.current_line:
            self.current_line.remove()

        # Draw new line
        self.current_line = self.ax.axvline(x=time_value, color='red', linestyle='--')
        self.position_label.setText(f"Time: {time_value:.5f} s")
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)

    # === Load data file ===
    file_path = "IGP-H-IFSW-1_OCT_raw_data.oct"  # <- Set your file path here
    df = read_custom_hist_file(file_path)

    # === Launch window ===
    main_window = PlotWindow(df)
    main_window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

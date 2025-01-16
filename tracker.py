import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class MetalAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Metal Analyzer")
        self.setGeometry(100, 100, 800, 600)

        # Initialize variables
        self.image = None
        self.click_color = None
        self.radius = 10  # Radius for averaging the color
        self.image_display_size = None  # To track QLabel display dimensions
        self.points = []  # Store two points for rotation

        # Layout
        self.central_widget = QWidget()
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # Buttons
        self.load_button = QPushButton("Load Image")
        self.select_button = QPushButton("Select Metal Ground")
        self.binarize_button = QPushButton("Binarize Image")
        self.edge_button = QPushButton("Select Edge and Rotate")
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        # Add to layout
        self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.select_button)
        self.layout.addWidget(self.binarize_button)
        self.layout.addWidget(self.edge_button)
        self.layout.addWidget(self.image_label)

        # Button functionality
        self.load_button.clicked.connect(self.load_image)
        self.select_button.clicked.connect(self.activate_selection_mode)
        self.binarize_button.clicked.connect(self.binarize_image)
        self.edge_button.clicked.connect(self.activate_edge_selection_mode)

        # Enable click detection on image label
        self.image_label.mousePressEvent = self.get_click_point

        # Flags for enabling/disabling functionality
        self.selection_mode = False
        self.edge_selection_mode = False

    def load_image(self):
        """Load an image from file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.tif *.png *.jpg *.bmp)"
        )
        if file_name:
            self.image = cv2.imread(file_name)
            self.display_image(self.image)

    def display_image(self, image):
        """Display an image in the QLabel."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.image_display_size = pixmap.size()

    def activate_selection_mode(self):
        """Activate selection mode to enable color picking."""
        if self.image is None:
            return
        self.selection_mode = True
        self.edge_selection_mode = False
        # self.image_label.setText("Click on the image to select the metal ground...")

    def activate_edge_selection_mode(self):
        """Activate edge selection mode to allow selecting two points for rotation."""
        if self.image is None:
            return
        self.edge_selection_mode = True
        self.selection_mode = False
        self.points = []  # Reset points
        # self.image_label.setText("Click two points on the edge of the metal piece...")

    def get_click_point(self, event):
        """Handle mouse clicks for selecting points."""
        if self.image is None:
            return

        if self.selection_mode:
            self.get_click_color(event)
        elif self.edge_selection_mode:
            self.get_edge_points(event)

    def get_click_color(self, event):
        """Capture the color of the clicked pixel."""
        if self.image is None:
            return

        # Map the click to the image coordinates
        x, y = self.map_click_to_image_coords(event)

        # Average color in the neighborhood
        x_min = max(0, x - self.radius)
        x_max = min(self.image.shape[1], x + self.radius)
        y_min = max(0, y - self.radius)
        y_max = min(self.image.shape[0], y + self.radius)

        region = self.image[y_min:y_max, x_min:x_max]
        self.click_color = np.mean(region, axis=(0, 1))  # Average BGR color

        self.selection_mode = False  # Disable selection mode after picking
        # self.image_label.setText(f"Selected color (BGR): {self.click_color}")

    def get_edge_points(self, event):
        """Capture two points on the edge of the metal piece."""
        x, y = self.map_click_to_image_coords(event)
        self.points.append((x, y))
        # self.image_label.setText(f"Selected points: {self.points}")

        if len(self.points) == 2:
            # Perform rotation
            self.rotate_image()

    def rotate_image(self):
        """Rotate the image so the selected edge is vertical (aligned with the Y-axis)."""
        if len(self.points) != 2:
            return

        # Calculate angle of rotation
        (x1, y1), (x2, y2) = self.points
        delta_x = x2 - x1
        delta_y = y2 - y1
        angle = np.arctan2(delta_y, delta_x) * 180 / np.pi  # Angle in degrees

        # Rotate the image
        center = (self.image.shape[1] // 2, self.image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated_image = cv2.warpAffine(self.image, rotation_matrix, (self.image.shape[1], self.image.shape[0]))

        # Display the rotated image
        self.image = rotated_image  # Update the stored image
        self.display_image(self.image)
        # self.image_label.setText("Image rotated successfully!")
        self.points = []  # Reset points after rotation

    def binarize_image(self):
        """Binarize the image based on the selected color."""
        if self.image is None or self.click_color is None:
            return

        # Define a tolerance range for color matching
        tolerance = 70  # Adjust as needed
        lower_bound = np.clip(self.click_color - tolerance, 0, 255).astype(np.uint8)
        upper_bound = np.clip(self.click_color + tolerance, 0, 255).astype(np.uint8)

        # Create a binary mask
        mask = cv2.inRange(self.image, lower_bound, upper_bound)

        # Display the binarized image
        binarized_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self.display_image(binarized_image)

    def map_click_to_image_coords(self, event):
        """Map the mouse click position to the actual image coordinates."""
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        pixmap_width = self.image_display_size.width()
        pixmap_height = self.image_display_size.height()

        # Map click position to image coordinates
        x_ratio = self.image.shape[1] / pixmap_width
        y_ratio = self.image.shape[0] / pixmap_height

        click_pos = event.pos()
        x = int(click_pos.x() * x_ratio)
        y = int(click_pos.y() * y_ratio)
        print(x,y)
        return x, y


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MetalAnalyzer()
    window.show()
    sys.exit(app.exec())

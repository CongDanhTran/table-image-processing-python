import os
import cv2
import numpy as np
import uuid
from flask import Flask, render_template, request, send_file, after_this_request
from PIL import Image
from natsort import natsorted

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

class TableExtractor:
    def __init__(self, image_array):
        self.image = image_array

    def execute(self):
        self.convert_image_to_grayscale()
        self.threshold_image()
        self.invert_image()
        self.dilate_image()
        self.find_contours()
        self.filter_contours_and_leave_only_rectangles()
        self.find_largest_contour_by_area()
        
        # Guard clause if no table is found
        if self.contour_with_max_area is None:
            return self.image

        self.order_points_in_the_contour_with_max_area()
        self.calculate_new_width_and_height_of_image()
        self.apply_perspective_transform()
        self.add_10_percent_padding()
        return self.perspective_corrected_image_with_padding

    def convert_image_to_grayscale(self):
        self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def threshold_image(self):
        self.thresholded_image = cv2.threshold(self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def invert_image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

    def dilate_image(self):
        self.dilated_image = cv2.dilate(self.inverted_image, None, iterations=5)

    def find_contours(self):
        self.contours, _ = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def filter_contours_and_leave_only_rectangles(self):
        self.rectangular_contours = []
        for contour in self.contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                self.rectangular_contours.append(approx)

    def find_largest_contour_by_area(self):
        max_area = 0
        self.contour_with_max_area = None
        for contour in self.rectangular_contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                self.contour_with_max_area = contour

    def order_points_in_the_contour_with_max_area(self):
        self.contour_with_max_area_ordered = self.order_points(self.contour_with_max_area)

    def calculate_new_width_and_height_of_image(self):
        existing_image_width = self.image.shape[1]
        width_90 = int(existing_image_width * 0.9)
        
        d_top = self.calculateDistanceBetween2Points(self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[1])
        d_side = self.calculateDistanceBetween2Points(self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[3])

        aspect_ratio = d_side / (d_top if d_top != 0 else 1)
        self.new_image_width = width_90
        self.new_image_height = int(self.new_image_width * aspect_ratio)

    def apply_perspective_transform(self):
        pts1 = np.float32(self.contour_with_max_area_ordered)
        pts2 = np.float32([[0, 0], [self.new_image_width, 0], [self.new_image_width, self.new_image_height], [0, self.new_image_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.perspective_corrected_image = cv2.warpPerspective(self.image, matrix, (self.new_image_width, self.new_image_height))

    def add_10_percent_padding(self):
        padding = int(self.perspective_corrected_image.shape[0] * 0.1)
        self.perspective_corrected_image_with_padding = cv2.copyMakeBorder(
            self.perspective_corrected_image, padding, padding, padding, padding, 
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

    def calculateDistanceBetween2Points(self, p1, p2):
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    
    def order_points(self, pts):
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return "No files part", 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return "No selected files", 400

    processed_pil_images = []
    
    # Process each uploaded image
    for file in natsorted(files, key=lambda x: x.filename):
        # Read image from stream
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Process
        extractor = TableExtractor(img)
        result_img = extractor.execute()

        # Convert OpenCv (BGR) to PIL (RGB)
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        processed_pil_images.append(Image.fromarray(result_rgb))

    if processed_pil_images:
        pdf_filename = f"output_{uuid.uuid4().hex}.pdf"
        pdf_path = os.path.join(PROCESSED_FOLDER, pdf_filename)
        
        # Save as PDF
        processed_pil_images[0].save(
            pdf_path, save_all=True, append_images=processed_pil_images[1:]
        )

        return send_file(pdf_path, as_attachment=True)

    return "Processing failed", 500

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
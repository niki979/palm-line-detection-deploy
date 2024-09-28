from flask import Flask, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice_score = (2. * intersection + smooth) / (denominator + smooth)
    return 1 - dice_score

# Load the model
model = tf.keras.models.load_model('model.h5', custom_objects={'dice_loss': dice_loss})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def prepare_image(file_path):
    image = Image.open(file_path).convert('RGB')  # Convert image to RGB
    image = image.resize((384, 384))  # Resize to the expected model input size
    image = np.array(image)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def dilate_mask(mask, kernel_size=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    return dilated_mask

def erode_mask(mask, kernel_size=4):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    return eroded_mask


def smooth_contours(contours, epsilon_factor=0.02):
    smoothed_contours = [cv2.approxPolyDP(cnt, epsilon_factor * cv2.arcLength(cnt, True), True) for cnt in contours]
    return smoothed_contours

# Apply Gaussian Blur
def apply_gaussian_blur(mask, kernel_size=1):
    blurred_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    return blurred_mask

def thin_mask(mask, dilation_iterations=2):
    # Ensure the mask is binary (0s and 255s)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Create an empty skeleton
    skeleton = np.zeros_like(binary_mask)

    # Get a structuring element (cross-shaped)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # Open the image (erosion followed by dilation)
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, element)
        
        # Subtract the opened image from the original image
        temp = cv2.subtract(binary_mask, opened)
        
        # Erode the original image
        eroded = cv2.erode(binary_mask, element)
        
        # Add the temporary image (subtracted) to the skeleton
        skeleton = cv2.bitwise_or(skeleton, temp)
        
        # If no more changes, break the loop
        binary_mask = eroded.copy()
        if cv2.countNonZero(binary_mask) == 0:
            break

    # Dilate the skeleton to thicken the lines slightly
    dilated_skeleton = cv2.dilate(skeleton, element, iterations=dilation_iterations)

    return dilated_skeleton


def thin_mask_medium(mask, erosion_iterations=1, dilation_iterations=1):
    # Ensure the mask is binary (0s and 255s)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Erode the mask (thin the lines)
    kernel = np.ones((3, 3), np.uint8)  # You can adjust the kernel size
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=erosion_iterations)
    
    # Dilate the mask slightly (thicken the lines back a little)
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=dilation_iterations)
    
    return dilated_mask



def filter_contours(mask, min_contour_area=25):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Create an empty mask
    mask = np.zeros_like(mask)

    # Draw filtered contours on the mask
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)
    return mask, filtered_contours

def overlay_mask(image, mask):
    overlay = image.copy()
    # Apply different colors to the mask
    color_map = {
        0: [0, 255, 0],    # Green
        1: [255, 255, 0],  # Yellow
        2: [255, 0, 0],    # Blue
        3: [0, 0, 255]     # Red
    }

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        cv2.drawContours(overlay, [cnt], -1, color_map[i % len(color_map)], thickness=cv2.FILLED)

    return overlay


def mask_to_base64(mask_image):
    buffered = BytesIO()
    mask_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image = prepare_image(file_path)
        pred_mask = model.predict(image)
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        pred_mask = np.squeeze(pred_mask, axis=0) * 255  # Convert to 255 scale

        # Process the mask
        dilated_mask = dilate_mask(pred_mask)
        eroded_mask = erode_mask(dilated_mask)
        blurred_mask=apply_gaussian_blur(eroded_mask)

        filtered_mask, filtered_contours = filter_contours(blurred_mask)

        smoothed_contours = smooth_contours(filtered_contours)

        # Create an empty mask
        smoothed_mask = np.zeros_like(filtered_mask)
        cv2.drawContours(smoothed_mask, smoothed_contours, -1, (255), thickness=cv2.FILLED)
        
        # Apply thinning to get medium-sized lines
        medium_thinned_mask = thin_mask(smoothed_mask)

        # Read and resize the original image
        original_image = cv2.imread(file_path)
        original_image = cv2.resize(original_image, (384, 384))  # Resize to the expected model input size

        # Ensure the mask dimensions match the image dimensions
        overlay = overlay_mask(original_image, medium_thinned_mask)
        overlay_image = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        # Convert overlay image to base64
        overlay_base64 = mask_to_base64(overlay_image)

        # Find the lengths of the remaining contours
        contour_lengths = []
        for i, cnt in enumerate(filtered_contours):
            arc_length = cv2.arcLength(cnt, True)
            contour_lengths.append((i + 1, arc_length))

        return render_template('index.html', mask_url=f"data:image/png;base64,{overlay_base64}", contour_lengths=contour_lengths)

    return redirect(request.url)

@app.route('/predict-webcam', methods=['POST'])
def predict_webcam():
    if 'image_data' in request.form:
        image_data = request.form['image_data']
        image_data = image_data.split(',')[1]
        image_data = base64.b64decode(image_data)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_image.png')

        with open(file_path, 'wb') as f:
            f.write(image_data)

        image = prepare_image(file_path)
        pred_mask = model.predict(image)
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        pred_mask = np.squeeze(pred_mask, axis=0) * 255  # Convert to 255 scale

        # Process the mask
        dilated_mask = dilate_mask(pred_mask)
        eroded_mask = erode_mask(dilated_mask)
        filtered_mask, filtered_contours = filter_contours(eroded_mask)
        
        # Read and resize the original image
        original_image = cv2.imread(file_path)
        original_image = cv2.resize(original_image, (384, 384))  # Resize to the expected model input size
        
        # Ensure the mask dimensions match the image dimensions
        overlay = overlay_mask(original_image, filtered_mask)
        overlay_image = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        # Convert overlay image to base64
        overlay_base64 = mask_to_base64(overlay_image)

        # Find the lengths of the remaining contours
        contour_lengths = []
        for i, cnt in enumerate(filtered_contours):
            arc_length = cv2.arcLength(cnt, True)
            contour_lengths.append((i + 1, arc_length))

        return render_template('index.html', mask_url=f"data:image/png;base64,{overlay_base64}", contour_lengths=contour_lengths)

    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

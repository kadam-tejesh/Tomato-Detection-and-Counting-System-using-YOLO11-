import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_non_empty_file(file_storage):
    file_storage.stream.seek(0, os.SEEK_END)
    size = file_storage.stream.tell()
    file_storage.stream.seek(0)
    return size > 0

# =====================================================
# Model loading and prediction
# =====================================================
def load_model():
    model = YOLO('C:/Users/Dileep Sathya/OneDrive/Desktop/obj_det_2/runs/detect/train4/weights/best.pt')
    return model

def predict_image(model, image_path):
    results = model(image_path, conf=0.3)
    result = results[0]
    tomato_count = len(result.boxes)
    label = f"Tomato count: {tomato_count}"
    annotated_img = result.plot()
    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)
    output_filename = f"{name}_predicted.jpg"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, annotated_img)
    return label, output_path

model = load_model()
# =====================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_upload', methods=['POST'])
def process_upload():
    saved_images = []

    # 1. Try single file first
    single_file = request.files.get('file')
    if single_file and single_file.filename != '':
        if allowed_file(single_file.filename) and is_non_empty_file(single_file):
            filename = secure_filename(single_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            single_file.save(filepath)
            saved_images.append(filepath)
        else:
            return jsonify(success=False, error='Single file must be a non‑empty JPG image.')
    else:
        # 2. No single file, try folder upload
        folder_files = request.files.getlist('folder')
        if folder_files and any(f.filename != '' for f in folder_files):
            for f in folder_files:
                if f.filename != '':
                    if allowed_file(f.filename) and is_non_empty_file(f):
                        filename = secure_filename(f.filename)
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        f.save(filepath)
                        saved_images.append(filepath)
                    else:
                        return jsonify(success=False, error='All files in the folder must be non‑empty JPG images.')
        else:
            return jsonify(success=False, error='Please select at least one JPG image or a folder with JPG images.')

    # Run predictions
    results = []
    total_count = 0
    for img_path in saved_images:
        label, output_path = predict_image(model, img_path)
        try:
            count = int(label.split(':')[-1].strip())
        except:
            count = 0
        total_count += count
        image_url = url_for('static', filename='uploads/' + os.path.basename(output_path))
        results.append({'image_url': image_url, 'label': label})

    # Store results in session
    session['results'] = results
    session['total_count'] = total_count

    return jsonify(success=True)

@app.route('/results')
def show_results():
    # Retrieve results from session
    results = session.get('results')
    total_count = session.get('total_count')
    if not results:
        return redirect(url_for('index'))
    return render_template('results.html', results=results, total_count=total_count)

if __name__ == '__main__':
    app.run(debug=True)
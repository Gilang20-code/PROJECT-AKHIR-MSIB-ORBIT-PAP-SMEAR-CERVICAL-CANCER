from flask import (Flask,render_template,request,redirect,url_for,jsonify,Response,send_from_directory,)
from werkzeug.utils import secure_filename
import os
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Konfigurasi folder upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk validasi ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Cek apakah ada file yang diunggah
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            # Amankan nama file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Simpan file
            file.save(filepath)
            
            # Analisis gambar
            analysis_result = analyze_image(filepath)
            file_url = url_for('uploaded_file', filename=filename)
            # return f"File uploaded successfully! Analysis result: {analysis_result}"
            return jsonify({'filename': filename, 'file_url': file_url, 'prediction': analysis_result})
            
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Fungsi untuk analisis gambar
def analyze_image(filepath):
    # Muat model TensorFlow dan analisis gambar
    model = tf.keras.models.load_model('./model/cnn_model_final.h5')
    class_labels = ['h-sil', 'l-sil', 'sel koilocyt', 'sel normal']  # Sesuaikan dengan dataset Anda
    
    
    img = tf.keras.preprocessing.image.load_img(filepath, target_size=(200, 200))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return f"{prediction.argmax()}"

if __name__ == "__main__":
     # Pastikan folder upload ada
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import time

app = Flask(__name__)

# Konfigurasi folder untuk menyimpan file yang diunggah dan ekstensi yang diizinkan
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Buat folder jika belum ada
upload_folder = app.config['UPLOAD_FOLDER']
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Load model machine learning yang telah dilatih sebelumnya
model_cnn = load_model('CNN.h5')
model_MobileNet = load_model('MobileNet.h5')

# Tentukan label kelas untuk hasil prediksi
labels = ['Sunburn', 'Bercak Daun', 'Sehat']

# Fungsi untuk memeriksa apakah file memiliki ekstensi yang diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Fungsi untuk mendapatkan waktu eksekusi prediksi
def get_prediction_time(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    return round(elapsed_time, 3)

# Rute untuk menampilkan halaman unggah awal
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            return render_template('upload.html', filename=filename)
    
    return render_template('upload.html')

# Rute untuk menangani prediksi gambar
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        filename = request.form['filename']
        selected_model = request.form['selected_model']
    else:
        filename = request.args.get('filename')
        selected_model = request.args.get('selected_model')

    # Path gambar
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    preprocessing_number = 0

    # Load gambar dan model yang dipilih
    if selected_model == 'cnn':
        selected_model = model_cnn
        img = image.load_img(file_path, target_size=(128, 128))
    elif selected_model == 'MobileNet':
        selected_model = model_MobileNet
        img = image.load_img(file_path, target_size=(128, 128))
        preprocessing_number = 1
    else:
        return render_template('error.html', message='Pemilihan model tidak valid')

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # mulai waktu untuk mengukur lama waktu prediksi
    start_time = time.time()
    prediction = selected_model.predict(img_array)
    predicted_class = ''
    prediction_confidence = 0

    # Proses prediksi berdasarkan preprocessing_number
    if preprocessing_number == 0:
        predicted_class_index = np.argmax(prediction)
        predicted_class = labels[predicted_class_index]
        prediction_confidence = prediction[0][predicted_class_index] * 100
    elif preprocessing_number == 1:
        predicted_class_indices = np.where(prediction > 0.5)[1]
        if len(predicted_class_indices) > 0:
            predicted_class = labels[predicted_class_indices[0]]
            prediction_confidence = prediction[0][predicted_class_indices[0]] * 100
        else:
            predicted_class = 'Tidak Ada Kelas yang Diprediksi'
            prediction_confidence = 0

    prediction_time = get_prediction_time(start_time)
    return render_template('predict.html', filename=filename, prediction=predicted_class, prediction_time=prediction_time, confidence=prediction_confidence)

# Jalankan aplikasi jika script dijalankan
if __name__ == '__main__':
    app.run(debug=True)

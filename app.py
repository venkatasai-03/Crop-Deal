import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

categories = ['Mirchi', 'Mango', 'Lemon', 'Papaya', 'Potato', 'Tomato', 'Banana']
quality_labels = ['90', '70', '60']

IMG_SIZE = 128

model_path = "final_model.h5"
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(image_path):
    image_array = preprocess_image(image_path)
    predictions = model.predict(image_array)

    crop_pred = predictions[0]
    quality_pred = predictions[1]

    crop_index = np.argmax(crop_pred)
    quality_index = np.argmax(quality_pred)

    predicted_crop = categories[crop_index]
    predicted_quality = quality_labels[quality_index]

    return predicted_crop, predicted_quality

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            predicted_crop, predicted_quality = predict_image(file_path)

            image_url = url_for('static', filename='uploads/' + file.filename)

            return render_template('index.html', crop=predicted_crop, quality=predicted_quality, image_url=image_url)

    return render_template('index.html', crop=None, quality=None, image_url=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

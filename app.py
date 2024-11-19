import os
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

categories = ['Mirchi', 'Mango', 'Lemon', 'Papaya', 'Potato', 'Tomato', 'Banana']
quality_labels = ['90', '70', '60']

IMG_SIZE = 128
model_path = "final_model.h5"

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model with error handling
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    """Check if the uploaded file is allowed based on the extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Preprocess the image before making predictions."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(image_path):
    """Predict the crop and quality of the uploaded image."""
    if model is None:
        return None, None

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
    """Handle image upload and prediction."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            predicted_crop, predicted_quality = predict_image(filename)

            if predicted_crop is None or predicted_quality is None:
                flash('Error in prediction. Please try again.', 'error')
                return redirect(request.url)

            image_url = url_for('static', filename='uploads/' + file.filename)

            return render_template('index.html', crop=predicted_crop, quality=predicted_quality, image_url=image_url)
        else:
            flash('Invalid file type. Please upload an image.', 'error')
            return redirect(request.url)

    return render_template('index.html', crop=None, quality=None, image_url=None)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

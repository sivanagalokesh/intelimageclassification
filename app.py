from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('c:\\Users\\sivan\\OneDrive\\Desktop\\csp\\my_model.h5')

# Define class labels (adjust based on your dataset)
class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Define the upload folder
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))

    if file:
        # Save the uploaded image with a secure filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        # Get the predicted label
        predicted_label = class_labels[predicted_class]

        return render_template('index.html', prediction=predicted_label, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json

app = Flask(__name__)

# Load the pre-trained model and class_indices
model = load_model('model/final_model.h5')
with open('model/class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            # Save the uploaded image file
            image_path = './static/' + file.filename
            file.save(image_path)

            # Load and preprocess the image
            img = load_img(image_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalization

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)
            predicted_class = class_labels[predicted_class_index[0]]

            return render_template('index.html', image_url=image_path, prediction=predicted_class)

    return render_template('index.html', image_url=None, prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

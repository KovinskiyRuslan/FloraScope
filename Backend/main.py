import tensorflow as tf
from firebase_admin import firestore
from firebase_admin import credentials, initialize_app, storage
from flask import request
import os
from flask import Flask
from uuid import uuid4
from flask import jsonify

app = Flask(__name__)
from cred import cred_path
import numpy as np
cred = credentials.Certificate(cred_path)
initialize_app(cred, {'storageBucket': 'florascope-69f98.appspot.com'})


db = firestore.client()
from tensorflow import keras
import numpy as np

from keras.utils import array_to_img, img_to_array

models = {}

for model_name in ['apple_model', 'grape_model', 'watermelon_model', 'cucumber_model']:
    model_path = f'models/{model_name}.h5'
    if os.path.exists(model_path):
        models[model_name] = tf.keras.models.load_model(model_path, compile=False)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_id = data.get('image_id')
    model_name = data.get('model_name')

    if not image_id or not model_name:
        return jsonify({'message': 'Image ID and model name are required'}), 400

    model = models.get(model_name)
    if not model:
        return jsonify({'message': 'Model not found'}), 404

    blob = storage.bucket().blob(f'images/{image_id}')
    if not blob.exists():
        return jsonify({'message': 'Image not found'}), 404
    image_bytes = blob.download_as_string()

    image = tf.io.decode_image(image_bytes, channels=3)
    image = tf.image.resize(image, [512, 512])
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)

    test_acc = model.predict(img)
    print(f"Test accuracy: {test_acc}")

    result_text = str(test_acc)
    return jsonify({'message': 'Analysis completed', 'id': image_id, "result": result_text}), 200



@app.route('/dab', methods=['GET'])
def get_r():
    print("dab")
    return jsonify('success'), 200

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)

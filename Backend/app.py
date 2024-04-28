import tensorflow as tf
from firebase_admin import firestore
from google.cloud import storage as google_storage

google_storage_client = google_storage.Client()
google_bucket_name = 'florascope1.appspot.com'
google_bucket = google_storage_client.bucket(google_bucket_name)

from firebase_admin import credentials, initialize_app, storage
from flask import request
import os
from flask import Flask
from uuid import uuid4
from flask import jsonify


def download_blob_to_tmp(source_blob_name):
    """Функция для загрузки blob во временный файл."""
    destination_file_name = f'/tmp/{uuid4()}.tflite'
    blob = google_bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    return destination_file_name

app = Flask(__name__)
from cred import cred
cred_path = cred


cred = credentials.Certificate(cred_path)
initialize_app(cred, {'storageBucket': 'florascope-27e15.appspot.com'})


db = firestore.client()


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        ext = os.path.splitext(file.filename)[1]
        unique_filename = str(uuid4()) + ext
        blob = storage.bucket().blob('images/' + unique_filename)
        blob.upload_from_string(file.read(), content_type=file.content_type)
        return jsonify({'id': unique_filename}), 200


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_id = data.get('image_id')
    model_name = data.get('model_name')

    if not image_id or not model_name:
        return 'Image ID and model name are required', 400

    model_blob_name = f'{model_name}.tflite'
    model_path = download_blob_to_tmp(model_blob_name)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    blob = storage.bucket().blob(f'images/{image_id}')
    if not blob.exists():
        return 'Image not found', 404
    image_bytes = blob.download_as_string()

    image = tf.io.decode_image(image_bytes, channels=3)
    image = tf.image.resize(image, [200, 200])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, 0)

    interpreter.set_tensor(input_details[0]['index'], image.numpy())
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    print(prediction)
    result_text = str(prediction)
    results_blob = storage.bucket().blob(f'results/{image_id}.txt')
    results_blob.upload_from_string(result_text, content_type='text/plain')

    return jsonify({'message': 'Analysis completed', 'id': image_id, "result": result_text}), 200


@app.route('/results/<image_id>', methods=['GET'])
def get_results(image_id):
    blob = storage.bucket().blob(f'results/{image_id}.txt')
    if not blob.exists():
        return 'Results not found', 404
    results = blob.download_as_text()
    return jsonify(results), 200

@app.route('/dab', methods=['GET'])
def get_r():
    print("dab")
    return jsonify('success'), 200

@app.route('/test_firestore')
def test_firestore():
    test_doc = db.collection(u'test_collection').document(u'test_document')
    test_doc.set({
        u'name': u'Test',
        u'value': u'Hello Firestore!'
    })
    return 'Document created'


if __name__ == '__main__':
    app.run(debug=True)

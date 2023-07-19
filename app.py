import base64
import json
import os
import requests
import subprocess

from flask import Flask, flash, redirect, request, jsonify, send_from_directory, url_for
from flask_sqlalchemy import SQLAlchemy
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema
from pymilvus import connections, utility
from werkzeug.utils import secure_filename

import jax.numpy as jnp

from kmeans import kmeans

MILVUS_IP="127.0.0.1"
PROJECT_ID="CHANGE ME"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

connections.connect(host=MILVUS_IP, port=19530)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    access_token = subprocess.getoutput('gcloud auth print-access-token')
    if 'image' in request.files:
        filename = secure_filename(request.files['image'].filename)
        request.files['image'].save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            response = requests.post(
                f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/multimodalembedding@001:predict",
                headers={
                    "Authorization": f"Bearer {access_token}"
                },
                json={
                    "instances": [
                        {"image": {"bytesBase64Encoded": encoded_image}}
                    ]
                }
            )
            collection.insert({
                "embedding": response.json()["predictions"][0]["imageEmbedding"],
                "image": filename,
                "description": ""
            })

        return jsonify({'image': url_for('download_file', name=filename)})
    elif 'description' in request.form:
        description = request.form['description']

        response = requests.post(
            f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/multimodalembedding@001:predict",
            headers={
                "Authorization": f"Bearer {access_token}"
            },
            json={
                "instances": [
                    {"text": description}
                ]
            }
        )
        collection.insert({
            "embedding": response.json()["predictions"][0]["textEmbedding"],
            "image": "",
            "description": description
        })

        return jsonify({'text': request.form['description']})
    else:
        return jsonify({'error': 'An error occurred'}), 400

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

@app.route('/cluster', methods=['POST'])
def cluster():
    collection.load()
    vectors = collection.query(
        expr = "id >= 0", 
        offset = 0,
        limit = 16384,
        output_fields = ["image", "description", "embedding"]
    )
    embeddings = jnp.array([vector["embedding"] for vector in vectors])

    clusters = kmeans(embeddings, k=int(request.get_json()["k"]))
    for i, vector in enumerate(vectors):
        vectors[i]["cluster"] = int(clusters[1][i])

    # Drop ids and embeddings from the dictionary
    for i, vector in enumerate(vectors):
        del vectors[i]["id"]
        del vectors[i]["embedding"]

    return jsonify({'vectors': vectors})

def refresh_milvus():
    global collection
    # Create collection
    utility.drop_collection("MultimodalDemo")
    id = FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True
    )
    embedding = FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=1408
    )
    image = FieldSchema(
        name="image",
        dtype=DataType.VARCHAR,
        max_length=256,
    )
    description = FieldSchema(
        name="description",
        dtype=DataType.VARCHAR,
        max_length=4096,
    )
    schema = CollectionSchema(
        fields=[id, embedding, image, description],
        auto_id=True
    )
    collection = Collection("MultimodalDemo", schema=schema)
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }

    collection.create_index("embedding", index)

if __name__ == '__main__':
    refresh_milvus()
    app.run(debug=True, host="0.0.0.0", port=8080)
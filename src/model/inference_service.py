import numpy as np
import os
import json
from flask import Flask, request, jsonify
import logging
import pandas as pd

logger = logging.getLogger(__name__)
app = Flask(__name__)

W = None
H = None
X = None
Y = None

@app.route("/load_model", methods=["POST"])
def load_model():
    global W, H, X, Y
    data = request.json
    w_path = data.get("w_path")
    h_path = data.get("h_path")
    x_path = data.get("x_path")
    y_path = data.get("y_path")

    if w_path and h_path and x_path and y_path and \
       os.path.exists(w_path) and os.path.exists(h_path) and \
       os.path.exists(x_path) and os.path.exists(y_path):
        W = np.loadtxt(w_path, delimiter=",")
        H = np.loadtxt(h_path, delimiter=",")
        X_df = pd.read_parquet(x_path)
        Y_df = pd.read_parquet(y_path)
        X = X_df.values
        Y = Y_df.values
        logger.info("Model and features loaded successfully.")
        return jsonify({"status":"ok","message":"Model and features loaded"}), 200
    else:
        logger.error("Invalid model/feature paths or files missing.")
        return jsonify({"status":"error","message":"Invalid paths"}),400

@app.route("/predict", methods=["POST"])
def predict_latency():
    global W, H, X, Y
    if W is None or H is None or X is None or Y is None:
        logger.error("Model or features not loaded.")
        return jsonify({"status":"error","message":"Model not loaded"}),400

    data = request.json
    qid = data.get("query_id")
    hid = data.get("hint_id")

    if qid is None or hid is None:
        return jsonify({"status":"error","message":"query_id and hint_id required"}),400

    if not (0 <= qid < X.shape[0]) or not (0 <= hid < Y.shape[0]):
        return jsonify({"status":"error","message":"Invalid qid/hid"}),400

    x_i = X[qid,:]
    y_j = Y[hid,:]
    query_factor = x_i @ W
    hint_factor = y_j @ H
    pred = float((query_factor * hint_factor).sum())

    logger.debug(f"Predicted latency qid={qid}, hid={hid}: {pred}")
    return jsonify({"predicted_latency": pred}), 200

def start_service(host="0.0.0.0", port=8080):
    logger.info("Starting inference service...")
    app.run(host=host, port=port)

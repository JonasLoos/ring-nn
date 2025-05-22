import pickle
from flask import Flask, jsonify, render_template, request
import numpy as np
import os
import glob

app = Flask(__name__)

# Helper to convert numpy arrays in logs to lists for JSON serialization
def convert_numpy_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, list):
        return [convert_numpy_to_list(item) for item in data]
    if isinstance(data, dict):
        return {k: convert_numpy_to_list(v) for k, v in data.items()}
    if isinstance(data, (np.float32, np.float64, np.int8, np.int16, np.int32, np.int64)):
        return data.item() # Convert numpy scalar types to native Python types
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/list-logs')
def list_logs():
    try:
        log_files = [os.path.basename(f) for f in glob.glob("logs/*log*.pkl")]
        if not log_files:
            # Try to find any .pkl file if the specific pattern yields nothing, for broader compatibility
            log_files = [os.path.basename(f) for f in glob.glob("logs/*.pkl")]
        return jsonify(log_files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/data')
def get_data():
    log_file = request.args.get('log_file')
    if not log_file:
        return jsonify({"error": "log_file parameter is required"}), 400

    try:
        file_path = os.path.join('logs', log_file)
        if not os.path.exists(file_path) or not os.path.isfile(file_path) :
             return jsonify({"error": f"Log file '{log_file}' not found in 'logs/' directory."}), 404

        with open(file_path, 'rb') as f:
            train_logs = pickle.load(f)

        # Convert numpy arrays to lists for JSON serialization
        train_logs_serializable = convert_numpy_to_list(train_logs)
        return jsonify(train_logs_serializable)
    except FileNotFoundError: # This specific exception might be less likely now with the os.path.exists check
        return jsonify({"error": f"Log file '{log_file}' not found. Checked: {file_path}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

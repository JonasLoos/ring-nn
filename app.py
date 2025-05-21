import pickle
from flask import Flask, jsonify, render_template
import numpy as np

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

@app.route('/data')
def get_data():
    try:
        with open('train_logs.pkl', 'rb') as f:
            train_logs = pickle.load(f)

        # Convert numpy arrays to lists for JSON serialization
        train_logs_serializable = convert_numpy_to_list(train_logs)
        return jsonify(train_logs_serializable)
    except FileNotFoundError:
        return jsonify({"error": "train_logs.pkl not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

import pickle
from flask import Flask, jsonify, render_template, request, Response
import numpy as np
import os
import glob
import json
import struct
from tensor import RingTensor, RealTensor

app = Flask(__name__)

# CORS helper
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Helper to combine arrays across training steps
def combine_training_arrays(train_logs):
    if not train_logs or len(train_logs) == 0:
        return train_logs
    
    # Extract scalar data (non-array data)
    scalar_data = []
    for log in train_logs:
        scalar_entry = {}
        for key, value in log.items():
            if not isinstance(value, list) or not value or not isinstance(value[0], np.ndarray):
                scalar_entry[key] = value
        scalar_data.append(scalar_entry)
    
    # Combine arrays for each layer across all training steps
    combined_data = {
        'scalar_data': scalar_data,
        'combined_arrays': {}
    }
    
    # Get the first log to understand structure
    first_log = train_logs[0]
    
    # Process each array type (weights, updates_float, updates_final)
    for array_type in ['weights', 'updates_float', 'updates_final']:
        if array_type not in first_log:
            continue
            
        combined_data['combined_arrays'][array_type] = []
        
        # For each layer
        for layer_idx in range(len(first_log[array_type])):
            # Collect all arrays for this layer across all training steps
            layer_arrays = []
            for log in train_logs:
                if array_type in log and layer_idx < len(log[array_type]):
                    layer_arrays.append(log[array_type][layer_idx])
            
            if layer_arrays:
                # Stack arrays along a new first dimension (time/step dimension)
                combined_array = np.stack(layer_arrays, axis=0)
                combined_data['combined_arrays'][array_type].append({
                    'type': 'numpy_array',
                    'shape': combined_array.shape,
                    'dtype': str(combined_array.dtype),
                    'data': combined_array.tobytes()
                })
    
    return combined_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/list-logs')
def list_logs():
    try:
        log_files = [os.path.basename(f) for f in sorted(glob.glob("logs/*log*.pkl"), reverse=True)]
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

        # Combine arrays across training steps for efficiency
        combined_data = combine_training_arrays(train_logs)
        
        # Convert combined arrays to binary format with indices
        binary_data_list = []
        
        def convert_to_binary_format(obj):
            if isinstance(obj, dict) and obj.get('type') == 'numpy_array':
                binary_data_list.append(obj['data'])
                return {
                    'type': 'numpy_array',
                    'shape': obj['shape'],
                    'dtype': obj['dtype'],
                    'index': len(binary_data_list) - 1
                }
            elif isinstance(obj, list):
                return [convert_to_binary_format(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_binary_format(v) for k, v in obj.items()}
            elif isinstance(obj, (np.float32, np.float64, np.int8, np.int16, np.int32, np.int64)):
                return obj.item()
            else:
                return obj
        
        metadata = convert_to_binary_format(combined_data)
        
        # Store binary data in session or cache (simple in-memory for now)
        if not hasattr(app, 'binary_cache'):
            app.binary_cache = {}
        
        cache_key = f"{log_file}_{len(binary_data_list)}"
        app.binary_cache[cache_key] = binary_data_list
        
        return jsonify({
            'metadata': metadata,
            'cache_key': cache_key,
            'binary_count': len(binary_data_list)
        })
        
    except FileNotFoundError:
        return jsonify({"error": f"Log file '{log_file}' not found. Checked: {file_path}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/binary/<cache_key>/<int:index>')
def get_binary_data(cache_key, index):
    if not hasattr(app, 'binary_cache') or cache_key not in app.binary_cache:
        return jsonify({"error": "Binary data not found"}), 404
    
    binary_data_list = app.binary_cache[cache_key]
    if index >= len(binary_data_list):
        return jsonify({"error": "Binary index out of range"}), 404
    
    return Response(binary_data_list[index], mimetype='application/octet-stream')

@app.route('/list-models')
def list_models():
    """List available .pkl model files in logs/ directory"""
    try:
        model_files = [os.path.basename(f) for f in sorted(glob.glob("logs/*.pkl"), reverse=True) 
                      if not 'log' in os.path.basename(f).lower()]
        return add_cors_headers(jsonify(model_files))
    except Exception as e:
        return add_cors_headers(jsonify({"error": str(e)})), 500

@app.route('/convert-model')
def convert_model():
    """Convert a .pkl model file to .bin format for TypeScript"""
    model_file = request.args.get('model_file')
    if not model_file:
        return jsonify({"error": "model_file parameter is required"}), 400
    
    try:
        file_path = os.path.join('logs', model_file)
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return jsonify({"error": f"Model file '{model_file}' not found in 'logs/' directory."}), 404
        
        # Load weights from pickle
        with open(file_path, 'rb') as f:
            weights = pickle.load(f)
        
        # Convert to .bin format (same as TypeScript saveToBlob)
        metas = []
        payloads = []
        offset = 0
        
        for w in weights:
            # Determine dtype and get raw bytes
            if isinstance(w, RingTensor):
                dtype = 'int16'
                raw_data = w.data.cpu().numpy().astype(np.int16)
                bytes_data = raw_data.tobytes()
            elif isinstance(w, RealTensor):
                dtype = 'float32'
                raw_data = w.data.cpu().numpy().astype(np.float32)
                bytes_data = raw_data.tobytes()
            else:
                return jsonify({"error": f"Unknown tensor type: {type(w)}"}), 400
            
            metas.append({
                'dtype': dtype,
                'shape': list(w.shape),
                'byteOffset': offset,
                'byteLength': len(bytes_data)
            })
            payloads.append(bytes_data)
            offset += len(bytes_data)
        
        # Build header
        header = {
            'version': 1,
            'tensors': metas
        }
        header_json = json.dumps(header)
        header_bytes = header_json.encode('utf-8')
        
        # Build binary file: [Uint32 headerLength][header][payloads...]
        result = bytearray()
        result.extend(struct.pack('<I', len(header_bytes)))  # Little-endian uint32
        result.extend(header_bytes)
        for payload in payloads:
            result.extend(payload)
        
        response = Response(bytes(result), mimetype='application/octet-stream',
                        headers={'Content-Disposition': f'attachment; filename={os.path.splitext(model_file)[0]}.bin'})
        return add_cors_headers(response)
    
    except FileNotFoundError:
        return jsonify({"error": f"Model file '{model_file}' not found. Checked: {file_path}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

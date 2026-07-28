"""
Video Transformation Engine - Web Server Backend
===============================================
Flask API server powering the VORTEX web UI. Provides endpoints for:
  - Video upload & preview
  - Standard transformation pipeline (FFmpeg)
  - Adversarial perturbation pipeline (PyTorch + FFmpeg)
  - Output comparison & download
"""

import os
import uuid
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from benchmark import VideoTransformationEngine

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['OUTPUT_FOLDER'] = str(OUTPUT_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

engine = VideoTransformationEngine()

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'flv'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_id = uuid.uuid4().hex[:8]
        filename = f"input_{unique_id}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        return jsonify({
            'success': True,
            'file_id': unique_id,
            'original_filename': file.filename,
            'filename': filename,
            'video_url': url_for('get_upload_file', filename=filename)
        })

    return jsonify({'error': 'Unsupported file format'}), 400


@app.route('/api/process', methods=['POST'])
def process_video():
    data = request.get_json() or {}
    filename = data.get('filename')

    if not filename:
        return jsonify({'error': 'Missing input filename'}), 400

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(input_path):
        return jsonify({'error': 'Input file not found on server'}), 404

    # Build full config from request data
    config = {
        # Video stream
        "video_codec": data.get('video_codec', 'h264'),
        "resolution_scale": float(data.get('resolution_scale', 0.85)),
        "crf": int(data.get('crf', 24)),
        "frame_rate": int(data.get('frame_rate', 30)),
        "interpolation": data.get('interpolation', 'bicubic'),
        "grain_strength": float(data.get('grain_strength', 0.0)),
        # Audio stream
        "audio_codec": data.get('audio_codec', 'aac'),
        "audio_bitrate": data.get('audio_bitrate', '128k'),
        "pitch_shift": float(data.get('pitch_shift', 0.0)),
        "eq_filter": bool(data.get('eq_filter', False)),
        # Evasion transforms
        "mirror": bool(data.get('mirror', False)),
        "zoom": bool(data.get('zoom', False)),
        "zoom_factor": float(data.get('zoom_factor', 1.05)),
        "speed": float(data.get('speed', 1.0)),
        "micro_rotate": bool(data.get('micro_rotate', False)),
        "add_border": bool(data.get('add_border', False)),
        # Adversarial
        "adversarial_enabled": bool(data.get('adversarial_enabled', False)),
        "adversarial_epsilon": float(data.get('adversarial_epsilon', 8.0)),
        "adversarial_steps": int(data.get('adversarial_steps', 40)),
        "adversarial_batch_size": int(data.get('adversarial_batch_size', 4)),
        # Metadata
        "strip_metadata": bool(data.get('strip_metadata', True)),
        "inject_metadata": bool(data.get('inject_metadata', False)),
        # Container
        "container": data.get('container', 'mp4')
    }

    container_ext = config["container"].lower()
    output_filename = f"processed_{Path(filename).stem}.{container_ext}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    try:
        result = engine.transform(input_path, output_path, config)
        result["processed_video_url"] = url_for('get_output_file', filename=output_filename)
        result["download_url"] = url_for('get_output_file', filename=output_filename, as_attachment='true')
        return jsonify({"success": True, "result": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/uploads/<path:filename>')
def get_upload_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/outputs/<path:filename>')
def get_output_file(filename):
    as_attachment = request.args.get('as_attachment', 'false').lower() == 'true'
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=as_attachment)


if __name__ == '__main__':
    print("=" * 60)
    print("  VORTEX Video Transformation Engine")
    print("  http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=False)

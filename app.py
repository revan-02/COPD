import os
import numpy as np
import librosa
from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit

from inference.clinical_inference import predict_clinical, MODEL_LOADED as CLINICAL_LOADED
from inference.audio_inference import predict_audio, MODEL_LOADED as AUDIO_LOADED, model as audio_model
from inference.fusion_inference import predict_fusion, MODEL_LOADED as FUSION_LOADED

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'copd-secret'
app.config['TEMPLATES_AUTO_RELOAD'] = True
os.makedirs('uploads', exist_ok=True)

socketio = SocketIO(app, cors_allowed_origins="*")

client_sessions = {}
TARGET_SR = 16000
WINDOW_SECONDS = 5.0
BUFFER_SIZE = int(TARGET_SR * WINDOW_SECONDS)

# ----------------- ROUTES -----------------

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/live')
def live_monitor():
    return render_template('live_audio.html')

RACE_OPTIONS = [
    ('caucasian', 'Caucasian / White'),
    ('african_american', 'African American / Black'),
    ('mexican_american', 'Mexican American / Hispanic'),
    ('other', 'Other / Asian (0.88 factor)')
]

@app.route('/', methods=['GET', 'POST'])
@app.route('/predict', methods=['POST'])
def predict():
    status = {
        'clinical': CLINICAL_LOADED,
        'audio': AUDIO_LOADED,
        'fusion': FUSION_LOADED
    }
    
    if request.method == 'GET':
        return render_template('index.html', race_options=RACE_OPTIONS, form=None, result=None, status=status)

    try:
        clinical_prob = predict_clinical(request.form)

        audio_file = request.files.get('audio_file')
        audio_prob = None
        is_audio_synthetic = False
        audio_filename = None
        if audio_file and audio_file.filename:
            audio_filename = audio_file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
            audio_file.save(filepath)
            audio_prob = predict_audio(filepath)
        elif request.form.get('live_audio_prob'):
            audio_prob = float(request.form.get('live_audio_prob'))
            audio_filename = "Live Microphone Audio"

        fusion_prob = predict_fusion(clinical_prob, audio_prob)

        result_data = {
            'clinical_prob': f"{clinical_prob*100:.1f}%" if clinical_prob is not None else "MODEL NOT LOADED",
            'audio_prob': audio_prob if isinstance(audio_prob, str) else (f"{audio_prob*100:.1f}%" if audio_prob is not None else ("MODEL NOT LOADED" if not AUDIO_LOADED else "No File")),
            'is_audio_synthetic': is_audio_synthetic,
            'audio_filename': audio_filename,
            'fusion_prob': f"{fusion_prob*100:.1f}%" if fusion_prob is not None else "MODEL NOT LOADED",
            'classification': "Unknown"
        }
        
        if fusion_prob is not None:
            if fusion_prob < 0.4:
                result_data['classification'] = "Low"
            elif fusion_prob < 0.7:
                result_data['classification'] = "Moderate"
            else:
                result_data['classification'] = "High"

        return render_template('index.html', race_options=RACE_OPTIONS, form=request.form, result=result_data, status=status)

    except Exception as e:
        return render_template('index.html', race_options=RACE_OPTIONS, form=request.form,
                               result={'error': f'Prediction error: {str(e)}'}, status=status)

# ----------------- WEBSOCKETS -----------------

@socketio.on('connect')
def test_connect():
    print('Client connected:', request.sid)
    client_sessions[request.sid] = {
        'buffer': np.array([], dtype=np.float32),
        'last_prediction_time': 0
    }

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected:', request.sid)
    if request.sid in client_sessions:
        del client_sessions[request.sid]

@socketio.on('start_session')
def handle_start():
    client_sessions[request.sid]['buffer'] = np.array([], dtype=np.float32)

@socketio.on('audio_chunk')
def handle_audio(chunk):
    """
    Receives Int16 Array Buffer from the mobile client, appends to the rolling 
    buffer, and runs inference if the buffer is full (5 seconds).
    """
    if request.sid not in client_sessions:
        client_sessions[request.sid] = {
            'buffer': np.array([], dtype=np.float32),
            'last_prediction_time': 0
        }

    # Convert binary chunk back to Int16, then normalize to float32
    int16_arr = np.frombuffer(chunk, dtype=np.int16)
    float_arr = int16_arr.astype(np.float32) / 32768.0

    session = client_sessions[request.sid]
    session['buffer'] = np.concatenate((session['buffer'], float_arr))
    # print(f"Buffer size: {len(session['buffer'])} / {BUFFER_SIZE}")

    if len(session['buffer']) >= BUFFER_SIZE:
        audio_window = session['buffer'][-BUFFER_SIZE:]
        # Keep last 1 second for overlap
        session['buffer'] = session['buffer'][-(TARGET_SR*1):]
        
        prediction, confidence, abnormal_prob = run_live_inference(audio_window)
        
        emit('prediction_result', {
            'prediction': prediction,
            'confidence': confidence,
            'abnormal_prob': abnormal_prob
        })

def run_live_inference(audio_array):
    if not AUDIO_LOADED or audio_model is None:
        return "MODEL NOT LOADED", 0.0, 0.0
    try:
        mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=TARGET_SR, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        if mel_db.shape[1] > 259:
            mel_db = mel_db[:, :259]
        else:
            pad_width = 259 - mel_db.shape[1]
            mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        img_array = np.expand_dims(mel_db, axis=(0, -1))
        # Min-max scaling as used in training
        img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-8)
        
        probs = audio_model.predict(img_array, verbose=0)[0]
        # Class Mapping from LabelEncoder in training script
        classes = ["Normal breathing", "Crackle detected", "Wheeze detected", "Both (Crackle+Wheeze)"]
        idx = int(np.argmax(probs))
        abnormal_prob = float(max(probs[1:]))
        
        return classes[idx], float(probs[idx]), abnormal_prob
    except Exception as e:
        print(f"Live inference error: {e}")
        return f"Error", 0.0, 0.0

if __name__ == '__main__':
    # Use socketio.run instead of app.run to support WebSocket upgrades
    socketio.run(app, host='127.0.0.1', port=5001, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)

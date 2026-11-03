import os
import numpy as np

try:
    import tensorflow as tf
    import librosa
    TF_LIBROSA_AVAILABLE = True
except ImportError:
    TF_LIBROSA_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'audio_crnn.keras')

if TF_LIBROSA_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        MODEL_LOADED = True
    except Exception:
        model = None
        MODEL_LOADED = False
else:
    model = None
    MODEL_LOADED = False

def predict_audio(filepath):
    """
    Returns the maximum probability of an adventitious sound (Crackle, Wheeze, Both)
    Assumes index 0 is Normal, index 1,2,3 are abnormal.
    """
    if not MODEL_LOADED or not TF_LIBROSA_AVAILABLE:
        return None
        
    try:
        y, sr = librosa.load(filepath, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        if mel_db.shape[1] > 259:
            mel_db = mel_db[:, :259]
        else:
            pad_width = 259 - mel_db.shape[1]
            mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        img_array = np.expand_dims(mel_db, axis=(0, -1))
        # Normalize (0 to 1) as used in training
        img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-8)
        
        probs = model.predict(img_array)[0]
        # Max of Crackle, Wheeze, Both
        abnormal_prob = max(probs[1:])
        return float(abnormal_prob)
    except Exception as e:
        return f"Audio error: {str(e)}"

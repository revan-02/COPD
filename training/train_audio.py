"""
Respiratory Audio CRNN Training Script (ICBHI 2017)
-----------------------------------------------------
Builds a CNN+BiLSTM (CRNN) model for classifying breathing cycles into
Normal / Crackle / Wheeze / Both, using Mel-Spectrograms.

Differences from a plain CNN approach:
  1. Architecture: Conv2D blocks extract local time-frequency patterns,
     then the feature maps are reshaped into a time sequence and fed
     into a Bidirectional LSTM, which models how crackle/wheeze energy
     evolves across the breathing cycle (a pure CNN with global pooling
     discards this temporal ordering).
  2. Class imbalance: ICBHI cycles are dominated by "Normal" -> we
     compute class weights and pass them to model.fit instead of
     training on the raw (skewed) distribution.
  3. Augmentation: SpecAugment-style time/frequency masking is applied
     to the training spectrograms only, which helps a lot given the
     dataset only has ~126 patients even after a strict patient-level
     split.
  4. Patient-level split: kept identical to the original script
     (GroupShuffleSplit on patient_id) because that part was already
     correct and is essential to avoid leakage.

Requirements: tensorflow, librosa, pandas, scikit-learn, matplotlib,
seaborn, kagglehub

SETUP INSTRUCTIONS
==================

Option A — Kaggle Notebook (easiest, free GPU, no credentials needed):
  1. Go to kaggle.com -> Create -> New Notebook.
  2. In the notebook settings (right sidebar), under "Accelerator" pick
     GPU T4 x2 (or any available GPU) for much faster training.
  3. Add the dataset: Add-ons / "Add Input" -> search for
     "Respiratory Sound Database" (vbookshelf/respiratory-sound-database)
     -> add it. This makes kagglehub.dataset_download() resolve instantly
     from local cache instead of re-downloading.
  4. Upload/paste this script into a notebook cell (or File -> Upload)
     and run it. kagglehub, tensorflow, librosa, pandas, scikit-learn are
     all pre-installed in the default Kaggle Python environment.
  5. When training finishes, download best_crnn_model.keras and
     crnn_confusion_matrix.png from the notebook's Output panel
     (top-right "Output" tab in the session).

Option B — Google Colab (also free GPU):
  1. Go to colab.research.google.com -> New Notebook.
  2. Runtime -> Change runtime type -> select a GPU (T4).
  3. Install/auth for Kaggle in the first cell:
         !pip install -q kagglehub
         import kagglehub
         kagglehub.login()   # opens a prompt for your Kaggle API token
     (Get a token from kaggle.com -> Account -> "Create New API Token",
     which downloads kaggle.json; paste its contents when prompted, or
     upload kaggle.json via the Colab file browser and set
     KAGGLE_CONFIG_DIR accordingly.)
  4. Install the remaining libraries:
         !pip install -q librosa
     (tensorflow, pandas, scikit-learn, matplotlib, seaborn are
     preinstalled on Colab.)
  5. Paste this script into a cell and run it. Download the .keras
     model and confusion matrix PNG from the Colab file browser
     (folder icon on the left) when it finishes.

Either way, training the full dataset (several thousand cycles) will
take roughly 20-60 minutes on a free GPU, depending on batch size and
how many epochs EarlyStopping allows before stopping.
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout,
    Reshape, Bidirectional, LSTM, Dense, GlobalAveragePooling1D
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# ==========================================
# 1. SETUP & DATA PATHS
# ==========================================
print("Locating ICBHI dataset...")
KAGGLE_DATA_DIR = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")

AUDIO_FILES = glob.glob(os.path.join(KAGGLE_DATA_DIR, '**', '*.wav'), recursive=True)
TXT_FILES = glob.glob(os.path.join(KAGGLE_DATA_DIR, '**', '*.txt'), recursive=True)
TXT_FILES = [f for f in TXT_FILES if os.path.basename(f)[0].isdigit()]
AUDIO_FILES.sort()
TXT_FILES.sort()

print(f"Found {len(AUDIO_FILES)} audio files and {len(TXT_FILES)} annotation files.")

# ==========================================
# 2. AUDIO SEGMENTATION
# ==========================================
TARGET_SAMPLE_RATE = 16000
N_MELS = 128
MAX_PAD_LEN = 259  # ~4-second slices

cycles_data = []

print("Extracting individual breathing cycles (this will take a few minutes)...")
for txt_path, wav_path in zip(TXT_FILES, AUDIO_FILES):
    patient_id = os.path.basename(txt_path).split('_')[0]
    y, sr = librosa.load(wav_path, sr=TARGET_SAMPLE_RATE)
    annotations = pd.read_csv(txt_path, sep='\t', header=None,
                               names=['start', 'end', 'crackle', 'wheeze'])

    for _, row in annotations.iterrows():
        start_time, end_time = row['start'], row['end']
        crackle, wheeze = row['crackle'], row['wheeze']

        if crackle == 0 and wheeze == 0:
            cycle_class = 'Normal'
        elif crackle == 1 and wheeze == 0:
            cycle_class = 'Crackle'
        elif crackle == 0 and wheeze == 1:
            cycle_class = 'Wheeze'
        else:
            cycle_class = 'Both'

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio_slice = y[start_sample:end_sample]

        mel_spec = librosa.feature.melspectrogram(y=audio_slice, sr=sr, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if mel_spec_db.shape[1] > MAX_PAD_LEN:
            mel_spec_db = mel_spec_db[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')

        cycles_data.append({
            'patient_id': patient_id,
            'spectrogram': mel_spec_db,
            'label': cycle_class
        })

df_cycles = pd.DataFrame(cycles_data)
print(f"Successfully extracted {len(df_cycles)} respiratory cycles.")

# ==========================================
# 3. PATIENT-LEVEL SPLITTING (no leakage)
# ==========================================
le = LabelEncoder()
df_cycles['label_enc'] = le.fit_transform(df_cycles['label'])
NUM_CLASSES = len(le.classes_)

gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_val_idx, test_idx = next(gss_test.split(df_cycles, groups=df_cycles['patient_id']))
df_train_val = df_cycles.iloc[train_val_idx].reset_index(drop=True)
df_test = df_cycles.iloc[test_idx].reset_index(drop=True)

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=42)
train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['patient_id']))
df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}")

# ==========================================
# 4. AUGMENTATION (SpecAugment-style, train only)
# ==========================================
def spec_augment(mel_db, num_time_masks=2, num_freq_masks=2,
                  time_mask_width=25, freq_mask_width=15):
    aug = mel_db.copy()
    n_mels, n_frames = aug.shape
    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_mask_width)
        f0 = np.random.randint(0, max(1, n_mels - f))
        aug[f0:f0 + f, :] = aug.min()
    for _ in range(num_time_masks):
        t = np.random.randint(0, time_mask_width)
        t0 = np.random.randint(0, max(1, n_frames - t))
        aug[:, t0:t0 + t] = aug.min()
    return aug

# ==========================================
# 5. PREPARE TENSORS
# ==========================================
def format_tensor(df, augment=False, augment_copies=1):
    specs, labels = [], []
    for _, row in df.iterrows():
        specs.append(row['spectrogram'])
        labels.append(row['label_enc'])
        if augment:
            for _ in range(augment_copies):
                specs.append(spec_augment(row['spectrogram']))
                labels.append(row['label_enc'])

    X = np.array(specs)
    X = np.expand_dims(X, axis=-1)
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    y = to_categorical(labels, num_classes=NUM_CLASSES)
    return X, y

X_train, y_train = format_tensor(df_train, augment=True, augment_copies=1)
X_val, y_val = format_tensor(df_val)
X_test, y_test = format_tensor(df_test)

# Class weights to counter the Normal-heavy imbalance
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df_train['label_enc']),
    y=df_train['label_enc']
)
class_weights = dict(enumerate(class_weights_arr))
print("Class weights:", dict(zip(le.classes_, class_weights_arr)))

# ==========================================
# 6. CRNN ARCHITECTURE (CNN -> BiLSTM)
# ==========================================
inputs = Input(shape=(N_MELS, MAX_PAD_LEN, 1))

x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)

# Collapse the frequency axis, keep time axis as a sequence for the RNN
shape_before = x.shape  # (batch, freq, time, channels)
x = Reshape((shape_before[2], shape_before[1] * shape_before[3]))(x)

x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalAveragePooling1D(name="audio_embedding")(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.4)(x)
outputs = Dense(NUM_CLASSES, activation='softmax', name="audio_probability")(x)

model = Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# 7. TRAINING
# ==========================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    tf.keras.callbacks.ModelCheckpoint('best_crnn_model.keras', save_best_only=True)
]

print("Starting CRNN Training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==========================================
# 8. EVALUATION
# ==========================================
print("\n--- Evaluating on Test Set ---")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Respiratory Sound CRNN Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('crnn_confusion_matrix.png')
print("Saved confusion matrix as crnn_confusion_matrix.png")
print("\nTraining complete! Download best_crnn_model.keras to your local machine.")

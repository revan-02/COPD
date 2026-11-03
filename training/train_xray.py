"""
Kaggle / Google Colab Training Script for Chest X-Ray CNN (Emphysema Proxy for COPD)

Instructions for running in Kaggle:
1. Create a new Kaggle Notebook.
2. Click "Add Data" and search for "NIH Chest X-ray Dataset". Add it.
3. Paste this code into a cell and run it!
4. When finished, download 'xray_cnn_model.keras' to your local computer.
"""

import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. SETUP & DATA PATHS
# ==========================================
# Adjust this path based on the Kaggle input environment
KAGGLE_DATA_DIR = '/kaggle/input/nih-chest-xrays/data' 
CSV_PATH = os.path.join(KAGGLE_DATA_DIR, 'Data_Entry_2017.csv')

# The images are often distributed across multiple folders in the Kaggle dataset
ALL_IMAGE_PATHS = glob.glob(os.path.join(KAGGLE_DATA_DIR, 'images_*', 'images', '*.png'))
image_path_map = {os.path.basename(p): p for p in ALL_IMAGE_PATHS}

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

print(f"Found {len(image_path_map)} images in directory.")

# ==========================================
# 2. LOAD & PREPROCESS METADATA
# ==========================================
df = pd.read_csv(CSV_PATH)
df['Image_Path'] = df['Image Index'].map(image_path_map)
df = df.dropna(subset=['Image_Path']) # Remove missing images

# Create our Target Label: "Emphysema" (Our proxy for COPD)
df['Target'] = df['Finding Labels'].apply(lambda x: 1 if 'Emphysema' in x else 0)

# Convert integer target to string for ImageDataGenerator
df['Target_Str'] = df['Target'].astype(str)

print(f"Total dataset size: {len(df)}")
print(f"Positive samples (Emphysema): {df['Target'].sum()}")

# ==========================================
# 3. STRICT PATIENT-LEVEL SPLITTING (DATA LEAKAGE PREVENTION)
# ==========================================
# We must split by Patient ID so a patient's images don't bleed across sets
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_val_idx, test_idx = next(gss_test.split(df, groups=df['Patient ID']))

df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=42) # ~15% of total
train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['Patient ID']))

df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}")

# ==========================================
# 4. CLASS BALANCING (Class Weights)
# ==========================================
neg, pos = np.bincount(df_train['Target'])
total = neg + pos
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}
print(f"Class weights: 0: {weight_for_0:.2f}, 1: {weight_for_1:.2f}")

# ==========================================
# 5. DATA GENERATORS
# ==========================================
# Minimal augmentation for medical images
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False # Hearts should stay on the left!
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_dataframe(
    dataframe=df_train, x_col='Image_Path', y_col='Target_Str',
    target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=True
)

val_gen = test_datagen.flow_from_dataframe(
    dataframe=df_val, x_col='Image_Path', y_col='Target_Str',
    target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=df_test, x_col='Image_Path', y_col='Target_Str',
    target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)

# ==========================================
# 6. MODEL ARCHITECTURE (TRANSFER LEARNING)
# ==========================================
# DenseNet121 is the industry standard baseline for Chest X-rays
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers initially
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
# This pooling layer acts as our "xray_embedding" for the multimodal fusion later!
xray_embedding = GlobalAveragePooling2D(name="xray_embedding")(x)
x = Dropout(0.5)(xray_embedding)
predictions = Dense(1, activation='sigmoid', name="xray_probability")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# ==========================================
# 7. TRAINING
# ==========================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=4, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_xray_model.keras', save_best_only=True, monitor='val_auc', mode='max')
]

print("Starting Training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)

model.save('final_xray_model.keras')

# ==========================================
# 8. EVALUATION & METRICS (NO FABRICATION)
# ==========================================
print("\n--- Evaluating on Test Set ---")
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

auc = roc_auc_score(y_true, y_pred_probs)
acc = accuracy_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0
specificity = tn / (tn + fp) if (tn+fp) > 0 else 0

print(f"ROC-AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Save Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues')
plt.title('Chest X-ray CNN Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('xray_confusion_matrix.png')
print("Saved confusion matrix as xray_confusion_matrix.png")
print("\nTraining complete! Download final_xray_model.keras to your local machine.")

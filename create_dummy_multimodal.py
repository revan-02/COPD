import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Create Dummy X-Ray Model (Expects 224x224x3, outputs 1 probability)
xray_model = Sequential([
    Input(shape=(224, 224, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
])
xray_model.compile(optimizer='adam', loss='binary_crossentropy')
xray_model.save('best_xray_model.keras')
print("Created dummy best_xray_model.keras")

# Create Dummy CRNN Model (Expects 128x259x1, outputs 4 probabilities for Normal/Crackle/Wheeze/Both)
audio_model = Sequential([
    Input(shape=(128, 259, 1)),
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(4, activation='softmax')
])
audio_model.compile(optimizer='adam', loss='categorical_crossentropy')
audio_model.save('best_crnn_model.keras')
print("Created dummy best_crnn_model.keras")

# Create Dummy Logistic Regression Fusion Model
import joblib
from sklearn.linear_model import LogisticRegression

# Features: [Clinical_Prob, Xray_Prob, Audio_Prob_Crackle, Audio_Prob_Wheeze, Audio_Prob_Both]
X_dummy = np.random.rand(100, 5)
y_dummy = np.random.randint(0, 2, 100)

fusion_model = LogisticRegression()
fusion_model.fit(X_dummy, y_dummy)
joblib.dump(fusion_model, 'fusion_meta_model.joblib')
print("Created dummy fusion_meta_model.joblib")

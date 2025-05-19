# AI-Quality Control in Manufacturing - Single File Version

import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime

# Load a simulated model
def load_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(64, 64, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

# Predict defect using the model
def predict_defect(model, frame):
    resized = cv2.resize(frame, (64, 64))
    input_tensor = np.expand_dims(resized / 255.0, axis=0)
    prediction = model.predict(input_tensor)[0]
    defect = prediction[1] > 0.5  # Class 1 = Defect
    confidence = prediction[1]
    return defect, confidence

# Log detected defects
def log_defect(confidence):
    with open("defect_log.txt", "a") as f:
        f.write(f"{datetime.now()}, Defect detected, Confidence: {confidence:.2f}\n")

# Main inspection loop
def main():
    print("Loading model...")
    model = load_model()

    print("Starting real-time inspection...")
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video path

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        defect, confidence = predict_defect(model, frame)
        if defect:
            print(f"Defect detected! Confidence: {confidence:.2f}")
            log_defect(confidence)
            cv2.putText(frame, "DEFECT DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "OK", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Quality Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

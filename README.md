
# 🛠️ AI-Powered Quality Control System

This project demonstrates a simple real-time quality control system using computer vision and a neural network to detect visual defects in manufactured products.

## 📸 Features

- Real-time video inspection via webcam
- Simple neural network for defect classification
- On-screen display with defect alerts
- Logging of defect events with timestamps
- Includes mock images and simulation outputs

## 📁 Files Included

- `ai_quality_control_single.py` – Main Python script
- `defect_log.txt` – Automatically generated when defects are detected
- `output.png` – Console Output of the defect 

## ⚙️ Requirements

- Python 3.7+
- Libraries:
  - `opencv-python`
  - `numpy`
  - `tensorflow`
  - `matplotlib` (for simulation images)
  - `Pillow` (for console simulation)

Install dependencies with:

```bash
pip install opencv-python numpy tensorflow matplotlib pillow
````

## 🚀 How to Run

### Live Inspection via Webcam

```bash
python ai_quality_control_single.py
```

Press `q` to exit the video stream.

### Test with a Mock Image

```python
from ai_quality_control_single import load_model, predict_defect
import cv2

model = load_model()
image = cv2.imread("mock_defect.png")  # or mock_non_defect.png
defect, confidence = predict_defect(model, image)
print(f"Defect: {defect}, Confidence: {confidence:.2f}")
```


## Console Output Image
![console output](output.png)


## 📌 Notes

* The model is randomly initialized (not trained), so predictions are **random** by default.
* Replace or train the model with your own dataset for meaningful results.

## 📈 Future Enhancements

* Train and integrate a real model using labeled defect data
* Add GUI for monitoring and controls
* Export results as reports



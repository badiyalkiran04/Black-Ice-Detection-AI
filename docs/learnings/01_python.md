# 🐍 Python for AI Development

> Project: AI-powered Black Ice Detection Rover

# Why Python?

Python is the primary programming language used in Artificial Intelligence, Machine Learning, Computer Vision, Robotics, and Data Science.

For this project, Python will be used to:

- Read images from cameras
- Process polarization images
- Build the physics-based detection model
- Train the CNN model
- Fuse sensor data
- Deploy the AI model on Raspberry Pi / Jetson Nano

# Why Python for This Project?

Our rover software will be written almost entirely in Python because it has powerful AI libraries such as:

- NumPy
- OpenCV
- PyTorch
- TensorFlow
- Scikit-learn
- Matplotlib

These libraries make image processing and AI development much easier.

# Python Concepts Required

Instead of learning every Python feature, I only need the concepts required for AI development.

## Variables

Variables store data.

Example:

```python
temperature = -3.5
road_type = "Ice"
confidence = 0.92
```

## Data Types

| Data Type | Example      | Used In              |
|-----------|--------------|----------------------| 
| int       | 5            | Image dimensions     |
| float     | 0.85         | CNN confidence score |
| str       | "Black Ice"  | Labels               |
| bool      | True         | Ice detected?        |
| list      | [1,2,3]      | Pixel values         |
| tuple     | (640,480)    | Image size           |
| dict      | {"temp": -5} | Sensor information   |


## Operators

Arithmetic

```python
+
-
*
/
%
```

Comparison

```python
>
<
==
!=
>=
<=
```

Logical

```python
and
or
not
```


## If Statements

Used to make decisions.

Example:

```python
if temperature < 0:
    print("Possible Ice")
else:
    print("No Ice")
```

Application:

The rover decides whether a detected region could contain black ice.

## Loops

### For Loop

```python
for i in range(5):
    print(i)
```

Used for:

- Processing images
- Reading datasets
- Training loops

---

### While Loop

```python
while True:
    capture_image()
```

Useful for continuous camera processing.

---

## Functions

Functions make code reusable.

```python
def calculate_difference(img1, img2):
    return img1 - img2
```

Our project will contain many functions such as:

- capture_image()
- preprocess_image()
- detect_black_ice()
- predict_cnn()

---

## Lists

Store multiple values.

```python
temperatures = [-5,-3,-1]
```

Useful for:

- Temperature history
- Sensor readings
- Pixel values

---

## Dictionaries

Store key-value pairs.

```python
sensor = {
    "temperature": -5,
    "speed": 2.3,
    "pitch": 8
}
```

Useful for storing sensor data.

---

## Classes (OOP)

Large AI projects use Object-Oriented Programming.

Example:

```python
class Camera:

    def capture(self):
        pass
```

Possible classes in this project:

- Camera
- TemperatureSensor
- CNNModel
- PhysicsModel
- HybridModel

---

## File Handling

Reading files.

```python
with open("notes.txt","r") as file:
    data = file.read()
```

Useful for:

- Reading configuration files
- Loading datasets
- Saving predictions

---

## Exception Handling

Used to prevent crashes.

```python
try:
    image = load_image()
except:
    print("Image not found")
```

Important for robotics because sensors may fail.

---

## Modules

Python code is separated into modules.

Example

```python
import cv2
import numpy as np
```

---

# Python Libraries Required

## NumPy

Purpose

Numerical computation

Used for

- Images
- Matrix operations
- Pixel calculations

---

## OpenCV

Purpose

Computer Vision

Used for

- Reading images
- Image filtering
- Polarization difference
- Thresholding

---

## PyTorch

Purpose

Deep Learning

Used for

- CNN training
- Model inference

---

## Matplotlib

Purpose

Visualization

Used for

- Accuracy graphs
- Loss curves
- Detection visualization

# Real Application in Our Rover

Python will control the following pipeline:

Camera Images
↓

OpenCV Processing
↓

Polarization Difference

↓

Temperature Sensor Reading

↓

CNN Prediction

↓

Physics Model

↓

Hybrid AI Model

↓

Final Decision

↓

Black Ice Detected

---

# Key Takeaways

- Python is the foundation of the entire software system.
- Every AI module in this project will be written in Python.
- OpenCV, NumPy, and PyTorch are the most important libraries.
- Clean, modular code will make the project easier to develop and debug.
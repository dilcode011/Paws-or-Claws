#  Dog vs Cat Classifier (CNN)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

A Deep Learning project that distinguishes between images of dogs and cats using a **Convolutional Neural Network (CNN)**. Built from scratch using TensorFlow and Keras, this model is trained on the Kaggle Dogs vs. Cats dataset.

##  Project Overview
The goal of this project is to build a binary image classifier capable of identifying whether an input image contains a dog or a cat. The project handles the entire pipeline:
1.  **Data Ingestion:** Fetching the dataset programmatically via the Kaggle API.
2.  **Preprocessing:** Rescaling images (normalization) and resizing them to `256x256` pixels.
3.  **Modeling:** Designing a custom CNN architecture to extract spatial features.
4.  **Evaluation:** Analyzing training vs. validation accuracy to check for overfitting.
5.  **Prediction:** Testing the model on real-world, unseen images.

##  Dataset
* **Source:** [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/datasets/salader/dogsvscats)
* **Structure:**
    * `train/`: 20,000 images (10,000 Dogs, 10,000 Cats)
    * `test/`: 5,000 images (2,500 Dogs, 2,500 Cats)
* **Input Shape:** `(256, 256, 3)` (RGB Images)

## 🛠️ Tech Stack
* **Core:** Python
* **Deep Learning Framework:** TensorFlow, Keras
* **Computer Vision:** OpenCV (`cv2`)
* **Data Handling:** NumPy, Pandas
* **Visualization:** Matplotlib

##  Model Architecture
The model uses a sequential CNN design optimized for feature extraction:

| Layer Type | Specifications | Purpose |
| :--- | :--- | :--- |
| **Conv2D** | 32 filters, 3x3 kernel, ReLU | Basic feature detection (edges, colors) |
| **MaxPooling2D** | 2x2 pool size | Reduces spatial dimensions |
| **Conv2D** | 64 filters, 3x3 kernel, ReLU | Complex feature detection (textures) |
| **MaxPooling2D** | 2x2 pool size | Reduces spatial dimensions |
| **Conv2D** | 128 filters, 3x3 kernel, ReLU | High-level feature detection (shapes) |
| **BatchNormalization** | - | Stabilizes learning and speeds up convergence |
| **Flatten** | - | Converts 2D maps to 1D vector |
| **Dense** | 128 neurons, ReLU | Fully connected classification layer |
| **Dropout** | 0.1 rate | Prevents overfitting by randomly dropping neurons |
| **Dense (Output)** | 1 neuron, Sigmoid | Outputs probability (0 = Cat, 1 = Dog) |

##  Installation & Setup
This project is designed to run in **Google Colab** or a local Jupyter Notebook environment.

### Prerequisites
* A Kaggle account and a `kaggle.json` API token.

### Steps
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/dilcode011/dog-vs-cat-cnn.git](https://github.com/dilcode011/dog-vs-cat-cnn.git)
    cd dog-vs-cat-cnn
    ```

2.  **Install Dependencies**
    ```bash
    pip install tensorflow opencv-python matplotlib
    ```

3.  **Setup Kaggle API**
    * Place your `kaggle.json` file in the root directory.
    * The notebook will automatically move it to `~/.kaggle/` to authenticate the download.

4.  **Run the Notebook**
    Open `DOGvsCAT.ipynb` and execute the cells sequentially.

## 📷 Sample Predictions
The notebook includes a testing phase where the model validates its learning on real-world unseen images. Two specific test cases were used to verify binary classification:

```python
import cv2
import matplotlib.pyplot as plt

# 1. Test on a Dog Image
test_img_dog = cv2.imread('dog.webp')
test_input_dog = cv2.resize(test_img_dog, (256,256)).reshape((1,256,256,3))
model.predict(test_input_dog)  
# Result: High Probability (> 0.5) -> Classified as DOG

# 2. Test on a Cat Image
test_img_cat = cv2.imread('cat.webp')
test_input_cat = cv2.resize(test_img_cat, (256,256)).reshape((1,256,256,3))
model.predict(test_input_cat)  
# Result: Low Probability (< 0.5) -> Classified as CAT

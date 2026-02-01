#  Dog vs Cat Image Classifier

A Deep Learning project that classifies images as either a **Dog** or a **Cat** using a Convolutional Neural Network (CNN). Built with TensorFlow and Keras, this model is trained on the Kaggle Dogs vs. Cats dataset.

##  Project Overview
This project demonstrates the end-to-end process of building a binary image classifier. It covers:
* **Data Handling:** Fetching data directly from Kaggle and preprocessing images (resizing, normalization).
* **Model Architecture:** Building a custom CNN from scratch with Convolutional, Pooling, and Dense layers.
* **Evaluation:** Analyzing training performance and checking for overfitting using Matplotlib.
* **Inference:** Predicting the class of new, unseen images.

##  Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV (`cv2`)
* **Visualization:** Matplotlib
* **Data Source:** Kaggle API

##  Model Architecture
The model is a Sequential CNN designed to extract features and perform binary classification:

1.  **Convolutional Blocks:**
    * 3 layers of **Conv2D** (32, 64, 128 filters) with **ReLU** activation.
    * **MaxPooling2D** to reduce spatial dimensions.
    * **BatchNormalization** for faster convergence.
2.  **Dense Layers:**
    * Flatten layer to convert 2D maps to 1D vectors.
    * Fully connected layers (128 & 64 neurons).
    * **Dropout (0.1)** to reduce overfitting.
3.  **Output Layer:**
    * 1 neuron with **Sigmoid** activation (outputs probability between 0 and 1).

##  Results
* **Training Accuracy:** ~95%
* **Validation Accuracy:** ~83%
* **Observation:** The model learns the training data very well. The gap between training and validation accuracy suggests some overfitting, which could be improved in future versions using Data Augmentation.

##  How to Run
This project is optimized for **Google Colab**.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repo-Name].git
    ```
2.  **Open in Colab:**
    Upload the `DOGvsCAT.ipynb` file to Google Colab.
3.  **Setup Kaggle API:**
    * You will need your own `kaggle.json` API key.
    * Upload `kaggle.json` to the Colab session when prompted or manually place it in the root directory.
4.  **Run All Cells:**
    Execute the cells to download the dataset, train the model, and view predictions.

##  Sample Prediction
The notebook includes code to test the model on real-world images:
```python
import cv2
test_img = cv2.imread('dog.webp')
# ... preprocessing steps ...
prediction = model.predict(test_input) # Output > 0.5 indicates Dog

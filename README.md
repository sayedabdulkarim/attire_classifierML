# **Attire Class Classification**

This project is a **deep learning model** developed to classify images of fashion items from the **Fashion MNIST dataset**. The dataset consists of grayscale images of 10 different types of clothing items such as T-shirts, trousers, dresses, etc. The model utilizes **Convolutional Neural Networks (CNNs)** to achieve this task with a high level of accuracy.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Installation and Setup](#installation-and-setup)
4. [Model Training](#model-training)
5. [Evaluation and Results](#evaluation-and-results)
6. [Visualization](#visualization)
7. [Future Work](#future-work)
8. [License](#license)

---

## **Project Overview**

The aim of this project is to create a classifier that can accurately predict the type of fashion item in a given image. We use the **Fashion MNIST dataset**, which includes 70,000 images of clothing items belonging to 10 different classes. The images are 28x28 pixels in grayscale, and each image is labeled according to its class.

The dataset is divided into:
- **60,000 training samples**
- **10,000 testing samples**

We employ a **Convolutional Neural Network (CNN)** for image classification, which is trained on the 60,000 images and evaluated on the remaining 10,000 images.

---

## **Dataset Description**

The dataset includes **10 classes**, which are as follows:
- **0:** T-shirt/top
- **1:** Trouser
- **2:** Pullover
- **3:** Dress
- **4:** Coat
- **5:** Sandal
- **6:** Shirt
- **7:** Sneaker
- **8:** Bag
- **9:** Ankle boot

Each image is a 28x28 pixel grayscale image, resulting in 784 pixels per image. Each pixel has an integer value between **0** (white) and **255** (black), which represents the intensity of the pixel.

---

## **Installation and Setup**

### **Requirements:**
- Python 3.x
- TensorFlow / Keras
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### **Steps to Install:**

1. Clone the repository:
    ```bash
    git clone https://github.com/sayedabdulkarim/attire_classifierML.git
    cd attire_classifier
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the Fashion MNIST dataset from the csv attached.

4. Run the Jupyter notebook to train the model:
    ```bash
    jupyter notebook
    ```
   
5. If running outside of Jupyter, execute the script:
    ```bash
    python app.py
    ```

---

## **Model Training**

The model is built using **Convolutional Neural Networks (CNN)**, which are effective for image classification tasks. The architecture consists of:

- **Conv2D layers**: To extract features from the input image.
- **MaxPooling2D layers**: For downsampling the feature maps.
- **Dropout**: To reduce overfitting.
- **Flatten**: To convert the 2D feature maps into a 1D vector.
- **Dense (Fully connected) layers**: For the final classification.

The model is compiled using the **Adam optimizer** and the **sparse categorical cross-entropy loss function**. The model is trained for 50 epochs with a batch size of 512.

---

## **Evaluation and Results**

After training the model, it is evaluated on the **test dataset**. The final test accuracy is printed, and a **confusion matrix** is plotted to analyze the performance across different classes.

### **Test Accuracy:**  
The test accuracy is calculated using the `evaluate()` function, and the result is printed as follows:
```
Test Accuracy: 0.XXX
```

### **Confusion Matrix:**  
The confusion matrix shows the number of correct and incorrect predictions for each class. A heatmap is generated using **Seaborn** to visualize this.

### **Classification Report:**
The classification report is printed to show **precision**, **recall**, and **F1-score** for each class.

---

## **Visualization**

Some visualizations included in this project:
- **Single Image Visualization:** Displays a random image from the dataset along with its label.
- **Grid of Images:** Displays a grid of images from the dataset to provide a better understanding of the variety in the dataset.
- **Confusion Matrix:** Displays the confusion matrix to show classification performance for each class.

---

## **Future Work**

- **Improve Accuracy:** Tune the hyperparameters and experiment with deeper CNN architectures to improve the accuracy further.
- **Deploy the Model:** Create an API to serve the model predictions using frameworks such as **Flask** or **FastAPI**.
- **Real-time Predictions:** Develop a real-time prediction system where users can upload images and get predictions.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

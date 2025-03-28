# Brain-Tumor-Detection-and-Classification-by-CNN

**Motivation:**

The objective of this study is to develop an automated model for brain tumor detection to assist healthcare professionals in providing quicker and more accurate diagnoses. The proposed system aims to reduce human errors and improve the overall effectiveness of medical imaging interpretations.

 - Importance of Early Diagnosis:
Early diagnosis of brain tumors significantly increases the chances of successful treatment and patient survival. By leveraging artificial intelligence (AI) and deep learning techniques, timely detection can be achieved, which helps in minimizing the risks associated with brain tumors.

 - Limitations of Manual Analysis:
Manual analysis of MRI images by healthcare professionals is a time-consuming and error-prone task. The interpretation of complex medical images, especially when tumors are small or in difficult-to-interpret locations, can lead to inaccurate results. This presents a strong motivation for automating the diagnostic process using machine learning.

 - Deep Learning Potential:
Deep learning models, particularly Convolutional Neural Networks (CNNs), have demonstrated outstanding performance in various image classification tasks, including medical image analysis. CNNs have the capability to learn and recognize intricate patterns in images, making them a suitable choice for brain tumor detection from MRI scans.

**Research Questions:**

The study focuses on the following research questions:

 - How can a Convolutional Neural Network (CNN) be optimized for detecting brain tumors from MRI images?

 - What impact does the quality and resolution of MRI images have on the performance of CNN models in brain tumor detection and classification?

 - How do pre-trained deep learning models trained on color images perform when applied to grayscale MRI image classification tasks?

**Data:**

The dataset used for this research is publicly available and obtained from Kaggle. It contains 4,600 labeled MRI scans, which are divided into two groups:

 - 2,513 images representing brain tumors.

 - 2,087 images representing healthy brains.

 - These images are in JPEG format, and the labels indicate the presence or absence of a brain tumor.

**Data Pre-processing:**
To ensure that the model receives uniform and high-quality input, several pre-processing steps were applied to the data:

 - Image Format and Color Mode Standardization:
All MRI images were standardized to the JPEG format and converted to the RGB color mode. This ensures that the images are in a consistent format and that the CNN receives appropriate input.

 - Image Resizing:
The MRI images were resized to a fixed size (e.g., 224x224 pixels) to ensure uniformity in input dimensions and to fit the CNN architecture.

 - Pixel Value Normalization:
Pixel values were normalized to a range of [0, 1] to improve the convergence of the training process.

 - Data Augmentation:
To prevent overfitting and improve generalization, data augmentation was performed on the training set. This involved random transformations, such as rotation, shifting, and zooming, to artificially increase the diversity of the training data.

 - Dataset Splitting:
The dataset was split into training and validation sets. The training set is used to train the model, while the validation set is used for evaluating the model's performance.

**Model Architecture:**
Base Model – Convolutional Neural Network (CNN):

The base model used for brain tumor detection consists of a relatively simple CNN architecture designed to learn relevant features from MRI images:

 - Convolutional Blocks:
The network has two convolution blocks, with two convolution layers in each block (total of 4 convolution layers).

 - Max Pooling Layers:
After each convolution block, a max pooling layer was added to reduce the spatial dimensions of the feature maps.

 - Dropout Layers:
Dropout layers were added after each convolution block to prevent overfitting by randomly setting a fraction of input units to zero during training.

 - Flattening and Dense Layer:
After the convolution and pooling layers, the feature maps are flattened, and a fully connected dense layer follows to generate the final output.

 - Activation Functions:
ReLU (Rectified Linear Unit) activation functions were used for the convolution and dense layers, allowing the model to learn non-linear relationships. The output layer used the Sigmoid activation function to classify the images into two categories: tumor or no tumor.

 - Optimizer:
The Adam optimizer was used for efficient weight updates during training.

 - Loss Function:
Binary crossentropy loss was used, as the classification is binary (tumor vs. no tumor).

 - Evaluation Metric:
Accuracy was used as the evaluation metric to assess the model's performance in distinguishing between brain tumor and healthy images.

**Final Model – Custom CNN:**

After evaluating the performance of the base model, a custom CNN architecture was designed to improve the detection accuracy and model robustness:

 - Convolutional Blocks:
The final model used four convolutional blocks, each containing one convolution layer (total of 4 convolution layers).

 - Max Pooling Layers:
Four max pooling layers were included, one after each convolution layer, to further reduce spatial dimensions and focus on important features.

 - Dropout Layer:
A dropout layer was added after the fully connected dense layer to reduce overfitting.

 - Flattening and Dense Layer:
After the convolutional and pooling layers, the data is flattened, and a fully connected dense layer generates the final output.

 - Activation Functions:
ReLU was used for all convolution and dense layers, and Sigmoid was used for the output layer to classify the images into two categories.

 - Optimizer:
The Adam optimizer was used for faster convergence.

 - Loss Function:
Binary crossentropy loss was used for binary classification.

 - Evaluation Metric:
Accuracy remains the evaluation metric to assess the model’s performance.

**Conclusion:**
This study demonstrates the potential of CNN-based models for automated brain tumor detection from MRI images. By implementing a custom CNN architecture, it is possible to classify MRI scans into two categories: brain tumor or healthy brain. The model has the potential to support healthcare professionals in providing quicker, more reliable diagnoses, especially in settings where expert radiologists may not be available.

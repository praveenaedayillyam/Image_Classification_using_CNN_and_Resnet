# 🖼️ Image Classification Using CNN and Transfer Learning (ResNet50)
- 📌 Project Overview<br>

This project focuses on multi-class image classification using two deep learning approaches: <br>

Custom Convolutional Neural Network (CNN) <br>

Transfer Learning using ResNet50<br>

The goal is to classify natural scene images into six categories using TensorFlow and Keras, and to compare the performance of a custom-built CNN with a pre-trained deep learning model.<br>

- 🧠 What is Transfer Learning?<br>

Transfer Learning is a machine learning technique where a pre-trained model, trained on a large dataset (like ImageNet), is reused for a new but related task.<br>

Instead of training a deep network from scratch, the model leverages previously learned features such as:<br>

edges<br>

textures<br>

shapes<br>

This significantly reduces training time and often improves performance, especially when the dataset is limited.<br>

- 🗂️ Dataset<br>

The dataset consists of natural scene images categorized into six classes:<br>

🏢 buildings<br>

🌲 forest<br>

❄️ glacier<br>

⛰️ mountain<br>

🌊 sea<br>

🛣️ street<br>

Dataset Details <br>

Training images: 14,034<br>

Testing images: 3,000<br>

Image size: 64 × 64 × 3<br>

Format: Directory-based (compatible with ImageDataGenerator)
<br>
- ⚙️ Technologies & Libraries Used<br>

Python<br>

TensorFlow / Keras<br>

NumPy<br>

OpenCV<br>

Matplotlib<br>

Seaborn<br>

Scikit-learn<br>

gdown (Google Drive download)<br>

- 🔄 Data Preprocessing & Augmentation<br>

Rescaled pixel values to [0,1]<br>

Applied data augmentation:<br>

Zoom<br>

Width & height shift<br>

Used ImageDataGenerator for:<br>

Training<br>

Validation<br>

Testing<br>

- 🧩 Models Implemented<br>
🔹 1. Custom CNN Model<br>

Architecture highlights:<br>

Convolution + ReLU<br>

MaxPooling<br>

Batch Normalization<br>

Dropout (to reduce overfitting)<br>

Fully connected dense layers<br>

Softmax output layer (6 classes)<br>

Total Parameters: ~206K<br>
Optimizer: Adam<br>
Loss Function: Categorical Crossentropy<br>

- 🔹 2. Transfer Learning – ResNet50<br>

Key points:<br>

Pre-trained on ImageNet<br>

include_top=False<br>

Global Average Pooling<br>

Fine-tuning enabled<br>

Additional dense layers for classification<br>

Total Parameters: ~23M<br>
Optimizer: Adam<br>
Loss Function: Categorical Crossentropy<br>

- 📊 Model Performance Comparison<br>
✅ CNN Model Performance<br>

Test Accuracy: 75% <br>

Better generalization on this dataset <br>

Stable training & validation curves <br>

Classification Report (CNN): <br>

Accuracy: 0.75 <br>
Macro Avg F1-score: 0.75 <br>
Weighted Avg F1-score: 0.74 <br>

- ⚠️ ResNet50 Model Performance <br>

Test Accuracy: 63% <br>

Overfitting observed <br>

Requires better fine-tuning and learning rate scheduling<br>

Classification Report (ResNet50):<br>

Accuracy: 0.63<br>
Macro Avg F1-score: 0.63<br>
Weighted Avg F1-score: 0.62<br>

- 📈 Visualizations<br>

Sample training images with labels<br>

Training vs Validation Accuracy plots<br>

Prediction vs Actual comparison on test images<br>

Class-wise precision, recall, and F1-score<br>

🧪 Model Testing<br>

Predictions generated on unseen test dataset<br>

Compared predicted labels with actual labels<br>

Evaluated using:<br>

Accuracy<br>

Precision<br>

Recall<br>

F1-score<br>

Confusion Matrix<br>

- 🏁 Conclusion<br>

The Custom CNN outperformed ResNet50 on this dataset<br>

Transfer learning requires careful fine-tuning<br>

Smaller image size (64×64) may limit deep models like ResNet50<br>

CNN proved to be more efficient and stable for this task<br>

- 🚀 Future Improvements<br>

Freeze initial ResNet50 layers and fine-tune selectively<br>

Increase input image resolution (e.g., 128×128)<br>

Apply learning rate scheduling<br>

Try other transfer learning models:<br>

MobileNet <br>

EfficientNet

Deploy model using Streamlit or Flask

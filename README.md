# 🚗 Indonesian License Plate Odd/Even Classifier

A deep learning project that classifies Indonesian license plates as odd or even based on the last digit of the plate number. This system uses a MobileNetV2 transfer learning model to identify the odd/even status directly from license plate images, making it useful for enforcing traffic regulations that restrict vehicle usage based on license plate numbers.

![image](https://github.com/user-attachments/assets/b669098a-923a-4006-9963-3f03ed07390b)


## 📋 Project Overview

In many Indonesian cities, traffic regulations restrict vehicle usage on certain days based on whether the license plate ends in an odd or even number. This project automates the classification process using computer vision and deep learning.

The project consists of two main components:
1. **Dataset Preparation Tool**: Automatically separates mixed image datasets into odd and even license plate folders
2. **Classifier Model**: A CNN model based on MobileNetV2 that predicts whether a license plate is odd or even

## ✨ Features

- **Accurate Classification**: Achieves high accuracy in distinguishing odd/even license plates
- **Transfer Learning**: Uses pre-trained MobileNetV2 for efficient learning with limited data
- **Data Augmentation**: Implements various image augmentations to improve model robustness
- **Automatic Dataset Organization**: Smart regex-based sorting tool for creating labeled datasets
- **License Plate Extraction**: Optional advanced feature to detect and isolate license plates from vehicle images
- **Interactive Testing**: Simple interface for testing the model on custom images

## 🧠 Model Architecture

The classification model is built using transfer learning with MobileNetV2 as the base:

```
MobileNetV2 (pre-trained, feature extraction) 
    → GlobalAveragePooling2D
    → BatchNormalization
    → Dense(128, ReLU)
    → Dropout(0.5)
    → Dense(64, ReLU)
    → Dropout(0.3)
    → Dense(1, Sigmoid)
```

This architecture balances performance and efficiency, making it suitable for deployment in real-world applications.


## 🛠️ Dataset Preparation

The project includes a dataset preparation script that:

1. Processes license plate images from multiple source folders
2. Uses regex patterns to identify the license plate number format
3. Extracts the last digit to determine if it's odd or even
4. Organizes images into separate folders for training

The script handles various Indonesian license plate formats:
- Standard format: `A1234BC`
- Special format: `AD1234IT` 
- Extended format: `351.E 6730 RC-07-19`

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy, Pandas, Matplotlib
- Google Colab (recommended for training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/indonesian-license-plate-classifier.git
cd indonesian-license-plate-classifier
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
- Create folders for your source images (`pics`, `train`, `test`, `dataset`)
- Run the dataset preparation script:
```bash
python prepare_dataset.py --source /path/to/source/folders
```

4. Train the model:
```bash
python train_model.py --data_dir /path/to/processed/data
```

### Using Google Colab

Alternatively, upload the notebooks to Google Colab:
1. Upload your license plate images to Google Drive
2. Mount your Drive in Colab
3. Run the dataset preparation notebook
4. Train the model using the classifier notebook

## 📸 Usage

### Dataset Preparation

```python
# Run this to organize your license plate images
from google.colab import drive
drive.mount('/content/drive')

# Set paths to your image folders
pics_folder = '/content/drive/MyDrive/pics'
train_folder = '/content/drive/MyDrive/train'
test_folder = '/content/drive/MyDrive/test'
dataset_folder = '/content/drive/MyDrive/dataset'

# Execute the preparation script
!python prepare_dataset.py
```

### Classifying License Plates

```python
# Load the trained model
model = tf.keras.models.load_model('/path/to/license_plate_model.h5')

# Test on a custom image
test_on_custom_image(model, '/path/to/your/image.jpg', class_mapping)

# Or use the complete pipeline for vehicle images
process_vehicle_image(model, '/path/to/vehicle/image.jpg', class_mapping)
```

## 🌟 Advanced Features

### License Plate Extraction

The project includes an experimental license plate extraction feature that:
1. Takes a full vehicle image as input
2. Uses contour detection to identify the license plate region
3. Extracts and processes just the plate for classification

Example usage:
```python
plate_img = extract_license_plate('/path/to/vehicle/image.jpg')
predicted_class, confidence = test_on_custom_image(model, plate_img, class_mapping)
```

### Fine-Tuning

For improved accuracy, you can fine-tune the model:
```python
# Unfreeze the top layers of MobileNetV2 and train with a lower learning rate
history_fine, fine_tuned_model = fine_tune_model(model, train_generator, validation_generator)
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgements

- The dataset used for this project is collected from various sources for educational purposes:
https://www.kaggle.com/datasets/firqaaa/indonesian-vehicle-plate-numbers/data
https://www.kaggle.com/datasets/imamdigmi/indonesian-plate-number?resource=download
https://www.kaggle.com/datasets/caasperart/haarcascadeplatenumber

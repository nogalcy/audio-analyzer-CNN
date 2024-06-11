# Image Classification Using Convolutional Neural Networks (CNN)

A deep learning project leveraging CNNs to accurately classify audio files into predefined categories. This project demonstrates the power of deep learning in audio recognition tasks with the help of tranformations and strong preprocessing.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Overview

This project explores the use of Convolutional Neural Networks (CNN) for audio classification tasks. The goal is to classify .wav files into the following 10 categories: 'dog_bark', 'children_playing', 'car_horn', 'air_conditioner', 'street_music',
 'gun_shot', 'siren', 'engine_idling', 'jackhammer', and 'drilling'. Through applying various deep learning methods and model structures, this notebook outlines the complete process of taking raw data, cleaning it, transforming it, creating some (poor) baseline models, evaluating the metrics, and eventually creating a classification network with a 91.5% prediction success rate.

## Dataset

We utilized the [UrbanSound8K-Dataset](https://urbansounddataset.weebly.com/urbansound8k.html), which consists of 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes. UrbanSound8K also provides the 'UrbanSound8k.csv' filled with metadata about each and every one of the 
provided samples. For this project specifically, much of the metadata provided in this .csv is ignored as I want to be able to use these techniques on datasets that are not quite as comprehensive as UrbanSound8K. In fact,
much of the use of the .csv file comes from data discovery and finding quirks in the dataset including imperfect sampling rates and noise lengths that are covered and "fixed" in the data preprocessing portion of the code. Prior to feeding the CNN, the dataset undergoes significant preprocessing including resampling,
padding, resizing, transformation into mel-spectrograms, and (eventually) augmentation to expand the dataset and provide an easier way for the CNN to learn general patterns in an attempt to prevent overfitting.
## Model Architecture

The final CNN model comprises several layers, including:

- **Convolutional Layers**: Extract features from input images.
- **Pooling Layers**: Reduce the dimensionality and retain important features.
- **Fully Connected Layers**: Perform the classification based on extracted features.

The architecture is as follows:
1. Conv2D (64 filters, 3x3) + ReLU
2. Dropout(0.3)
3. Conv2D (32 filters, 3x3) + ReLU
4. Dropout(0.3)
5. MaxPooling2D (2x2)
6. Conv2D (32 filters, 3x3) + ReLU
7. Dropout(0.3)
8. Conv2D (16 filters, 3x3) + ReLU
9. Dropout(0.3)
10. MaxPooling2D (2x2)
11. Flatten
12. Dense (64 units) + ReLU
13. Dense (10 units) + Softmax

## Training

We trained the CNN using the following parameters:
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam(lr = 0.001)
- **Batch Size**: 32
- **Epochs**: 100

During training, several callbacks were employed for future visualization and to try and prevent overfitting. The ModelCheckpoint callback from the tf.keras.callbacks API was utilized to save the weights of the best performing model epoch throughout the training process and restore the weights to that model upon completion.
The EarlyStopping callback was used in an attempt to prevent overfitting and long model fit time, with a patience of 10 epochs on the validation loss monitoring metric. A custom python function was used to create TensorBoard callbacks allowing for
visualization of the model performances throughout the entire experimentation process. Finally, the ReduceLROnPlateau callback was used to, once again, try and combat overfitting by adjusting the Adam learning rate as the model began to slow down in imporvement efficiency.

## Results

The final CNN achieved an accuracy of 91.5% on the UrbanSound8K test set. Below are some key metrics:

- **Precision**: 0.92
- **Recall**: 0.92
- **F1-Score**: 0.92

### Classification Report
![Classification Report](classification_report.png)

### Loss Curves
![Loss Curves](model7_loss.png)

### Accuracy Curves
![Accuracy Curves](model7_accuracy.png)

## Usage

To use this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
3. Run the 'sound-model.ipynb' notebook in order to get full results

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please reach out to me:

- **Email**: clogan2004@gmail.com
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/cy-logan/)
- **GitHub**: [Your GitHub Profile](https://github.com/nogalcy)



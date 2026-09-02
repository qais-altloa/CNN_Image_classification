# CNN Image Classification with TensorFlow and Keras

A practical Convolutional Neural Network (CNN) project implementing image classification with **TensorFlow and Keras**. The project demonstrates both the **Sequential API** and the **Functional API** through two image-classification tasks: binary classification on the **Happy House** dataset and multiclass classification on the **SIGNS** dataset.

> **Program:** Deep Learning Specialization  
> **Course:** Course 4 — Convolutional Neural Networks  
> **Assignment:** Week 1  
> **Provider:** DeepLearning.AI  
> **Instructor:** Andrew Ng  
> **Framework:** TensorFlow / Keras

---

## 📌 Project Overview

This project demonstrates how Convolutional Neural Networks can be designed and trained for image classification.

The notebook contains two CNN applications:

### 🏠 Happy House
A **binary classification** task implemented using the **Keras Sequential API**. The model predicts whether an input image belongs to the target Happy House category.

### ✋ SIGNS
A **six-class classification** task implemented using the **Keras Functional API**. The model predicts which of six hand-sign classes is represented in an input image.

The project covers the main CNN workflow:

```text
Input Image
     ↓
Convolution
     ↓
Activation
     ↓
Pooling
     ↓
Feature Extraction
     ↓
Flattening
     ↓
Dense Layer
     ↓
Prediction
```

It also demonstrates convolution, padding, pooling, batch normalization, ReLU activation, sigmoid and softmax outputs, model compilation, training, evaluation, and learning-curve analysis.

---

## 🎯 Objectives

The main objectives of this project are to:

- Build image-classification models using TensorFlow/Keras.
- Understand the main components of a CNN.
- Implement a model using the Keras Sequential API.
- Implement a model using the Keras Functional API.
- Apply convolutional layers for spatial feature extraction.
- Use padding, strides, and pooling to control feature-map dimensions.
- Use ReLU activations to introduce non-linearity.
- Use batch normalization in a CNN architecture.
- Implement binary and multiclass classification.
- Choose appropriate output activations and loss functions.
- Train models using the Adam optimizer.
- Analyze training and held-out performance using accuracy and loss curves.
- Identify differences between training performance and generalization performance.

---

# 🧠 CNN Concepts Demonstrated

## Convolution

A convolutional layer applies learnable filters to an image to extract spatial features.

For an RGB input image:

```text
64 × 64 × 3
```

a convolutional layer can produce multiple feature maps:

```text
64 × 64 × number_of_filters
```

The three input channels represent:

```text
Red
Green
Blue
```

During training, the convolutional filters learn useful patterns such as edges, textures, and shapes.

---

## Padding

Padding adds pixels around the input before convolution.

For example, the Happy House model uses:

```python
ZeroPadding2D((3, 3))
```

This adds a border of three pixels around the height and width dimensions.

Padding can help preserve spatial information before applying convolution.

---

## Stride

Stride determines how far the convolution or pooling operation moves across the input.

For example:

```text
stride = 1
```

moves one pixel at a time, while larger strides reduce the spatial dimensions more aggressively.

---

## Pooling

Pooling reduces the spatial dimensions of feature maps.

For example, a `2 × 2` max-pooling operation with stride 2 changes approximately:

```text
64 × 64
```

to:

```text
32 × 32
```

Pooling reduces the computational cost and compresses spatial information while retaining strong local responses.

This project uses different pooling configurations in the two models.

---

## ReLU

The Rectified Linear Unit is used after convolutional layers:

```text
ReLU(x) = max(0, x)
```

ReLU introduces non-linearity, allowing the network to learn more complex relationships.

---

## Batch Normalization

The Happy House model uses:

```python
BatchNormalization(axis=3)
```

The images use the channels-last format:

```text
(batch, height, width, channels)
```

Therefore, `axis=3` corresponds to the channel dimension.

Batch normalization normalizes activations during training and can help stabilize the optimization process.

---

# 🏠 Model 1 — Happy House

## Task

The first model performs **binary image classification**.

The output is a single probability representing the model's prediction for the binary target class.

```text
Input Image
     ↓
CNN
     ↓
Sigmoid
     ↓
Binary Prediction
```

The final layer is:

```python
Dense(1, activation="sigmoid")
```

---

## Dataset

The Happy House dataset used in the notebook contains:

| Property | Value |
|---|---:|
| Training examples | 600 |
| Test examples | 150 |
| Image dimensions | 64 × 64 |
| Channels | 3 |
| Input shape | `(64, 64, 3)` |
| Number of classes | 2 |

Label shapes:

```text
Y_train → (600, 1)
Y_test  → (150, 1)
```

---

## Architecture

The Happy House model is implemented using the **Keras Sequential API**.

```text
Input
(64, 64, 3)
      │
      ▼
ZeroPadding2D
padding = (3, 3)
      │
      ▼
Conv2D
32 filters
7 × 7 kernel
stride = 1
      │
      ▼
Batch Normalization
      │
      ▼
ReLU
      │
      ▼
MaxPooling2D
2 × 2
stride = 2
      │
      ▼
Flatten
      │
      ▼
Dense
1 unit
sigmoid
      │
      ▼
Binary Prediction
```

The model is defined as:

```python
model = tf.keras.Sequential([
    ZeroPadding2D(padding=(3, 3), input_shape=(64, 64, 3)),
    Conv2D(32, (7, 7), strides=(1, 1)),
    BatchNormalization(axis=3),
    ReLU(),
    MaxPooling2D(),
    Flatten(),
    Dense(1, activation="sigmoid")
])
```

---

## Model Configuration

The model uses:

```text
Optimizer: Adam
Loss: Binary Cross-Entropy
Metric: Accuracy
```

The output and loss are matched to the binary classification task:

```text
Dense(1, sigmoid)
        ↓
Binary Cross-Entropy
```

---

## Training Configuration

The recorded notebook training configuration is:

```text
Epochs:      10
Batch size: 16
```

The model is trained using the 600 training images and evaluated on the 150-image test set.

---

## Parameter Count

The Happy House architecture contains:

```text
37,633 trainable parameters
```

---

## Results

The recorded evaluation result is:

| Metric | Result |
|---|---:|
| Test Loss | ~1.3184 |
| Test Accuracy | **66.0%** |

This result corresponds to the specific training run stored in the notebook.

---

# ✋ Model 2 — SIGNS

## Task

The second model performs **multiclass image classification**.

The model predicts one of six hand-sign classes.

```text
Input Image
     ↓
CNN
     ↓
Softmax
     ↓
6 Class Probabilities
     ↓
Predicted Class
```

The final layer is:

```python
Dense(6, activation="softmax")
```

---

## Dataset

The SIGNS dataset used in the notebook contains:

| Property | Value |
|---|---:|
| Training examples | 1080 |
| Test examples | 120 |
| Image dimensions | 64 × 64 |
| Channels | 3 |
| Input shape | `(64, 64, 3)` |
| Number of classes | 6 |

The labels are one-hot encoded:

```text
Y_train → (1080, 6)
Y_test  → (120, 6)
```

A one-hot encoded label can be represented as:

```text
[0, 0, 1, 0, 0, 0]
```

where the position containing `1` identifies the target class.

---

# 🔧 SIGNS Architecture

The SIGNS model is implemented using the **Keras Functional API**.

```text
Input
(64, 64, 3)
      │
      ▼
Conv2D
8 filters
4 × 4 kernel
stride = 1
padding = same
      │
      ▼
ReLU
      │
      ▼
MaxPooling2D
8 × 8
stride = 8
padding = same
      │
      ▼
Conv2D
16 filters
2 × 2 kernel
stride = 1
padding = same
      │
      ▼
ReLU
      │
      ▼
MaxPooling2D
4 × 4
stride = 4
padding = same
      │
      ▼
Flatten
      │
      ▼
Dense
6 units
softmax
      │
      ▼
6-Class Prediction
```

The model is constructed as:

```python
def convolutional_model(input_shape):
    input_img = Input(shape=input_shape)

    Z1 = Conv2D(
        8,
        (4, 4),
        strides=1,
        padding="same"
    )(input_img)

    A1 = ReLU()(Z1)

    P1 = MaxPooling2D(
        (8, 8),
        strides=8,
        padding="same"
    )(A1)

    Z2 = Conv2D(
        16,
        (2, 2),
        strides=1,
        padding="same"
    )(P1)

    A2 = ReLU()(Z2)

    P2 = MaxPooling2D(
        (4, 4),
        strides=4,
        padding="same"
    )(A2)

    F = Flatten()(P2)

    outputs = Dense(
        6,
        activation="softmax"
    )(F)

    model = Model(
        inputs=input_img,
        outputs=outputs
    )

    return model
```

---

# 📐 SIGNS Dimension Flow

The feature-map dimensions through the model are:

```text
Input
64 × 64 × 3

      ↓ Conv2D
      8 filters
      padding = same

64 × 64 × 8

      ↓ MaxPooling2D
      pool = 8 × 8
      stride = 8

8 × 8 × 8

      ↓ Conv2D
      16 filters
      padding = same

8 × 8 × 16

      ↓ MaxPooling2D
      pool = 4 × 4
      stride = 4

2 × 2 × 16

      ↓ Flatten

64

      ↓ Dense(6)

6
```

The flattening operation is:

```text
2 × 2 × 16 = 64
```

Therefore, the Dense layer receives 64 input features.

---

# 📊 SIGNS Model Parameters

The recorded model summary contains:

| Layer | Output Shape | Parameters |
|---|---|---:|
| InputLayer | `(None, 64, 64, 3)` | 0 |
| Conv2D | `(None, 64, 64, 8)` | 392 |
| ReLU | `(None, 64, 64, 8)` | 0 |
| MaxPooling2D | `(None, 8, 8, 8)` | 0 |
| Conv2D | `(None, 8, 8, 16)` | 528 |
| ReLU | `(None, 8, 8, 16)` | 0 |
| MaxPooling2D | `(None, 2, 2, 16)` | 0 |
| Flatten | `(None, 64)` | 0 |
| Dense | `(None, 6)` | 390 |

### Total parameters

```text
1,310 trainable parameters
```

---

# ⚙️ SIGNS Model Configuration

The model is compiled with:

```python
optimizer = Adam()
loss = "categorical_crossentropy"
metrics = ["accuracy"]
```

The classification setup is:

```text
Dense(6, softmax)
        ↓
Categorical Cross-Entropy
```

Softmax converts the six outputs into a probability distribution across the six classes.

---

# 🏋️ SIGNS Training

The recorded training configuration is:

```text
Epochs:      100
Batch size: 64
Optimizer:   Adam
Loss:        Categorical Cross-Entropy
Metric:      Accuracy
```

---

# 📈 SIGNS Training Results

The notebook contains a stored 100-epoch training history.

### Training Accuracy

```text
Initial accuracy ≈ 15.6%
Final accuracy  ≈ 84.0%
Best accuracy   ≈ 84.17%
```

### Held-Out/Test-Split Accuracy

```text
Final accuracy = 75.0%
Best accuracy  = 75.0%
```

### Final Loss

```text
Training loss            ≈ 0.511
Held-out/test-split loss ≈ 0.692
```

The stored training curves are included in the repository:

![SIGNS Accuracy](images/signs_accuracy.png)

![SIGNS Loss](images/signs_loss.png)

---

# 📌 Results Summary

| Model | Task | Training Setup | Reported Result |
|---|---|---|---:|
| Happy House CNN | Binary classification | 10 epochs, batch size 16 | **66.0% test accuracy** |
| SIGNS CNN | Multiclass classification | 100 epochs, batch size 64 | **75.0% held-out/test-split accuracy** |

### Important evaluation note

For the SIGNS experiment, the notebook passes the provided test split to Keras as `validation_data` during training.

Therefore, the reported **75.0%** should be described as **held-out/test-split accuracy**, not as a completely untouched final test-set evaluation.

In a production machine-learning workflow, a separate validation set would normally be used for model selection, while the test set would remain untouched until final evaluation.

```text
Training Set
     ↓
Model Training

Validation Set
     ↓
Model Selection / Tuning

Test Set
     ↓
Final Evaluation
```

---

# 🔎 Interpretation of the Results

The SIGNS model reaches approximately:

```text
Training accuracy ≈ 84%
Held-out accuracy ≈ 75%
```

The difference between training and held-out performance indicates that the model fits the training data better than the held-out data.

This suggests some degree of **overfitting** and indicates that generalization could be improved.

The loss and accuracy curves provide a visual way to inspect this behavior throughout training.

---

# 🏗️ Sequential API vs Functional API

## Sequential API

The Happy House model uses:

```python
tf.keras.Sequential(...)
```

The Sequential API represents a straightforward stack of layers:

```text
Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
Layer 4
```

It is simple and well suited to architectures where each layer has a single sequential connection.

---

## Functional API

The SIGNS model uses:

```python
Input(...)
Model(inputs=..., outputs=...)
```

The Functional API provides explicit control over how tensors flow through the network.

It is particularly useful for more complex architectures involving:

- multiple inputs
- multiple outputs
- branches
- skip connections
- shared layers
- non-linear computation graphs

Using both APIs demonstrates practical experience with two major Keras model-building approaches.

---

# 🧪 Data Representation

The CNNs receive RGB images in channels-last format:

```text
(batch, height, width, channels)
```

For these datasets:

```text
(height, width, channels)
=
(64, 64, 3)
```

The target representation differs by task:

### Happy House

```text
Y_train → (600, 1)
Y_test  → (150, 1)
```

### SIGNS

```text
Y_train → (1080, 6)
Y_test  → (120, 6)
```

The SIGNS labels use one-hot encoding because the model performs six-class classification with categorical cross-entropy.

---

# 🧮 Binary vs Multiclass Classification

The two models demonstrate how the output layer and loss function change depending on the task.

## Binary Classification

```text
Dense(1, sigmoid)
        ↓
Binary Cross-Entropy
```

Used for:

**Happy House**

---

## Multiclass Classification

```text
Dense(6, softmax)
        ↓
Categorical Cross-Entropy
```

Used for:

**SIGNS**

This demonstrates an important principle in supervised learning:

```text
Problem type
     ↓
Output representation
     ↓
Appropriate loss function
```

---

# 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- HDF5
- Jupyter Notebook
- Convolutional Neural Networks
- Deep Learning

---

# 📁 Project Structure

The intended portfolio repository structure is:

```text
CNN_Image_Classification/
│
├── CNN_Image_Classification.ipynb
├── README.md
├── cnn_utils.py
├── .gitignore
│
└── images/
    ├── signs_accuracy.png
    └── signs_loss.png
```

### File descriptions

| File / Folder | Purpose |
|---|---|
| `CNN_Image_Classification.ipynb` | Complete CNN implementation, training, evaluation, and analysis |
| `cnn_utils.py` | Supporting utility functions used by the notebook |
| `README.md` | Project documentation |
| `.gitignore` | Prevents temporary and unnecessary files from being committed |
| `images/` | Stores training-result visualizations |

---

# 📦 Dataset

The course datasets are **not included in this public repository**.

To run the notebook locally, place the required HDF5 files in:

```text
datasets/
├── train_signs.h5
├── test_signs.h5
├── train_happy.h5
└── test_happy.h5
```

These files are required by the notebook for training and evaluation.

The datasets and course-specific files are intentionally kept outside the public portfolio repository.

---

# ▶️ How to Run the Project

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CNN_Image_Classification
```

## 2. Install the required packages

```bash
pip install tensorflow numpy pandas matplotlib h5py jupyter
```

## 3. Add the datasets

Place the required HDF5 files inside:

```text
datasets/
```

as described in the dataset section above.

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

## 5. Open the notebook

```text
CNN_Image_Classification.ipynb
```

## 6. Run the notebook

The notebook performs the following workflow:

```text
Load Data
    ↓
Prepare Labels / Inputs
    ↓
Build CNN
    ↓
Compile Model
    ↓
Train
    ↓
Evaluate
    ↓
Analyze Accuracy and Loss
```

---

# ⚠️ Reproducibility Note

The notebook is based on an educational course assignment and contains some saved execution-state inconsistencies.

One saved SIGNS execution contains the following shape-mismatch error:

```text
ValueError:
Shapes (None, 1) and (None, 6) are incompatible
```

However, the notebook also contains the correct one-hot encoded SIGNS label shapes:

```text
Y_train → (1080, 6)
Y_test  → (120, 6)
```

and contains a stored 100-epoch training history from a successful model run.

Therefore, the SIGNS results and learning curves reported in this README correspond to the **stored successful training history** in the notebook.

For maximum reproducibility, the notebook should ideally be restarted from a clean kernel and executed from beginning to end before using it as a final experimental benchmark.

---

# 🚀 Potential Improvements

The current CNN architectures are compact educational models. Several improvements could be explored in future experiments:

- Data augmentation
- Additional convolutional layers
- Different filter sizes and numbers of filters
- Dropout regularization
- Improved pooling strategies
- Learning-rate tuning
- Early stopping
- A separate validation set
- Confusion matrix analysis
- Precision, recall, and F1-score
- Example prediction visualizations
- Transfer learning using architectures such as MobileNet or ResNet
- Reproducible random seeds
- Model checkpointing
- Saved model weights
- A dedicated `requirements.txt`

---

# 📚 Key Learning Outcomes

This project provided practical experience with:

### CNN Fundamentals

- Convolutional filters
- Feature maps
- Kernel size
- Strides
- Padding
- Pooling
- Flattening
- Fully connected layers

### TensorFlow / Keras

- `Conv2D`
- `ZeroPadding2D`
- `BatchNormalization`
- `ReLU`
- `MaxPooling2D`
- `Flatten`
- `Dense`
- `Input`
- `Model`
- `Sequential`

### Model Training

- Adam optimization
- Binary cross-entropy
- Categorical cross-entropy
- Sigmoid activation
- Softmax activation
- Batch training
- Epoch-based training

### Model Evaluation

- Accuracy
- Loss
- Training curves
- Held-out performance
- Generalization
- Overfitting analysis

---

# 📌 Project Takeaway

This project demonstrates the end-to-end process of constructing CNN image classifiers with TensorFlow and Keras.

The two models illustrate both simple and flexible approaches to building neural networks:

```text
Happy House
     ↓
Sequential API
     ↓
Binary Classification
     ↓
66.0% recorded test accuracy
```

```text
SIGNS
     ↓
Functional API
     ↓
Six-Class Classification
     ↓
75.0% recorded held-out/test-split accuracy
```

The SIGNS model reached approximately **84% training accuracy** compared with **75% held-out/test-split accuracy**, highlighting the difference between fitting the training data and generalizing to unseen examples.

Overall, the project demonstrates practical understanding of CNN architecture design, TensorFlow/Keras implementation, binary and multiclass classification, model training, and performance analysis.

---

---

# 🎓 Course Attribution

This project was developed as part of the **Deep Learning Specialization**, specifically:

- **Course 4 — Convolutional Neural Networks**
- **Week 1 Assignment**
- **Provider:** DeepLearning.AI
- **Instructor:** Andrew Ng

This repository presents my implementation and analysis of the CNN concepts covered in the course, using TensorFlow/Keras.

The project is maintained as an educational portfolio project to demonstrate practical understanding of:

- Convolutional Neural Networks
- TensorFlow/Keras
- Sequential and Functional APIs
- Binary and multiclass image classification
- Model training and evaluation
- Learning-curve analysis

Course-provided datasets and course-specific testing/grading materials are not included in the public repository.

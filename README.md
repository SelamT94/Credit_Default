# 💳 Credit Default Prediction System

> **Deep Learning & End-to-End Implementation Project**  
> A comprehensive machine learning pipeline for predicting credit default risk using traditional ML, deep learning, and ensemble methods.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for credit default prediction. It combines traditional machine learning algorithms with deep learning techniques and automated ML solutions to create a robust prediction system. The project includes data preprocessing, model training, evaluation, and an interactive web dashboard for predictions.

**Key Objectives:**
- Predict credit default risk using multiple ML approaches
- Compare performance of traditional ML vs. Deep Learning models
- Implement ensemble methods for improved predictions
- Deploy an interactive Streamlit dashboard for real-time predictions
- Demonstrate end-to-end deep learning implementation best practices

---

## ✨ Features

- 🔍 **Exploratory Data Analysis**: Comprehensive data exploration notebooks
- 🤖 **Multiple ML Models**: Random Forest, Gradient Boosting, SVM, Logistic Regression
- 🧠 **Deep Learning**: Multi-Layer Perceptron (MLP) with TensorFlow/Keras
- 🚀 **AutoML Integration**: AutoGluon for automated model selection
- 🎭 **Ensemble Methods**: Voting classifiers for model combination
- ⚖️ **Class Imbalance Handling**: SMOTE for balanced training data
- 📊 **Interactive Dashboard**: Streamlit web application for predictions
- 📈 **Comprehensive Evaluation**: Detailed metrics and visualizations
- 💾 **Model Persistence**: Save and load trained models

---

## 📁 Project Structure

```
Credit_Default_Project/
├── data/
├── notebooks/
│   ├── 1_data_exploration.ipynb    # EDA and data analysis
│   ├── 2_model_training.ipynb      # Model training pipeline
│   └── 3_evaluation.ipynb          # Model evaluation and comparison
├── src/
│   ├── preprocessing.py             # Data preprocessing utilities
│   ├── ml_models.py                 # Traditional ML models
│   ├── dl_models.py                 # Deep learning models
│   ├── pretrained_models.py         # AutoGluon integration
│   └── ensemble.py                  # Ensemble methods
├── models/
│   └── saved_models/                # Trained model files
├── app/
│   └── streamlit_app.py             # Streamlit dashboard
├── reports/
│   └── visualizations/              # Generated plots and figures
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🛠️ Technologies Used

### Core Libraries
- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Traditional machine learning algorithms
- **TensorFlow/Keras** - Deep learning framework
- **AutoGluon** - Automated machine learning

### Data Processing
- **imbalanced-learn** - SMOTE for handling class imbalance
- **StandardScaler** - Feature scaling
- **OneHotEncoder** - Categorical encoding

### Visualization & Deployment
- **Matplotlib** - Plotting
- **Seaborn** - Statistical visualization
- **Plotly** - Interactive visualizations
- **Streamlit** - Web application framework

### Model Persistence
- **Joblib** - Model serialization

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

### Setup Instructions

1. **Clone the repository** (or download the project):
   ```bash
   git clone <repository-url>
   cd Credit_Default_Project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your dataset**:
   - Place your credit dataset CSV file in `data/credit_dataset.csv`
   - Ensure the dataset has a target column named `default` (or update the code accordingly)

---

## 🚀 Usage

### 1. Data Exploration

Start by exploring your dataset using the provided Jupyter notebook:

```bash
jupyter notebook notebooks/1_data_exploration.ipynb
```

This notebook will help you:
- Understand the dataset structure
- Identify missing values
- Explore feature distributions
- Analyze correlations

### 2. Model Training

Train models using the training notebook:

```bash
jupyter notebook notebooks/2_model_training.ipynb
```

Example Python code:

```python
from src.preprocessing import load_data, preprocess
from src.ml_models import train_rf, evaluate_model, save_model
from src.dl_models import build_mlp, train_model, save_dl_model

# Load and preprocess data
df = load_data('data/credit_dataset.csv')
X_train, X_test, y_train, y_test = preprocess(df, target='default')

# Train Random Forest
rf_model = train_rf(X_train, y_train)
evaluate_model(rf_model, X_test, y_test)
save_model(rf_model, 'models/saved_models/rf_model.pkl')

# Train Deep Learning Model
input_dim = X_train.shape[1]
mlp_model = build_mlp(input_dim)
history = train_model(mlp_model, X_train, y_train, X_test, y_test, 
                      epochs=100, batch_size=32)
save_dl_model(mlp_model, 'models/saved_models/dl_model.h5')
```

### 3. Model Evaluation

Evaluate and compare models:

```bash
jupyter notebook notebooks/3_evaluation.ipynb
```

This notebook provides:
- Detailed performance metrics
- Confusion matrices
- ROC curves
- Feature importance analysis
- Model comparison tables

### 4. Streamlit Dashboard

Launch the interactive web dashboard:

```bash
streamlit run app/streamlit_app.py
```

The dashboard allows you to:
- Upload CSV files for batch predictions
- View prediction results
- See default probabilities
- Explore the dataset interactively

---

## 🧠 Model Details

### Machine Learning Models

#### Random Forest
- **Algorithm**: Random Forest Classifier
- **Parameters**: 
  - `n_estimators=200`
  - `max_depth=20`
  - `random_state=42`

### Deep Learning Models

#### Multi-Layer Perceptron (MLP)
- **Architecture**:
  - Input Layer: Features from dataset
  - Hidden Layer 1: 64 neurons (ReLU activation)
  - Dropout: 0.3
  - Hidden Layer 2: 32 neurons (ReLU activation)
  - Output Layer: 1 neuron (Sigmoid activation)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

### Pretrained Models

#### AutoGluon TabularPredictor
- Automated feature engineering
- Model selection and hyperparameter tuning
- Ensemble creation
- Evaluation metric: ROC-AUC

### Ensemble Methods

#### Voting Classifier
- Combines Random Forest, MLP, and Tabular models
- Soft voting for probability-based predictions
- Weighted averaging of predictions

---

## 📊 Results

### Evaluation Metrics

Models are evaluated using the following metrics:

- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

### Model Comparison

*(Add your results here after training and evaluation)*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | - | - | - | - | - |
| MLP | - | - | - | - | - |
| AutoGluon | - | - | - | - | - |
| Ensemble | - | - | - | - | - |

---

## 🔧 Configuration

### Preprocessing Settings

- **Test Size**: 20% of the data
- **Random State**: 42 (for reproducibility)
- **SMOTE**: Applied for class imbalance handling
- **Scaling**: StandardScaler for numerical features
- **Encoding**: One-hot encoding for categorical features

### Model Training Settings

- **Validation Split**: Used for deep learning models
- **Early Stopping**: Patience of 10 epochs
- **Batch Size**: 32 (for deep learning)
- **Epochs**: 100 (default, with early stopping)

---

## 📝 Notes

- Ensure your dataset has a target column named `default` (binary: 0/1)
- Update the target column name in `preprocess()` if your dataset uses a different name
- Adjust model hyperparameters in the respective model files as needed
- For AutoGluon, ensure sufficient computational resources (memory and time)

---

## 🤝 Contributing

This is a course project. For improvements or suggestions:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Your Name**  
Master's in [Your Program]  
[Your University]

---

## 🙏 Acknowledgments

- Course instructors and teaching assistants
- Scikit-learn, TensorFlow, and AutoGluon communities
- Streamlit team for the amazing framework
- Open-source contributors and the ML community

---

## 📚 References

- Scikit-learn Documentation: https://scikit-learn.org/
- TensorFlow Documentation: https://www.tensorflow.org/
- AutoGluon Documentation: https://auto.gluon.ai/
- Streamlit Documentation: https://docs.streamlit.io/
- SMOTE: Chawla et al. (2002) - "SMOTE: Synthetic Minority Over-sampling Technique"

---

<div align="center">
  
**Made with ❤️ for Deep Learning & End-to-End ML Implementation**

⭐ Star this repo if you find it helpful!

</div>


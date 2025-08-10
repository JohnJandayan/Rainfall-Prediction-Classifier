# Rainfall Prediction Classifier

A machine learning project for predicting rainfall based on historical weather data from Australia. This project was developed as part of the **Machine Learning with Python** course from IBM on Coursera, demonstrating comprehensive data science and machine learning skills through feature engineering, model selection, and performance evaluation.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/seaborn-%23ffffff.svg?style=for-the-badge&logo=seaborn&logoColor=%2318188c)
![Coursera](https://img.shields.io/badge/Coursera-%230056D2.svg?style=for-the-badge&logo=Coursera&logoColor=white)

## 🌟 Project Overview

Welcome to the Rainfall Prediction Classifier project! This comprehensive machine learning project simulates a real-world scenario where you work as a data scientist at **WeatherTech Inc.**, responsible for building a predictive model that determines whether it will rain tomorrow based on historical weather data from Australia.

The project demonstrates proficiency in:
- **Data Exploration & Preprocessing**: Feature engineering, data cleaning, and handling missing values
- **Model Development**: Building robust classifier pipelines with multiple algorithms
- **Model Optimization**: Hyperparameter tuning and cross-validation techniques
- **Performance Evaluation**: Comprehensive model assessment using various metrics and visualizations

## 🎯 Business Problem

As a data scientist at WeatherTech Inc., the goal is to develop an accurate rainfall prediction model that can:
- Help agricultural planning and irrigation management
- Assist in outdoor event planning and logistics
- Support weather-dependent business decision making
- Provide reliable short-term weather forecasting capabilities

## 📊 Dataset Description

The dataset originates from the **Australian Government's Bureau of Meteorology** and contains weather observations from 2008 to 2017. The data includes various meteorological features such as:

| Feature | Description | Unit |
|---------|-------------|------|
| **MinTemp** | Minimum temperature | Celsius |
| **MaxTemp** | Maximum temperature | Celsius |
| **Rainfall** | Amount of rainfall | Millimeters |
| **Evaporation** | Amount of evaporation | Millimeters |
| **Sunshine** | Amount of bright sunshine | Hours |
| **WindGustDir** | Direction of strongest wind gust | Compass Points |
| **WindGustSpeed** | Speed of strongest wind gust | Kilometers/Hour |
| **Humidity9am/3pm** | Humidity levels | Percent |
| **Pressure9am/3pm** | Atmospheric pressure | Hectopascal |
| **Cloud9am/3pm** | Sky cloud coverage | Eights |
| **Temp9am/3pm** | Temperature readings | Celsius |
| **WindDir9am/3pm** | Wind direction | Compass Points |
| **WindSpeed9am/3pm** | Wind speed | Kilometers/Hour |
| **RainToday** | Rainfall occurrence today | Yes/No |
| **RainTomorrow** | **Target Variable** - Rainfall tomorrow | Yes/No |

## 🔬 Methodology

### 1. Data Exploration & Preparation
- **Data Quality Assessment**: Analyze missing values and data distribution
- **Feature Engineering**: Create seasonal features from date information
- **Location Filtering**: Focus on Melbourne area (Melbourne, Melbourne Airport, Watsonia)
- **Data Leakage Prevention**: Careful consideration of temporal relationships

### 2. Data Preprocessing
- **Missing Value Handling**: Strategic removal of incomplete records
- **Feature Selection**: Automatic detection of numerical and categorical variables
- **Data Transformation**: Standardization for numerical features, encoding for categorical
- **Target Stratification**: Ensure balanced representation in train/test splits

### 3. Model Development
- **Pipeline Construction**: Automated preprocessing and model fitting
- **Algorithm Selection**: Implementation of multiple classification algorithms
- **Cross-Validation**: Robust model validation using stratified k-fold
- **Hyperparameter Optimization**: Grid search for optimal model parameters

### 4. Model Evaluation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Confusion Matrix Analysis**: Detailed error analysis
- **Visualization**: Performance comparison charts and model interpretability

## 🛠️ Technology Stack

- **Programming Language**: Python 3.x
- **Development Environment**: Jupyter Notebook
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Data Visualization**: Matplotlib, Seaborn
- **Model Algorithms**: Random Forest, Logistic Regression, and others

## 📈 Key Features

- **Comprehensive EDA**: Thorough exploratory data analysis with statistical insights
- **Robust Preprocessing**: Automated feature detection and transformation pipelines
- **Multiple Algorithms**: Comparison of various machine learning classifiers
- **Model Optimization**: Systematic hyperparameter tuning with cross-validation
- **Performance Analysis**: Detailed evaluation with multiple metrics and visualizations
- **Class Imbalance Handling**: Strategic approaches to address dataset imbalance (76.3% no rain vs 23.7% rain)

## 🎓 Learning Outcomes

This project demonstrates mastery of:
- **Data Science Workflow**: End-to-end project development from data to deployment
- **Feature Engineering**: Creative feature creation and selection techniques
- **Machine Learning Pipeline**: Automated preprocessing and model building
- **Model Evaluation**: Comprehensive performance assessment methodologies
- **Business Application**: Practical problem-solving in meteorological prediction

## 📝 Project Structure

```
Rainfall-Prediction-Classifier/
│
├── FinalProject_AUSWeather.ipynb    # Main Jupyter notebook containing the complete ML pipeline
└── README.md                        # Comprehensive project documentation
```

## 🚀 Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/JohnJandayan/Rainfall-Prediction-Classifier.git
   cd Rainfall-Prediction-Classifier
   ```

2. **Install Dependencies**
   ```bash
   pip install pandas numpy matplotlib scikit-learn seaborn jupyter
   ```

3. **Run the Notebook**
   ```bash
   jupyter notebook FinalProject_AUSWeather.ipynb
   ```

## 📊 Results & Insights

The project achieves significant insights into rainfall prediction:
- **Baseline Accuracy**: 76.3% (always predicting "no rain")
- **Model Performance**: Substantial improvement over baseline through machine learning
- **Feature Importance**: Identification of key meteorological predictors
- **Seasonal Patterns**: Discovery of weather seasonality impacts on prediction accuracy

## 🎯 Future Enhancements

- **Feature Engineering**: Additional temporal and spatial features
- **Advanced Algorithms**: Deep learning and ensemble methods
- **Real-time Prediction**: API development for live weather prediction
- **Geographic Expansion**: Multi-location and national-scale modeling
- **Mobile Application**: User-friendly interface for daily predictions

## 📚 Course Context

This project is part of the **Machine Learning with Python** course offered by **IBM** on **Coursera**. It represents the culmination of learning in:
- Supervised learning algorithms
- Model selection and evaluation
- Feature engineering techniques
- Data preprocessing methodologies
- Performance optimization strategies

## 🎓 Attribution

This project is based on coursework from IBM's Machine Learning with Python course on Coursera. The original course materials and framework are provided by IBM/Skills Network. The implementation and solutions are my own work completed as part of the course requirements.

**Built with 🌧️ for Weather Prediction by John Jandayan**

*Predicting Tomorrow's Weather, Today*

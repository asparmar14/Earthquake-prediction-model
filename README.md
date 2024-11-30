# Earthquake Magnitude Classification and Visualization

This project analyzes earthquake data in India, classifies earthquakes based on their magnitudes, and visualizes the results through various charts, maps, and models. The analysis uses machine learning models such as Random Forest, Decision Trees, Logistic Regression, and Support Vector Machines (SVM) to predict earthquake magnitudes based on geographical and seismic data.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation](#model-evaluation)
- [Visualization](#visualization)
- [Map Visualization](#map-visualization)
- [Dependencies](#dependencies)

## Project Overview

This project utilizes earthquake data from India to:

- Classify earthquake magnitudes into different categories.
- Build machine learning models to predict earthquake magnitudes based on geographical features.
- Visualize earthquake data through histograms, boxplots, confusion matrices, and maps.
- Compare the performance of multiple models and choose the best one based on accuracy and other metrics.

## Data

The dataset used for this project contains information on earthquakes in India, including:

- `latitude`: Latitude of the earthquake's epicenter
- `longitude`: Longitude of the earthquake's epicenter
- `depth`: Depth of the earthquake in kilometers
- `mag`: Magnitude of the earthquake
- `place`: Location where the earthquake occurred

The dataset is stored in a CSV file, `eq_India.csv`.

## Feature Engineering

In this project, new features were created to enhance model performance:

- **Magnitude Categories**: Earthquakes were classified into categories based on their magnitude (e.g., 'Mega', 'Great', 'Major').
- **Location Extraction**: The location and country of the earthquake were extracted from the 'place' column.
- **Geographical Features**: Additional features were added based on the distance from the equator and prime meridian.

## Machine Learning Models

The project uses the following machine learning models to predict earthquake magnitude categories:

- **Random Forest Classifier**
- **Logistic Regression**
- **Decision Tree Classifier**
- **Support Vector Machine (SVM)**

Hyperparameter tuning was performed using GridSearchCV to find the optimal parameters for each model.

## Model Evaluation

The performance of the models was evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score

Additionally, a confusion matrix was used to visualize the classification performance.

## Visualization

The project includes various visualizations to explore and interpret the earthquake data:

- **Histogram**: Distribution of earthquake magnitudes.
- **Boxplot**: Distribution of earthquake magnitudes with outliers.
- **Feature Importance**: Visualization of feature importance from the Random Forest model.
- **Model Comparison**: Bar chart comparing the accuracy and other metrics of different models.

## Map Visualization

The project uses **Cartopy** and **Folium** to visualize earthquake data on maps. Earthquakes are color-coded based on their magnitude categories, and interactive maps are generated showing the locations of the earthquakes.

### Cartopy Map

A static map is generated with Cartopy to show the distribution of earthquakes in India.

### Folium Map

An interactive map is created using Folium where each earthquake is represented as a circle marker, color-coded based on its magnitude category. The map also includes a custom legend.

## Dependencies

This project requires the following Python libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- cartopy
- folium

# COS40007 Artificial Intelligence for Engineering - Portfolio Assessments

## Overview
This repository contains the portfolio assessments for COS40007 Artificial Intelligence for Engineering. Each portfolio covers different machine learning and deep learning tasks, progressing from fundamental data analysis to deep learning model development using various techniques. 

## Portfolio Assessments

### Portfolio Assessment 1: "Hello Machine Learning for Engineering"
**Due Date:** Week 3 (11/08/2024)

**Objective:**
- Demonstrate familiarity with Python programming, data exploration, feature engineering, and decision tree models.

**Tasks:**
1. Define class labels for the dataset.
2. Normalize numerical features and categorize integer-based features.
3. Perform feature engineering and create new features.
4. Develop decision tree models with different feature sets and compare results.

**Submission:**
- PDF report including dataset details, EDA summary, model comparisons, and source code shared via a link.

---

### Portfolio Assessment 2: "Systematic Approach to Develop ML Model"
**Due Date:** Week 4 (18/08/2024)

**Objective:**
- Develop a classification model for activity recognition using acceleration data from body-worn sensors.

**Tasks:**
1. Extract relevant sensor data and create composite features.
2. Compute statistical features (mean, standard deviation, peaks, etc.).
3. Train SVM classifiers with different configurations.
4. Compare various ML models including SGD, RandomForest, and MLP.
5. Select the best model and justify the choice.

**Submission:**
- PDF report including data processing steps, model training results, and code link.

---

### Portfolio Assessment 3: "Develop AI Model by Your Own Decision"
**Due Date:** Week 5 (25/08/2024)

**Objective:**
- Independently explore and analyze a dataset to develop an optimized ML model for classification.

**Tasks:**
1. Preprocess the dataset by balancing classes, removing redundant columns, and feature selection.
2. Train multiple ML models and evaluate them using classification reports and confusion matrices.
3. Convert ML to AI by applying the trained model to real-time unseen data.
4. Generate decision rules from the trained model.

**Submission:**
- PDF report with dataset exploration, model evaluation, decision rules, and source code link.

---

### Portfolio Assessment 4: "Deep Learning Using TensorFlow and Keras"
**Due Date:** Week 6 (06/09/2024)

**Objective:**
- Implement deep learning models (CNN and ResNet50) for image classification tasks.

**Tasks:**
1. Train a CNN model on corrosion dataset and evaluate performance.
2. Train a ResNet50 model on the same dataset and compare results.
3. Develop Mask RCNN for detecting logs in images and count detected objects.
4. Extend log labeling by adding a new class for broken logs.

**Submission:**
- Folder containing model source code, labeled dataset, test outcome images, and PDF report with evaluation results.

---

### Portfolio Assessment 5: "Deep Learning Using YOLO v5"
**Due Date:** Week 7 (20/09/2024)

**Objective:**
- Train a YOLO v5 model for graffiti detection using deep learning.

**Tasks:**
1. Convert dataset annotations to YOLO format.
2. Train a YOLO model iteratively until performance meets the desired threshold.
3. Evaluate model performance using IoU scores and generate CSV results.
4. Apply the trained model to detect graffiti in real-time video data.

**Submission:**
- Folder containing trained YOLO models, test outcome images, CSV results, and a PDF report with model performance analysis.

---

### COS40007 Design Project
**Due Date:** Week 12 (01/11/2024)

**Objective:**
- Develop an AI-based optimization and anomaly detection model for Vegemite production using machine learning techniques.

**Tasks:**
1. Process and clean the dataset collected from machine sensor readings.
2. Train and compare multiple ML models including Decision Tree, Random Forest, SVC, SGD, and MLP.
3. Evaluate models using classification accuracy, confusion matrices, and performance reports.
4. Implement a user interface to allow easy interaction with the model.
5. Document challenges, solutions, and lessons learned from the project.

**Submission:**
- Project report detailing data processing, model evaluation, and UI implementation.
- Source code for models and UI shared via GitHub.

---

### COS40007 Project Brief
**Due Date:** Week 12 (01/11/2024)

**Objective:**
- Define the scope and requirements for the AI-based Vegemite production optimization project.

**Tasks:**
1. Outline project objectives, background, and expected outcomes.
2. Detail the dataset sources, processing methods, and feature engineering techniques.
3. Specify must-have functionalities including setpoint recommendations and anomaly detection.
4. Identify optional features such as graphical UI integration and real-time data adaptation.

**Submission:**
- Project brief document covering objectives, requirements, and expected outcomes.

---

## Repository Structure
```
/Portfolio-Assessments
│── Portfolio-week2.pdf  # Portfolio Assessment 1
│── Portfolio-week3.pdf  # Portfolio Assessment 2
│── Portfolio-week4.pdf  # Portfolio Assessment 3
│── Portfolio-week5.pdf  # Portfolio Assessment 4
│── Portfolio-week6.pdf  # Portfolio Assessment 5
│── COS40007_Design_Project.pdf  # Design Project Report
│── Project Brief.pdf  # Project Brief Document
│── code/               # Source code for all assessments
│── datasets/           # Datasets used for training
│── models/             # Trained models and results
│── README.md           # This file
```

## Requirements
- Python 3.x
- Jupyter Notebook
- TensorFlow & Keras
- PyTorch
- OpenCV
- Scikit-learn
- Matplotlib, Pandas, NumPy

## How to Use
1. Clone the repository:
   ```sh
   git clone https://github.com/aalexandros47/Portfolio_Assessments.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Portfolio_Assessments
   ```
3. Open Jupyter Notebook and run the scripts inside the `code/` directory.
4. Modify dataset paths if necessary and execute scripts to train and evaluate models.

## Author
**Arnob Ghosh**

For any queries, feel free to reach out via GitHub issues.



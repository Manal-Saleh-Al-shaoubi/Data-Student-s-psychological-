ModelBuildingProject

Introduction

analysis and modeling for a dataset related to student mental health.

Objective

The main objective of this project is to analyze student mental health data and build predictive models to understand the factors influencing mental health and potentially predict it.

About the dataset

A set of organized information regarding students' mental health, which facilitates analysis and modeling to understand various impacts on their psychological lives.

The database refers to a CSV file containing data on students' mental health. This file includes records that represent important information about their psychological condition.

Steps

1. Library Imports:
   - The code imports necessary libraries such as numpy, pandas, matplotlib, seaborn, and various modules from scikit-learn.

2. Data Loading:
   - Reads a CSV file containing student mental health data.

3. Initial Data Exploration:
   - Displays the first few rows of the data
   - Shows general information about the dataset
   - Checks for missing values
   - Presents descriptive statistics of the data

4. Data Preprocessing:
   - Fills missing values in the 'Age' column with the median value
   - Removes duplicate rows

5. Data Visualization:
   - Plots the distribution of ages
   - Plots the count of students for each age
   - Plots the count of students by timestamp
   - Plots the age distribution based on time

6. Preparing Data for Modeling:
   - Converts categorical values to numbers using LabelEncoder
   - Splits the data into independent variables (X) and dependent variable (y)
   - Divides the data into training and testing sets

7. Calculates and displays the correlation matrix

8. Training and Evaluating Different Models:
   a. Logistic Regression:
      - Trains the model
      - Calculates accuracy
      - Displays the confusion matrix
      - Saves the model

   b. K-Nearest Neighbors (KNN):
      - Trains the model
      - Calculates accuracy
      - Finds the best K value

   c. Support Vector Machine (SVM):
      - Trains the model
      - Calculates accuracy

   d. Decision Tree:
      - Trains the model
      - Calculates accuracy
      - Finds the optimal number of leaves

   e. Random Forest:
      - Trains the model
      - Calculates accuracy

9. Model Comparison:
   - Creates a DataFrame containing the accuracy of each model
   - Displays the results sorted by accuracy

This code performs a comprehensive analysis of student mental health data, starting with data exploration and cleaning, moving through data visualization, and ending with training and evaluating several machine learning models to predict

Conclusion

This project is an simple application of what I have learned about ML fundamentals. For each step within the project implementation, I support the codes with comments and description.

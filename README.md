# Time Series Forecasting with LSTM (Long Short-Term Memory)

## Overview

This project focuses on forecasting time series data using **Long Short-Term Memory (LSTM)** networks, a type of **Recurrent Neural Network (RNN)**. LSTM models are effective for sequential data, making them ideal for predicting future values in time series data, such as stock prices, sales data, or temperature readings.

The notebook demonstrates how to apply LSTM networks to time series forecasting, from data preprocessing to model evaluation. By using historical data, we predict future values and measure the model's accuracy.

## Key Concepts

- **Time Series Data**: A time series is a sequence of data points indexed in time order. In this project, the goal is to predict future data points based on the historical values.
- **LSTM (Long Short-Term Memory)**: LSTMs are a type of RNN designed to learn long-term dependencies. They are particularly useful for time series forecasting because they can remember past data over long periods.
- **Supervised Learning**: The task is treated as a supervised learning problem where past data (features) is used to predict future values (target).
- **Data Preprocessing**: Proper preparation of the dataset is crucial for the performance of the LSTM model, including data normalization, splitting the data into training and test sets, and reshaping the data to be fed into the model.

## Libraries and Tools Used

- **Python**: Core language for implementation.
- **Jupyter Notebook**: Interactive environment for code execution and documentation.
- **Libraries**:
  - `pandas`: Data manipulation and preprocessing.
  - `numpy`: Numerical operations.
  - `matplotlib`/`seaborn`: Data visualization.
  - `tensorflow`/`keras`: For building the LSTM model.
  - `scikit-learn`: For splitting data and evaluating the model.

## Project Structure

- **Notebook**: `Time Series With LSTM.ipynb`
  - Contains the step-by-step process, from loading data to model evaluation.

## Code Walkthrough

### 1. **Data Loading and Preprocessing**

The first step is loading the time series data. We typically use historical data points to predict future values. The following operations are performed:



- **Date Parsing**: The date is parsed and set as the index of the dataset to ensure that the time series is indexed by time.
  
- **Data Normalization**: Time series data often has different ranges, so it's important to scale the data. LSTM models typically perform better when input data is scaled.



- **Scaling**: The `MinMaxScaler` is used to scale the data between 0 and 1 to make the LSTM model more effective.

### 2. **Data Splitting**

The dataset is divided into training and test sets, where the training data is used to fit the model, and the test data is used for evaluation.

- **Training Data**: 80% of the data is used for training.
- **Test Data**: 20% of the data is kept aside for testing.

### 3. **Creating Time Series Data for LSTM Input**

The LSTM model requires a specific format for input. We create sequences of past data to predict the future.

- **Creating Sequences**: The `create_dataset` function transforms the data into sequences of `time_step` length, with each sequence corresponding to a set of features (`X`) and the following value as the target (`Y`).

- **Reshaping Data**: The data is reshaped to be in the format that LSTM models expect:
  

### 4. **Building the LSTM Model**

The LSTM model is constructed using Keras. It consists of an LSTM layer followed by dense layers for output prediction.


- **LSTM Layers**: Two LSTM layers are used, each with 50 units. The first LSTM layer returns sequences, while the second does not.
- **Dropout**: Dropout layers are added to prevent overfitting during training.
- **Dense Layer**: The output layer predicts the next value in the time series.

### 5. **Training the Model**

The model is trained using the training data:

- **Epochs**: The model trains for 20 epochs to learn the patterns in the data.
- **Batch Size**: The batch size is set to 32.

### 6. **Model Evaluation and Prediction**

After training, the model is evaluated using the test data, and predictions are made:


- **Inverse Transform**: The predictions are scaled back to the original range using the inverse of the `MinMaxScaler`.

### 7. **Visualization**

Finally, the results are visualized by comparing the actual and predicted values:


- **Plotting**: The actual and predicted stock prices are plotted to visually assess the model’s accuracy.

## Results

- **Accuracy**: The model successfully captures the trends and patterns in the time series data.
- **Visualization**: The comparison of actual vs. predicted values shows how well the LSTM model performs in forecasting future stock prices.
  
  <img width="446" alt="image" src="https://github.com/user-attachments/assets/a113a6f1-210c-4f04-91d4-8e98257b35e2">


## Future Work

- **Hyperparameter Tuning**: Experiment with different configurations of LSTM layers, units, and learning rates to improve model performance.
- **Incorporating External Factors**: Include additional features like market sentiment or technical indicators to improve the predictions.
- **Model Deployment**: Deploy the trained model as a web application using Flask or FastAPI for real-time predictions.


## Connect with Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/sanjay-karnati)

---

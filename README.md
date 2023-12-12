# Stochastic Time Series Analysis and Forecasting
## An ensemble learning approach
### ML Zoomcamp Capstone Project

**Javier Blanco**

Cohort 2023-2024

# Table of Contents

1. [Project Approach](#project-approach)
2. [Motivation](#motivation)
3. [Synthetic Data and Time Series Dataset](#synthetic-data-and-time-series-dataset)
   1. [Stochastic Differential Equation (SDE) Object](#stochastic-differential-equation-sde-object)
   2. [Simulations Overview](#simulations-overview)
   3. [Simulation Results](#simulation-results)
4. [Technical Solution](#technical-solution)
5. [Software Tools and Models Used](#software-tools-and-models-used)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Model Experiment and Final Hyperparameters](#model-experiment-and-final-hyperparameters)
8. [Results](#results)
9. [Main Conclusions](#main-conclusions)
10. [References](#references)


## Project Approach

For this project, I chose to experiment with time series forecasting methods based on `XGBoost` as an ensemble learner and other methods such as `ARIMA` `ETS`, and `Theta` as base learner models. The goal was to develop a more robust machine learning model capable of achieving superior forecasting performance. To do this I will try the [Montero-Manso et al.(2019)](https://robjhyndman.com/papers/fforma.pdf) article, titled *"FFORMA: Feature-based Forecast Model Averaging"*. Here, they proposed an automated method for obtaining `weighted forecast combinations` using **time series features**. The approach involved two phases. Firstly, a collection of time series was used to train a meta-model, assigning weights to various possible forecasting methods with the aim of minimizing the average forecasting loss from a weighted forecast combination. The inputs to the meta-model were features extracted from each series. In the second phase, new series were forecasted using a `weighted forecast combination`, where the weights were obtained from the previously `trained meta-model`. Their method outperformed a simple forecast combination and surpassed all of the most popular individual methods in the time series forecasting literature. The approach achieved the second position in the M4 competition.

## Motivation


## Synthetic Data and Time Series Dataset

### Stochastic Differential Equation (SDE) Object
$$
dX_t = a(b - X_t) dt + σ* \sqrt(X_t) dW_t
$$

This equation models the dynamics of a stochastic time series according to the [M10a model](Docs/), where:

- `X_t` is the stochastic variable representing the time series.
- `a` and `b` are parameters of the model.
- `σ` is another parameter of the model.
- `dW_t` is a stochastic differential, typically associated with a Wiener process or Brownian motion.


This stochastic differential equation describes how the variable `X_t` evolves over time, with a deterministic term `a(b - X_t) dt` and a stochastic term `σ sqrt(X_t) dW_t`. The presence of stochastic terms reflects randomness in the process and is characteristic of stochastic differential equations (SDEs).

### Simulations Overview

One hundred simulations have been conducted using the stochastic model with the following parameters:

- `a = 1`
- `b = 0.1`
- `sigma = 5`

These simulations aim to generate 100 datasets of time series that emulate the behavior of `weekly` sales for the past `five` years across `100` different stores. The simulations were performed using the Finance Toolbox in `MATLAB R2021b `for the Stochastic Differential Equation (SDE) model [(see here)](matlab/SDE_M10a.m). The Finance Toolbox provides tools and functions tailored for financial modeling, making it a suitable choice for capturing the complexities of sales dynamics in a retail context.

Each simulation provides insights into the stochastic dynamics of the sales process, capturing both deterministic trends and stochastic fluctuations that may resemble real-world scenarios.


### Simulation Results

The simulations of the stochastic differential equation (SDE) with the specified parameters have yielded insightful results. Four key visualizations have been constructed to analyze and interpret the behavior of the simulated time series data:

1. **Simulated Time Series:**

   ![Simulated Time Series](img/simulations.png)

   This plot illustrates the trajectories of the stochastic process across multiple simulations. It provides an overview of the evolving behavior of the time series.

2. **First Derivative:**

   ![First Derivative](img/first_diff.png)

   The first derivative plot depicts the rate of change of the simulated time series. It highlights trends and variations in the slope, offering valuable insights into the underlying dynamics.

3. **Second Derivative:**

   ![Second Derivative](img/second_diff.png)

   The second derivative plot further analyzes the acceleration of the simulated time series. It helps identify points of inflection and provides a deeper understanding of the system's dynamics.

4. **Density Histograms of 5 Datasets:**

   ![Density Histograms](img/kdensity.png)

   This composite plot showcases the density histograms of five datasets generated from the simulations. It offers a visual representation of the distribution of values within each dataset, aiding in the analysis of variability.

These visualizations collectively provide a comprehensive overview of the stochastic dynamics captured by the SDE model.

## Technical Solution

## Software Tools and Models Used

#### Time Series Analysis Models (Base Learners):

1. **AutoArima:**
   - **Description:** Automatic ARIMA model selection for time series forecasting.

2. **ETS (Exponential Smoothing State Space Model):**
   - **Description:** Framework for forecasting that includes simple exponential smoothing among its components.

3. **Theta Model:**
   - **Description:** Simple exponential smoothing model with a damping parameter (Theta) for forecasting.

#### Programming Languages:

4. **Matlab R2021a:**
   - **Description:** Used for modeling Stochastic Differential Equations (SDE) and creating synthetic time series datasets.

5. **Python 3.8.18:**
   - **Description:** Main programming language for data analysis, machine learning, and project implementation.

#### Development Environment:
6. **Anaconda (Miniconda):**
   - **Description:** Python and dependencies distribution.

#### Project Management:
7. **MLFlow:**
   - **Description:** Used for tracking experiments, packaging code into reproducible runs, and sharing and deploying models, specially tunning hyperparameter.

#### Collaborative Development:
8. **Google Colab:**
   - **Description:** Cloud-based platform for Python that allows collaborative editing of Jupyter notebooks.

#### Libraries and Frameworks:
9. **Statsmodel:**
   - **Description:** Utilized for statistical modeling and hypothesis testing, ETS and Theta models were taken from this library.

10. **pmarima:**
   - **Description:** Library for Auto-ARIMA modeling in time series analysis.

11. **scikit-learn:**
   - **Description:** Employed for various machine learning tasks, including Mean Absolute Percentage Error (MAPE) calculation.

12. **Xgboost:**
   - **Description:** Implemented for building an ensemble machine learning model (meta-learner).


## Exploratory Data Analysis

In the Exploratory data analysis (EDA), we conducted a stationary test using the Dickey-Fuller test, identified seasonality and trends, and examined the short-term (t+1) and long-term (t+10) autocorrelation functions (ACF). Despite the synthetic nature of the dataset created using a Stochastic Differential Equation (SDE) model, the EDA notebook provides valuable insights. To explore the notebook with accompanying annotations, [see here]('XD')   

## Model Experiment and Final Hyperparameters

## Results

## Main Conclusions

## REFERENCES

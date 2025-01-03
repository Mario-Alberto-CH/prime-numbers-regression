# Prime Numbers Regression

This project explores the use of machine learning techniques, particularly multi-linear regression, to analyze and approximate the distribution of prime numbers. The dataset was generated using the Sieve of Eratosthenes, with additional features calculated to improve the predictive performance of the model.

## Project Overview
- **Goal**: Predict the behavior of prime numbers using a dataset generated with the Sieve of Eratosthenes.
- **Features**:
  - **Index**: The position of each prime in the sequence.
  - **Prime Density**: The ratio of the prime number to its index (\( \text{Prime} / \text{Index} \)).
  - **Prime Difference**: The difference between consecutive prime numbers.

## Files and Structure
The project files are organized as follows:
- `README.md`: This file, providing an overview of the project.
- `Primes numbers regression.pdf`: A detailed report of the project, including methodology, results, and future work.
- `code and data/`:
  - `prime_analysis.ipynb`: A Jupyter Notebook with step-by-step code and explanations.
  - `prime_analysis.py`: A Python script for generating the dataset and running the analysis.
  - `real_vs_predicted.png`: A visualization of the comparison between real and predicted prime numbers.
  - `residuals.png`: A plot of the residuals (errors) from the model predictions.
  - `prime_numbers_dataset.csv`: The dataset generated using the Sieve of Eratosthenes.

## How to Run
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/prime-numbers-regression.git

# Credit Scoring & Risk Assessment System

## Project Overview
This project uses machine learning to analyze credit risk using the German Credit dataset. It compares multiple classification models and evaluates their performance using accuracy, bad-credit recall, confusion matrices, and a cost-based evaluation.

## Dataset

* **Dataset:** German Credit Data
* **Records:** 1,000
* **Features:** 20 input features
* **Target:** Credit risk classification
* **Source:** UCI Machine Learning Repository — Statlog (German Credit Data)

The dataset is not included in this repository. Please obtain it from the original source and place the required dataset files in the `data/` folder before running the project.

This project uses the dataset for educational and portfolio purposes. Please refer to the original dataset source for its documentation and usage terms.


## Machine Learning Models
The following models are compared:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Cost-Sensitive Logistic Regression

## Model Comparison

| Model | Accuracy | Bad Credit Recall | Total Cost |
|---|---:|---:|---:|
| Logistic Regression | 77.5% | 58.3% | 145 |
| Decision Tree | 64.5% | 46.7% | 199 |
| Random Forest | 76.5% | 43.3% | 183 |
| Cost-Sensitive Logistic Regression | 63.5% | 86.7% | 105 |

## Key Findings
- Logistic Regression achieved the highest accuracy at 77.5%.
- Cost-Sensitive Logistic Regression achieved the highest bad-credit recall at 86.7%.
- Cost-Sensitive Logistic Regression had the lowest total cost under the project's chosen cost evaluation.
- The results demonstrate the trade-off between overall accuracy and identifying bad-credit cases.

## Project Structure

- `main.py` — Main analysis and model training code
- `data/` — Dataset files
- `graphs/` — Visualizations and model evaluation charts
- `requirements.txt` — Required Python libraries
- `cost_sensitive_logistic_regression.pkl` — Saved model
- `preprocessor.pkl` — Saved preprocessing object
- `scaler.pkl` — Saved scaler

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## How to Run

1. Install Python.
2. Clone or download this repository.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Run the project:

   `python main.py`

## Disclaimer
This project is for educational and portfolio purposes only. Its predictions should not be used as the sole basis for real-world lending or credit decisions.

## Author

**Ashutosh Raghav**

 B.Tech CSE GURUGRAM UNIVERSITY

### Connect With Me

* **GitHub:** [aashuraghav807-ux](https://github.com/aashuraghav807-ux)
* **LinkedIn:** [Ashutosh Raghav](https://www.linkedin.com/in/ashutosh-raghav-639a05383/)

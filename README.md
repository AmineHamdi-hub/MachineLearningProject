# Machine Learning Analysis with CRISP-DM

## Overview
This project applies the **CRISP-DM (Cross Industry Standard Process for Data Mining)** methodology to analyze the **Auto.csv** dataset. It includes data preprocessing, exploratory data analysis, and implementation of both supervised and unsupervised learning models.

## Features
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA) with **Boxplots and other visualizations**
- Feature Scaling using **StandardScaler**
- Machine Learning Models:
  - **Supervised Learning:** Regression models (e.g., Linear Regression)
  - **Unsupervised Learning:** Clustering (e.g., K-Means)
- Model Evaluation & Interpretation

## Installation
### Prerequisites
Ensure you have **Python 3.8+** and the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```
2. Run code:
   ```bash
   streamlit run main.py
   ```

## File Structure
```
├── data
│   ├── Auto.csv            # Dataset
├── notebooks
│   ├── EDA.ipynb           # Exploratory Data Analysis
├── src
│   ├── preprocess.py       # Data Cleaning & Preprocessing
│   ├── model.py            # Model Training & Evaluation
├── reports
│   ├── results.pdf         # Findings & Analysis
├── README.md
```

## Authors
- **Amine**

## License
This project is licensed under the MIT License.


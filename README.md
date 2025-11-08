
# EPITRENDS 🦠 Disease Outbreak Prediction App

EPITRENDS is an **interactive Streamlit application** that predicts the risk of **COVID-19** and **H1N1 Flu** infections using machine learning models.  
The app allows users to input personal and health-related data, get predictions, and view dashboards to understand the risk and model accuracies.

---

## 🔹 About the Diseases

### COVID-19
COVID-19 is a highly contagious respiratory disease caused by the SARS-CoV-2 virus. It spreads primarily through droplets and close contact. Predicting COVID-19 risk helps in early isolation and treatment to reduce transmission.

### H1N1 Flu
H1N1, also known as swine flu, is a subtype of the influenza virus. It causes seasonal flu outbreaks and can be severe in vulnerable populations. Early prediction helps in vaccination and preventive measures.

---
> **Note:** All predictions are based on historical data to provide **insights for future risk prevention**.  
---
## 🔹 Why These Models?

For both COVID-19 and H1N1 prediction, the following **machine learning models** were chosen:

1. **Logistic Regression** – Simple and interpretable, good for baseline predictions.  
2. **Random Forest** – Handles non-linear relationships and interactions between features effectively.  
3. **XGBoost** – Powerful ensemble method, often provides the best predictive performance.

Using multiple models allows comparison of predictions and accuracies, helping to choose the **best-performing model** for each disease.

---

## 📂 Project Structure
```
EPITRENDS/
│
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── covid_19_datasets.csv
├── H1N1_Flu_Vaccines.csv
├── covid_19.ipynb
├── H1N1_FLU.ipynb
└── ... # Other scripts/helpers
```
---
## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/Umisalma848/EPITRENDS.git
cd EPITRENDS
```
## create and activate virtual environment
```
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```
## install dependencies
```
pip install -r requirements.txt
```
## Run the app
```
streamlit run app.py
```
- The app will open in your browser.
- Navigate through tabs:
1. About – Information about COVID-19 and H1N1.
2. COVID-19 Prediction – Interactive prediction form for COVID-19.
3. H1N1 Prediction – Interactive prediction form for H1N1 Flu.
4. Dashboard – Visual comparison of predictions and model accuracies.

## 🔹 Download Pre-trained Models

Download the following model files and place them in the **project root folder**:

> **Note:** All the below model files is *.pkl file type

| Model                       | Download Link |
|------------------------------|---------------|
| COVID Logistic Regression    | [Download Link](https://drive.google.com/file/d/1O8QYHh9B0eBXdq9yhZCY1ANnPgiSOiKq/view?usp=sharing) |
| COVID Random Forest          | [Download Link](https://drive.google.com/file/d/1KSegHVb5TKVNou8v6CWTuTqswrFdPg9G/view?usp=sharing) |
| COVID XGBoost                | [Download Link](https://drive.google.com/file/d/1-gRi8QckRK7N6Tr8wgyACJLZv060Z48j/view?usp=sharing) |
| H1N1 Logistic Regression     | [Download Link](https://drive.google.com/file/d/1_0IHx1K_2ZjWcwu6TW1KBSp8Y0Gcj8p4/view?usp=sharing) |
| H1N1 Random Forest           | [Download Link](https://drive.google.com/file/d/1uqx8VI4mIJDPp1YJrInPY-XDkpdfSwkG/view?usp=sharing) |
| H1N1 XGBoost                 | [Download Link](https://drive.google.com/file/d/1Hx0723ATCWLTmPceIjbOdEqGmZMeCT-Z/view?usp=sharing) |

## 📊 Dashboard

- Visualizes predictions for all three models per disease.
- Displays accuracy of each model.
- Allows comparison of COVID-19 and H1N1 risks interactively.
- Helps understand past trends to prevent future outbreaks.

## 💡 Notes

- Predictions are based on historical outbreak data, aiming to provide insights for future risk management.
- Do not include venv/ or .pkl model files in the repository — they are large.
- Use requirements.txt to install dependencies.
- Recommended Python version: 3.8+

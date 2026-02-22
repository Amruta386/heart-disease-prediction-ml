# ❤️ Heart Disease Prediction System

A Machine Learning-based web application that predicts the likelihood of heart disease using multiple classification algorithms and provides insights into major contributing health factors through data visualization.

---

## 📌 Overview

Heart disease remains one of the leading causes of mortality worldwide. Early detection plays a critical role in prevention and treatment.  

This project applies multiple machine learning algorithms to analyze medical parameters and predict whether a patient is likely to have heart disease.

The system:
- Trains and compares multiple ML models
- Selects the best-performing model automatically
- Provides real-time prediction through a web dashboard
- Visualizes key contributing factors using feature importance graphs

---

## 🚀 Key Features

- ✔ Multi-model training and evaluation  
- ✔ Automatic best model selection  
- ✔ Real-time heart disease prediction  
- ✔ Feature importance visualization  
- ✔ Streamlit-based interactive dashboard  
- ✔ Model persistence using Joblib / Keras  

---

## 🧠 Machine Learning Models Used

- Logistic Regression  
- Random Forest Classifier  
- K-Nearest Neighbors (KNN)  
- Naive Bayes (GaussianNB)  
- Support Vector Machine (SVM)  
- Artificial Neural Network (ANN)  

Models are trained and evaluated based on accuracy, and the best-performing model is selected for deployment.

---

## 📊 Dataset

**Dataset:** Cleveland Heart Disease Dataset (UCI Machine Learning Repository)  

- Total Records: 303  
- Total Features: 13  
- Target Variable:
  - `1` → Heart Disease Present  
  - `0` → No Heart Disease  

Key medical parameters include:
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- ECG Results
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression (Oldpeak)
- Number of Major Vessels (CA)
- Thalassemia

---

## 🏗 Project Structure
HEART_DISEASE_PROJECT/
│── heart.csv
│── train_model.py
│── app.py
│── best_model.joblib
│── best_model_ann.keras
│── scaler.joblib
│── accuracy_results.joblib
│── requirements.txt
│── README.md


---

## ⚙ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv

Activate:
Windows
venv\Scripts\activate
Mac/Linux
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Train the Models
python train_model.py

5️⃣ Run the Application
streamlit run app.py

The application will launch at:
http://localhost:8501

🖥 Application Workflow

User enters medical parameters in the dashboard.
Input data is scaled using the saved scaler.
The best-trained model generates a prediction.

The system outputs:
❤️ Heart Disease Present
🍀 No Heart Disease

A feature importance graph highlights major contributing factors.
📈 Model Evaluation

All models are evaluated on test data.
The model with the highest accuracy is selected and saved for deployment.
Random Forest is typically used to compute feature importance and identify key risk factors.

🛠 Technologies Used

Python
Pandas
NumPy
Scikit-learn
TensorFlow / Keras
Matplotlib
Streamlit
Joblib

🔮 Future Enhancements
Add ROC Curve and Confusion Matrix visualization
Hyperparameter tuning for improved accuracy
Deploy on Streamlit Cloud / AWS / Heroku
Add additional medical parameters
Improve UI with advanced visualizations

👩‍💻 Author
Amruta Talawar

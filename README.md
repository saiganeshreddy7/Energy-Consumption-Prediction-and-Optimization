# **🔌 Live Energy Consumption Prediction & Optimization**  

## **📌 Project Overview**  
This project simulates **real-time energy consumption**, predicts energy usage using **Machine Learning**, and provides **optimization insights** through a **live dashboard**. The system uses multiple machine learning models to predict energy consumption based on various environmental factors and provides real-time optimization suggestions.

---

## **🚀 Features**  
✔ **Live Synthetic Data Generation** – Simulates real-time energy consumption.  
✔ **Machine Learning Model** – Predicts energy usage based on environmental factors.  
✔ **FastAPI Prediction API** – Provides real-time predictions via an API.  
✔ **Streamlit Dashboard** – Displays live predictions and energy-saving suggestions.  
✔ **Optimization Insights** – Provides tips to reduce energy consumption.  

---

## **📁 Project Structure**
The project consists of several key components:

### **1. train_model.py**
- Trains multiple machine learning models (Linear Regression, Random Forest, etc.)
- Performs feature engineering and data preprocessing
- Generates performance visualizations
- Saves trained models and performance metrics

### **2. predict_api.py**
- FastAPI backend server for real-time predictions
- Supports multiple ML models with model selection
- Provides energy consumption predictions and optimization tips
- Includes confidence levels and key contributing factors

### **3. data_generator.py**
- Simulates real-time energy consumption data
- Generates realistic sensor readings based on time and conditions
- Simulates different scenarios (energy efficient, high consumption, etc.)

### **4. dashboard.py**
- Interactive Streamlit dashboard
- Real-time energy consumption monitoring
- Historical trends and patterns visualization
- Temperature distribution heatmap
- Energy optimization recommendations

### **Pretrained Models**
The pretrained models are available at:
[Google Drive Link](https://drive.google.com/drive/folders/1qvpqBlfgw3FYvUFNX8oKENjGV00x0NAw?usp=drive_link)

Download and place them in the `models/` directory before running the application.

---

## **🛠 Tools & Technologies Used**  
- **Python** (Main programming language)  
- **FastAPI** (Backend API for real-time predictions)  
- **Streamlit** (Dashboard for visualization)  
- **Scikit-learn** (Machine Learning model training)  
- **Joblib** (Model saving/loading)  
- **Pandas & NumPy** (Data handling)  
- **Requests** (API calls)  

---

## **🖥 How to Run the Project**  

### **1️⃣ Install Dependencies**  
Run the following command in the project directory:  
```bash
pip install -r requirements.txt
```
---

### **2️⃣ Project Startup Sequence**

#### **Step 1: Download Pretrained Models**
- Download models from the provided Google Drive link
- Place them in the `models/` directory of the project

#### **Step 2: Start the FastAPI Prediction Server**  
In your first terminal window:
```bash
uvicorn predict_api:app --reload
```
🔹 **API URL:** `http://127.0.0.1:8000/`  

#### **Step 3: Start the Live Data Generator**  
Open a new terminal tab/window and run:
```bash
python data_generator.py
```
This will simulate real-time energy consumption data every 2 seconds.

#### **Step 4: Launch the Streamlit Dashboard**  
Open a new terminal tab/window and run:
```bash
python -m streamlit run dashboard.py
```
🔹 **Dashboard URL:** Open in browser → `http://localhost:8501/`  

---

## **🛑 How to Stop the Project**  
1. **Stop All Processes:** Press **CTRL + C** in each terminal window to stop the respective services.
2. **Close All Terminal Windows** when finished.

---

## **📡 API Endpoints**  
| Method | Endpoint | Description |  
|--------|---------|-------------|  
| **GET** | `/` | Check API status |  
| **POST** | `/predict/` | Send real-time data and get energy usage prediction |  
| **POST** | `/predict/?model_name=KNN` | Get predictions using a specific model (KNN, RF, etc.) |

---

## **💡 Important Notes**

- The system requires all three components (API, data generator, and dashboard) to run simultaneously
- Make sure to start the API server before the dashboard
- The API supports different ML models via the `model_name` query parameter
- All services must be running for live predictions to work correctly
- Check terminal outputs for any error messages if components aren't working properly

---

## **📈 Future Enhancements**  
- **Deploy to Cloud** (AWS/GCP/Heroku)  
- **Use Real IoT Data** instead of synthetic simulation  
- **Improve Model Accuracy** with better ML techniques  
- **Add Authentication** for API security
- **Implement Data Storage** for historical analysis

---

## **🎯 Author**  
**Sai Ganesh Reddy Kodekandla**  
📧 **saiganeshreddygana@gmail.com**  

---

**Note**: This project simulates real-time energy data. In a production environment, you would connect to actual IoT sensors and energy monitoring devices.

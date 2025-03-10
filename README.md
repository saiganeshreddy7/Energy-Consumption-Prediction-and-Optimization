# **🔌 Live Energy Consumption Prediction & Optimization**  

## **📌 Project Overview**  
This project simulates **real-time energy consumption**, predicts energy usage using **Machine Learning**, and provides **optimization insights** through a **live dashboard**.  

---

## **🚀 Features**  
✔ **Live Synthetic Data Generation** – Simulates real-time energy consumption.  
✔ **Machine Learning Model** – Predicts energy usage based on environmental factors.  
✔ **FastAPI Prediction API** – Provides real-time predictions via an API.  
✔ **Streamlit Dashboard** – Displays live predictions and energy-saving suggestions.  
✔ **Optimization Insights** – Provides tips to reduce energy consumption.  

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

#### **Step 1: Open Project Folder in Terminal**
Navigate to your project directory:
```bash
cd path/to/project
```

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

#### **Step 4: Test the API (Optional)**  
Open a new terminal tab/window and verify the API is working:
```bash
curl -X 'POST' 'http://127.0.0.1:8000/predict/?model_name=KNN' \
     -H 'Content-Type: application/json' \
     -d '{"T1": 20.5, "RH_1": 50, "T2": 19.8, "RH_2": 48, "T3": 21.2, "RH_3": 52,
          "T4": 20.1, "RH_4": 49, "T5": 19.5, "RH_5": 47, "T6": 18.3, "RH_6": 45,
          "T7": 22.4, "RH_7": 55, "T8": 21.8, "RH_8": 53, "T9": 20.0, "RH_9": 50,
          "T_out": 15, "Press_mm_hg": 730, "RH_out": 40, "Windspeed": 3,
          "Visibility": 60, "Tdewpoint": 5, "rv1": 12.5, "rv2": 13.3,
          "hour": 14, "weekday": 3, "month": 6}'
```

#### **Step 5: Launch the Streamlit Dashboard**  
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

📌 **Example Request:**  
See the curl example in Step 4 of the startup sequence.

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
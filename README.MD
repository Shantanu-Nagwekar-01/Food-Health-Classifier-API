# 🥗 Food Health Classifier API

This project predicts whether a food item is **Healthy (1)** or **Unhealthy (0)** based on its nutritional composition.
The final deployed model uses **XGBoost**, which showed the best performance among all tested algorithms.

---

## 🚀 Model Summary

| Model               | Test Accuracy | Precision (Healthy) | Recall (Healthy) |
| ------------------- | ------------- | ------------------- | ---------------- |
| Logistic Regression | 84.15%        | 0.80                | 0.80             |
| Decision Tree       | 81.50%        | 0.75                | 0.79             |
| Random Forest       | 88.40%        | 0.88                | 0.81             |
| AdaBoost            | 88.35%        | 0.86                | 0.83             |
| **XGBoost** ✅       | **89.50%**    | **0.87**            | **0.86**         |

XGBoost achieved the highest test accuracy and balanced precision–recall scores, making it the final model used for deployment.

---

## 🧠 Features Used

| Feature       | Description                      |
| ------------- | -------------------------------- |
| Calories      | Total energy (kcal)              |
| Protein       | Protein content (g)              |
| Carbohydrates | Total carbs (g)                  |
| Fat           | Total fat (g)                    |
| Fiber         | Dietary fiber (g)                |
| Sugars        | Total sugar (g)                  |
| Sodium        | Salt content (mg)                |
| Cholesterol   | Cholesterol level (mg)           |
| Diabetes      | 1 if diabetic, else 0            |
| Obesity       | 1 if obese, else 0               |
| BP            | 1 if high blood pressure, else 0 |

---

## ⚙️ How to Run Locally

### 1️⃣ Clone this repository

```bash
git clone https://github.com/your-username/FoodHealthClassifier.git
cd FoodHealthClassifier
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Start the Flask API

```bash
python app.py
```

### 4️⃣ Test the API

Use **Postman** or **curl** to send a POST request:

```bash
POST http://127.0.0.1:5000/predict
Content-Type: application/json
```

#### Example Input

```json
{
  "Calories": 420,
  "Protein": 20,
  "Carbohydrates": 50,
  "Fat": 22,
  "Fiber": 4,
  "Sugars": 15,
  "Sodium": 400,
  "Cholesterol": 80,
  "bp": 0,
  "diabetes": 0,
  "obesity": 1
}
```

#### Example Output

```json
{
  "prediction": 1,
  "probability": {
    "healthy": 0.7059,
    "unhealthy": 0.2940
  }
}
```

---

## 🧾 Files in Repository

```
├── app.py                     # Flask API file
├── requirements.txt           # Dependencies
├── Model/
│   ├── xgb_food_classifier.pkl
├── Scaler
│   └── feature_scaler.pkl
├── Dataset/
│   └── food_dataset.csv
├──Jupyter Notebook
├── Food_Health_Classifier.ipynb  # Training notebook
└── README.md
```

---

## ☁️ Deployment

The API can be deployed easily on:

* **Render**
* **Vercel**
* **Railway.app**
* or any Flask-compatible hosting service.



---

## 🧩 Author

**Shantanu Nagwekar**


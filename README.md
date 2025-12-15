# Phishing URL Detector

![Project Screenshot](./frontend/image.png)  
_Screenshot of the project interface._

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [How it Works](#how-it-works)
- [Results & Accuracy](#results--accuracy)
- [License](#license)

---

## Project Overview

The **Hybrid Phishing URL Detector** is a web application designed to detect phishing URLs using a machine learning model. It combines traditional URL features and keyword analysis to determine whether a given URL is malicious or safe. This tool helps users quickly check URLs to prevent phishing attacks and cyber fraud.

The project has a **FastAPI backend** for prediction and a **frontend interface** to interact with the model. Accuracy metrics are displayed on the frontend for transparency.

---

## Features

- Predict if a URL is **safe** or **phishing**.
- Displays **model accuracy** and **prediction confidence**.
- Interactive and responsive **frontend**.
- Uses a **hybrid dataset** with features like URL length, number of dots, digits, HTTPS presence, and keywords.
- Fast and efficient predictions using **pre-trained ML model**.
- Can detect phishing attempts targeting **Bangladesh domains** (e.g., `.bd`) as well as global URLs.

---

## Dataset

The dataset contains the following columns:

- `url` – The URL string.
- `url_length` – Length of the URL.
- `num_dots` – Number of dots in the URL.
- `num_digits` – Number of numeric characters.
- `has_https` – Whether the URL uses HTTPS (1 = yes, 0 = no).
- `keyword_bank` – Presence of phishing-related keywords.
- `local_keyword_flag` – Flag for local domain keywords.
- `label` – 0 for safe, 1 for phishing.

The dataset is preprocessed to train the ML model used in this project.

---

## Technologies Used

- **Backend:** FastAPI, Python 3.x
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** scikit-learn, pandas, numpy, scipy
- **Database:** None (model-based detection)
- **Model:** Pre-trained ML model (stored as `model.pkl`)
- **Tools:** Joblib for model serialization

---

## Installation

1. **Clone the repository:**

```bash
https://github.com/ShoaibSikder/Phishing-URL-Detector.git
```

2. **Set up Python environment:**

```bash
python -m venv venv
# Activate virtual environment
# On Linux/macOS
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r backend/requirements.txt
```

---

## Usage

1. **Run the FastAPI backend:**

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
uvicorn backend.app:app --reload --port 8000
```

2. **Open the frontend:**  
   Open `frontend/index.html` in your browser.

3. **Test URLs:**  
   Enter any URL to check if it is safe or phishing. The frontend will display the prediction and model accuracy.

---

## API Endpoints

- **POST `/predict`** – Predicts if a URL is phishing or safe.  
  **Request body:**

```json
{
  "url": "https://google.com/login"
}
```

**Response:**

```json
{
  "prediction": "Safe",
  "accuracy": 0.9
}
```

---

## How it Works

1. User inputs a URL in the frontend.
2. The URL is sent to the FastAPI backend.
3. Backend extracts features:
   - URL length
   - Number of dots
   - Number of digits
   - HTTPS presence
   - Keyword analysis
4. Features are fed into the pre-trained ML model (`model.pkl`).
5. Prediction result and model accuracy are returned and displayed.

---

## Results & Accuracy

- **Model Accuracy:** ~90.5%
- Confusion matrix, ROC curve, and precision-recall analysis can be generated in the backend for evaluation.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

_Developed by Md. Shoaib Sikder._

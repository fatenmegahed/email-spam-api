# Email Spam Classification API

A machine learning API for classifying emails as **Spam** or **Ham** using TF-IDF feature extraction and Multinomial Naive Bayes.

## Project Overview

This project converts a trained NLP classification model into a REST API using FastAPI.

The application receives an email as text and returns a prediction:

* `spam`
* `ham`

## Machine Learning Pipeline

Email Text
↓
TF-IDF Vectorization
↓
Multinomial Naive Bayes
↓
Prediction: Spam / Ham

## Technologies

* Python 3.14
* FastAPI
* Uvicorn
* Scikit-learn
* Joblib
* TF-IDF
* Multinomial Naive Bayes
* Swagger UI

## Project Structure

```text
email_spam_api/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── model/
│   ├── cv.pkl
│   └── spam_ham_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Model Files

`cv.pkl` contains the trained TF-IDF vectorizer.

`spam_ham_model.pkl` contains the trained Multinomial Naive Bayes model.

The label mapping is:

```text
0 = ham
1 = spam
```

## Installation

Create and activate a virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

## Run the API

From the project root:

```bash
python -m uvicorn app.main:app --reload --port 8001
```

The API will be available at:

```text
http://127.0.0.1:8001
```

## Swagger Documentation

FastAPI provides interactive API documentation at:

```text
http://127.0.0.1:8001/docs
```

## API Endpoint

### POST `/predict`

Request:

```json
{
  "email": "Congratulations! You won a free prize. Click here now!"
}
```

Response:

```json
{
  "email": "Congratulations! You won a free prize. Click here now!",
  "prediction": "spam"
}
```

## Example Ham Email

Request:

```json
{
  "email": "Hi, please send me the report for today's meeting."
}
```

Example response:

```json
{
  "email": "Hi, please send me the report for today's meeting.",
  "prediction": "ham"
}
```

## Purpose

This project demonstrates the transition from a trained NLP machine learning model to a deployable API that can be consumed by other applications.

## Future Improvements

* Add input validation and error handling
* Add automated API tests
* Add model performance metrics
* Add a web interface
* Deploy the API to a cloud platform
* Add monitoring and logging

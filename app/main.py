from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib

app=FastAPI()
cv=joblib.load('model/cv.pkl')
model=joblib.load('model/spam_ham_model.pkl')

class EmailRequest(BaseModel):
    email:str

@app.get('/')
def home():
    return{'message':'Email Spam Classification API is running'}

@app.post('/predict')
def predict_email(request:EmailRequest):
    email_vector=cv.transform([request.email])
    prediction=model.predict(email_vector)[0]

    if prediction ==1:
        result ='spam'
    else:
        result ='ham'
    return {
        'email': request.email,
        'prediction': result }
'''from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

cv = joblib.load(BASE_DIR / "model" / "cv.pkl")
model = joblib.load(BASE_DIR / "model" / "spam_ham_model.pkl")


class EmailRequest(BaseModel):
    email: str


@app.get("/")
def home():
    return {"message": "Email Spam Classification API is running"}


@app.post("/predict")
def predict_email(request: EmailRequest):
    email_vector = cv.transform([request.email])
    prediction = model.predict(email_vector)[0]

    if prediction == 1:
        result = "spam"
    else:
        result = "ham"

    return {
        "email": request.email,
        "prediction": result
    }'''

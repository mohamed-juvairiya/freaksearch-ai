# backend/main.py

import os
import joblib
from dotenv import load_dotenv
from pathlib import Path
import mysql.connector
from mysql.connector import Error

from passlib.context import CryptContext

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict

# --- Initial Setup & Configuration ---

# Load environment variables from .env file
load_dotenv()

# --- Password Hashing Setup ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Database Configuration ---
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# Define base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Initialize the FastAPI app
app = FastAPI()


# --- Database Connection ---
def get_db_connection():
    """Creates and returns a new database connection."""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        if conn.is_connected():
            return conn
    except Error as e:
        print(f"❌ Error while connecting to MySQL: {e}")
        return None

# --- Security & Password Hashing ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password):
    return pwd_context.hash(password)

# --- Loading Your Custom AI Model and Vectorizer ---
try:
    vectorizer_path = BASE_DIR / "backend/freaksearch_vectorizer_indian_v1.pkl"
    vectorizer = joblib.load(vectorizer_path)
    model_path = BASE_DIR / "backend/freaksearch_model_indian_v1.pkl"
    model = joblib.load(model_path)
    print("✅ Custom model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    vectorizer = model = None
    print(f"❌ Error loading model: {e}. Intent detection will be skipped.")

def predict_intent_with_custom_model(text: str) -> str:
    """Predicts user intent using your custom model."""
    if model and vectorizer:
        try:
            text_vectorized = vectorizer.transform([text])
            prediction = model.predict(text_vectorized)
            return prediction[0]
        except Exception as e:
            print(f"Error during prediction: {e}")
    return "unknown"

# --- Pydantic Models for Data Validation ---
class UserAuth(BaseModel):
    username: str
    password: str

class ChatPart(BaseModel): text: str
class ChatMessage(BaseModel): role: str; parts: List[ChatPart]
class ChatRequest(BaseModel): message: str; chatHistory: List[ChatMessage]

# --- API Endpoints for User Authentication ---

@app.post("/api/register")
async def register_user(user: UserAuth):
    """Registers a new user in the database."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = %s", (user.username,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists.")
    
    hashed_pw = hash_password(user.password)
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
            (user.username, hashed_pw)
        )
        conn.commit()
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to register user: {e}")
    finally:
        conn.close()
        
    return {"message": f"User '{user.username}' registered successfully."}

@app.post("/api/login")
async def login_user(user: UserAuth):
    """Authenticates a user."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
        
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s", (user.username,))
    db_user = cursor.fetchone()
    conn.close()
    
    if not db_user or not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
        
    return {"message": "Login successful!"}

# --- API Endpoints for Chatbot ---

@app.post("/api/chatbot")
async def handle_chat(request: ChatRequest):
    """
    Main endpoint to handle chat messages using ONLY the custom model.
    """
    user_message = request.message
    
    # 1. Analyze intent with your custom model
    intent = predict_intent_with_custom_model(user_message)
    print(f"Custom model detected intent: {intent}")

    # 2. Map the intent to a predefined response
    #    You can customize these responses as needed.
    if intent == "greeting":
        bot_response = "Hello there! How can I assist you today?"
    elif intent == "goodbye":
        bot_response = "Goodbye! Have a great day."
    elif intent == "about_freaksearch":
        bot_response = "FreakSearch is a platform for verified information."
    elif intent == "fact_check_request":
        bot_response = "My fact-checking capabilities are currently offline."
    else:
        # A default fallback response
        bot_response = "I'm sorry, I don't understand that yet. Can you please rephrase?"
        
    return {"text": bot_response}

@app.post("/api/upload-media")
async def upload_media(file: UploadFile = File(...)):
    """Handles file uploads from the chat interface."""
    file_path = UPLOADS_DIR / file.filename
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    return {"message": f"File '{file.filename}' uploaded successfully."}

# --- Serving Frontend Files ---

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
@app.get("/")
async def serve_landing_page():
    return FileResponse(os.path.join(STATIC_DIR, "landing.html"))
@app.get("/chat")
async def serve_chat_page():
    return FileResponse(os.path.join(STATIC_DIR, "chat.html"))
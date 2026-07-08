import logging
import time
from datetime import datetime, timezone
import jwt
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session  # REFACTOR: Modern declarative_base import

from config import settings

# Logging Configuration
logging.basicConfig(
    filename="sentiment_activity.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api")

# Initial startup logs
logger.info("--- Starting Airline Sentiment API Service ---")

# Database Setup
try:
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()  # Modern stable approach
    logger.info("Database engine initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize database engine: {e}")

# Prediction Log Model for Database
class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text)
    sentiment = Column(String(50))
    confidence = Column(Float)
    # REFACTOR: Using timezone-aware UTC datetime for future-proofing and modern Python standards
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Create tables in the database
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified/created.")
except Exception as e:
    logger.error(f"Error creating database tables: {e}")

# Pydantic Schemas for Request and Response
class TweetRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    inference_time: float

# Load the MLflow model from the registered URI
logger.info(f"Attempting to load MLflow model from: {settings.MODEL_URI}")
try:
    mlflow_model = mlflow.pyfunc.load_model(settings.MODEL_URI)
    logger.info("🚀 MLflow Model Loaded successfully into API context.")
except Exception as e:
    mlflow_model = None
    logger.error(f"❌ Model Load Error: {e}")

# Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Identify user for Rate Limiting (JWT sub or IP address)
def get_smart_identifier(request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            # Decode token safely without signature verification just to extract user metadata
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get("sub")
            if user_id:
                return f"user:{user_id}"
        except Exception as e:
            logger.debug(f"JWT decode failed: {e}")
    return f"ip:{get_remote_address(request)}"

# Initialize Rate Limiter
limiter = Limiter(key_func=get_smart_identifier)
app = FastAPI(title="Airline Sentiment Enterprise API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
logger.info("CORS and Rate Limiter configured.")

# Middleware to calculate and add processing time to headers
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Main Prediction Endpoint
@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
async def predict(
        payload: TweetRequest,
        request: Request,
        background_tasks: BackgroundTasks,
):
    if mlflow_model is None:
        logger.error("Prediction failed: Model is not loaded (None).")
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    try:
        start_time = time.time()
        logger.info(f"Prediction started for text: {payload.text[:50]}...")
        
        # Prepare input for MLflow model (Perfect matching with PyFunc DataFrame expectation)
        df = pd.DataFrame({"text": [payload.text]})
        result_df = mlflow_model.predict(df)

        sentiment = str(result_df["label"].iloc[0])
        confidence = float(result_df["confidence"].iloc[0])
        inference_time = round(time.time() - start_time, 4)

        logger.info(f"Inference complete: Sentiment={sentiment}, Confidence={confidence:.2f}")

        # Log to database as a background task to keep API highly responsive under load
        background_tasks.add_task(db_log_prediction, payload.text, sentiment, confidence)

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "inference_time": inference_time
        }
    except Exception as e:
        logger.error(f"Prediction Runtime Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

# Helper function to save prediction results to the database
def db_log_prediction(text: str, sentiment: str, confidence: float):
    new_db = SessionLocal()
    try:
        log_entry = PredictionLog(text=text, sentiment=sentiment, confidence=confidence)
        new_db.add(log_entry)
        new_db.commit()
        logger.info(f"Database log entry created for sentiment: {sentiment}")
    except Exception as e:
        logger.error(f"Background database logging failed: {str(e)}")
    finally:
        new_db.close()

# Endpoint to fetch prediction history
@app.get("/predictions")
def get_predictions(db: Session = Depends(get_db)):
    logger.info("Fetching history logs from database.")
    try:
        # Retrieve the latest 50 predictions to power dashboard view components
        logs = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(50).all()
        return logs
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch history from database")

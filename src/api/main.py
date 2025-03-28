import os
from fastapi import FastAPI, HTTPException, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import datetime
import time
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analytics.analytics import BookingAnalytics
from rag.vector_store import VectorStore
from rag.llm_interface import LLMInterface

app = FastAPI(
    title="Hotel Booking System API",
    description="API for hotel booking analytics and question answering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyticsRequest(BaseModel):
    metric: Optional[str] = None
    time_period: Optional[str] = "M"

class AnalyticsResponse(BaseModel):
    results: Dict[str, Any]
    processing_time: float

class QuestionRequest(BaseModel):
    query: str

class QuestionResponse(BaseModel):
    answer: str
    processing_time: float

class QueryHistoryItem(BaseModel):
    query: str
    answer: str
    timestamp: str

# Global variables
base_dir = Path(__file__).resolve().parent.parent.parent
data_path = base_dir / 'data' / 'processed' / 'hotel_bookings_processed.csv'
vector_store_path = base_dir / 'data' / 'vector_store'

# Create directories if they don't exist
os.makedirs(base_dir / 'data' / 'processed', exist_ok=True)
os.makedirs(vector_store_path, exist_ok=True)
os.makedirs(base_dir / 'logs', exist_ok=True)

# Initialize components when they are needed
analytics = None
vector_store = None
llm_instance = None
query_history = []

# Helper functions
def get_analytics():
    global analytics
    if analytics is None:
        analytics = BookingAnalytics(data_path)
    return analytics

def get_vector_store():
    global vector_store
    if vector_store is None:
        vector_store = VectorStore(data_path)
        vector_store.create_chroma_collection(collection_name="hotel_bookings", save_path=str(vector_store_path))
    return vector_store

def get_llm_interface():
    global llm_instance
    
    if llm_instance is None:
        try:
            # Get the vector store using existing function
            vector_store = get_vector_store()
            
            # Get Mistral API key from environment
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment variables")
            
            # Initialize the LLM interface with vector store
            model_name = os.environ.get("MISTRAL_MODEL_NAME", "mistral-medium")
            
            # Create instance with vector store already obtained
            llm_instance = LLMInterface(model_name=model_name, vector_store=vector_store)
            
            # Load model (tests API connection)
            llm_instance.load_model()
            
            print("LLM interface initialized successfully")
        except Exception as e:
            print(f"Failed to initialize LLM interface: {e}")
            raise ValueError(f"LLM initialization failed: {e}")
    
    return llm_instance

def log_query(query, answer):
    global query_history
    query_history.append({
        "query": query,
        "answer": answer,
        "timestamp": datetime.datetime.now().isoformat()
    })
    # Keep only the last 100 queries
    if len(query_history) > 100:
        query_history = query_history[-100:]

# API routes
@app.get("/")
def read_root():
    return {"message": "Welcome to the Hotel Booking System API"}

@app.post("/analytics", response_model=AnalyticsResponse)
def get_analytics_endpoint(request: AnalyticsRequest = Body(...)):
    start_time = time.time()
    
    try:
        analytics_instance = get_analytics()
        
        if request.metric is None:
            results = analytics_instance.generate_all_analytics()
        elif request.metric == "revenue_trends":
            results = {request.metric: analytics_instance.revenue_trends(request.time_period)}
        elif request.metric == "cancellation_rate":
            results = {request.metric: analytics_instance.cancellation_rate()}
        elif request.metric == "geographical_distribution":
            results = {request.metric: analytics_instance.geographical_distribution()}
        elif request.metric == "lead_time_distribution":
            results = {request.metric: analytics_instance.lead_time_distribution()}
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric: {request.metric}"
            )
        
        processing_time = time.time() - start_time
        
        return {
            "results": results,
            "processing_time": round(processing_time, 3)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing analytics: {str(e)}"
        )

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest = Body(...)):
    start_time = time.time()
    
    try:
        llm = get_llm_interface()
        response = llm.answer_with_rag(request.query)
        answer = response["answer"]
        
        # Log the query
        log_query(request.query, answer)
        
        processing_time = time.time() - start_time
        
        return {
            "answer": answer,
            "processing_time": round(processing_time, 3)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error answering question: {str(e)}"
        )

@app.get("/query-history", response_model=List[QueryHistoryItem])
def get_query_history():
    return query_history

@app.get("/health")
def health_check():
    health_status = {
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "components": {
            "api": "healthy",
            "analytics": "unknown",
            "vector_store": "unknown",
            "llm": "unknown"
        }
    }
    
    # Check analytics
    try:
        analytics_instance = get_analytics()
        health_status["components"]["analytics"] = "healthy"
    except:
        health_status["components"]["analytics"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check vector store
    try:
        vector_store_instance = get_vector_store()
        health_status["components"]["vector_store"] = "healthy"
    except:
        health_status["components"]["vector_store"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check LLM API
    global llm_instance
    if llm_instance is not None:
        if llm_instance.is_healthy():
            health_status["components"]["llm"] = "healthy"
        else:
            health_status["components"]["llm"] = "unhealthy"
            health_status["status"] = "degraded"
    else:
        health_status["components"]["llm"] = "not_initialized"
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

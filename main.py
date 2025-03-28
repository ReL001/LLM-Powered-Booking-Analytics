import os
import argparse
import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from src.data_processing.preprocess import DataPreprocessor
from src.analytics.analytics import BookingAnalytics
from src.rag.vector_store import VectorStore
from src.rag.llm_interface import LLMInterface

vector_store = None

def setup_directories():
    base_dir = Path(__file__).resolve().parent
    
    directories = [
        base_dir / 'data' / 'raw',
        base_dir / 'data' / 'processed',
        base_dir / 'data' / 'vector_store',
        base_dir / 'reports' / 'visualizations',
        base_dir / 'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    return base_dir

def check_dataset(base_dir):
    raw_data_path = base_dir / 'data' / 'raw' / 'hotel_bookings.csv'
    
    if not os.path.exists(raw_data_path):
        print("Dataset not found. Please download the dataset and place it in the 'data/raw' directory.")
        print("You can download the dataset from: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand")
        return False
        
    return True

def process_data(base_dir):
    raw_data_path = base_dir / 'data' / 'raw' / 'hotel_bookings.csv'
    processed_data_path = base_dir / 'data' / 'processed' / 'hotel_bookings_processed.csv'
    
    print("Processing data...")
    preprocessor = DataPreprocessor(raw_data_path)
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.add_derived_features()
    preprocessor.save_processed_data(processed_data_path)
    print("Data processing completed.")
    
    return processed_data_path

def generate_analytics(processed_data_path, base_dir):
    visualizations_dir = base_dir / 'reports' / 'visualizations'
    
    print("Generating analytics...")
    analytics = BookingAnalytics(processed_data_path)
    analytics.save_analytics_visualizations(visualizations_dir)
    print(f"Analytics and visualizations saved to {visualizations_dir}")
    
    return analytics

def create_vector_store(processed_data_path, base_dir):
    vector_store_path = base_dir / 'data' / 'vector_store'
    
    print("Creating vector store...")
    vector_store = VectorStore(processed_data_path)
    vector_store.create_chroma_collection(collection_name="hotel_bookings", save_path=str(vector_store_path))
    print(f"Vector store created at {vector_store_path}")
    
    return vector_store

def get_vector_store():
    global vector_store
    if vector_store is None:
        # Use ChromaDB
        base_dir = Path(__file__).resolve().parent
        vector_store_path = base_dir / 'data' / 'vector_store'
        processed_data_path = base_dir / 'data' / 'processed' / 'hotel_bookings_processed.csv'
        
        vector_store = VectorStore(processed_data_path)
        
        vector_store.create_chroma_collection(collection_name="hotel_bookings", save_path=str(vector_store_path))
            
    return vector_store

def run_api(port=8000):
    from src.api.main import app
    import uvicorn
    
    print(f"Starting API server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

def main():
    parser = argparse.ArgumentParser(description="Hotel Booking System")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument("--analytics", action="store_true", help="Generate analytics")
    parser.add_argument("--vector-store", action="store_true", help="Create vector store")
    parser.add_argument("--api", action="store_true", help="Run the API server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline")
    
    args = parser.parse_args()
    
    if not any([args.preprocess, args.analytics, args.vector_store, args.api, args.all]):
        args.all = True
    
    base_dir = setup_directories()
    
    if args.all or args.preprocess:
        if check_dataset(base_dir):
            processed_data_path = process_data(base_dir)
        else:
            return
    
    processed_data_path = base_dir / 'data' / 'processed' / 'hotel_bookings_processed.csv'
    
    if args.all or args.analytics:
        if os.path.exists(processed_data_path):
            generate_analytics(processed_data_path, base_dir)
        else:
            print(f"Processed data not found at {processed_data_path}. Run preprocessing first.")
            return
    
    if args.all or args.vector_store:
        if os.path.exists(processed_data_path):
            create_vector_store(processed_data_path, base_dir)
        else:
            print(f"Processed data not found at {processed_data_path}. Run preprocessing first.")
            return
    
    if args.all or args.api:
        run_api(port=args.port)

if __name__ == "__main__":
    main()

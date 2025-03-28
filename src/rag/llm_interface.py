import os
import requests
from .vector_store import VectorStore
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

class LLMInterface:
    def __init__(self, model_name="mistral-medium", vector_store=None):
        self.model_name = model_name
        self.vector_store = vector_store
        
        # API settings
        self.api_key = os.environ.get("MISTRAL_API_KEY", "")
        self.api_url = os.environ.get("MISTRAL_API_URL", "https://api.mistral.ai/v1/chat/completions")
        
        # Check if API key is provided
        if not self.api_key:
            print("No API key")
        
        self.is_loaded = False
        
    def load_model(self):
        print(f"Using Mistral API with model {self.model_name}")
        
        # Validate API key is set
        if not self.api_key:
            raise ValueError("API key not provided.")
        
        # Test API connection to verify credentials
        self.is_healthy()
        
        self.is_loaded = True
        print(f"API connection validated successfully.")
    
    def generate_response(self, query, max_new_tokens=512, temperature=0.7):

        if not self.is_loaded:
            raise ValueError("API connection not initialized.")
        
        if not self.api_key:
            raise ValueError("API key not provided.")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.95
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API request failed: {e}")
    
    def retrieve_relevant_context(self, query, top_k=3):
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        # Query the vector store
        results = self.vector_store.query(query, top_k=top_k)
        
        # Format the context from the results
        context = "Based on the hotel booking data, here are some relevant records:\n\n"
        
        for i, result in enumerate(results):
            context += f"Record {i+1}:\n"
            for key, value in result['metadata'].items():
                if key in ['adr', 'lead_time', 'arrival_date', 'arrival_date_year', 'arrival_date_month', 'total_revenue', 'country', 'is_canceled', 'reservation_status', 'hotel', 'stays_in_weekend_nights', 'stays_in_week_nights', 'total_nights']:
                    context += f"- {key}: {value}\n"
            context += "\n"
        
        return context
    
    def answer_with_rag(self, query, top_k=3):
        # Retrieve relevant context
        context = self.retrieve_relevant_context(query, top_k=top_k)
        
        # Craft the prompt with the context and better instructions
        prompt = f"""
You are an AI assistant that helps analyze hotel booking data.
Use the following retrieved records to answer the question accurately.
If the answer cannot be found in the records, say so clearly.
Be concise but thorough and provide numerical data when relevant.

{context}

Question: {query}

Answer:
"""
        
        # Generate response
        answer = self.generate_response(prompt)
        
        return {
            'query': query,
            'context': context,
            'answer': answer
        }
    
    def is_healthy(self) -> bool:
        # Check if API key is available
        if not self.api_key:
            return False
            
        # Try to make a simple API call
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers)
            return response.status_code == 200
        except Exception as e:
            print(f"API health check failed: {e}")
            return False

def main():
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    processed_data_path = base_dir / 'data' / 'processed' / 'hotel_bookings_processed.csv'
    
    # For ChromaDB
    chroma_path = base_dir / 'data' / 'vector_store' / 'chroma_collection'
    
    # Check if vector store exists
    if not os.path.exists(chroma_path):
        print(f"Vector store not found at {chroma_path}")
        return
    
    # Initialize vector store
    vector_store = VectorStore(processed_data_path)
    
    # Initialize LLM interface
    llm = LLMInterface(model_name="mistral-medium", vector_store=vector_store)
    
    try:
        llm.load_model()
        
        # Test some queries
        test_queries = [
            "What is the average price of a hotel booking?",
            "Which locations had the highest booking cancellations?",
            "Show me total revenue for July 2017."
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = llm.answer_with_rag(query)
            print(f"Answer: {response['answer']}")
            
    except Exception as e:
        print(f"Error using API: {e}")

if __name__ == "__main__":
    main()

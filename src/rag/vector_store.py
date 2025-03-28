import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import pickle
import json
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.embeddings = None
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.chroma_client = None
        self.collection = None

        # Only load data if the path exists
        if os.path.exists(data_path):
            self.load_data()
        else:
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        
    def load_data(self):
        file_extension = os.path.splitext(self.data_path)[1].lower()
        
        if file_extension == '.csv':
            self.data = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        print(f"Loaded data for vector store with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
    
    def prepare_documents(self):
        documents = []
        metadata = []
        
        for _, row in self.data.iterrows():
            doc_text = ""
            doc_meta = {}
            
            for col, value in row.items():
                if pd.notna(value):
                    doc_text += f"{col}: {value}, "
                    doc_meta[col] = value
            
            documents.append(doc_text.strip(", "))
            metadata.append(doc_meta)
            
        return documents, metadata
    
    def create_chroma_collection(self, collection_name="hotel_bookings", save_path=None):
        # Initialize the client based on whether we want persistence
        if save_path:
            path = os.path.join(save_path, "chroma_db")
            os.makedirs(path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=path)
        else:
            self.chroma_client = chromadb.Client()
        
        # Get all collections to check if our collection exists
        all_collections = self.chroma_client.list_collections()
        collection_exists = any(col.name == collection_name for col in all_collections)
        
        try:
            if collection_exists:
                self.collection = self.chroma_client.get_collection(name=collection_name)
                print(f"Collection '{collection_name}' found. Using existing collection.")
                return self.collection
        except Exception as e:
            print(f"Collection '{collection_name}' not found or error accessing it: {e}. Creating new collection.")
            
        # Prepare documents and metadata
        documents, metadata = self.prepare_documents()
        document_ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Embedding function
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-MiniLM-L6-v2" 
        )
        
        # Create new collection
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef,
            metadata={"description": "Hotel booking data"}
        )
        
        # Add documents to collection
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end_idx],
                ids=document_ids[i:end_idx],
                metadatas=metadata[i:end_idx]
            )
            
            print(f"Added batch {i//batch_size + 1} to ChromaDB collection")
        
        print(f"Created ChromaDB collection '{collection_name}' with {len(documents)} documents")
        
        return self.collection
    
    def query(self, query_text, top_k=5):
        if self.collection is None:
            raise ValueError("ChromaDB collection not initialized.")
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    processed_data_path = base_dir / 'data' / 'processed' / 'hotel_bookings_processed.csv'
    vector_store_dir = base_dir / 'data' / 'vector_store'
    
    # Check if the data file exists
    if not processed_data_path.exists():
        print(f"Error: Data file not found at {processed_data_path}")
        return
        
    # Create vector store instance
    try:
        store = VectorStore(str(processed_data_path))
        
        # Create directory for vector store
        os.makedirs(vector_store_dir, exist_ok=True)
        
        # Create collection
        store.create_chroma_collection(collection_name="hotel_bookings", save_path=str(vector_store_dir))
        
        print("\nVector store creation completed successfully.")
    except Exception as e:
        print(f"Error creating vector store: {e}")

if __name__ == "__main__":
    main()
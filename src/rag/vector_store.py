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
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.chroma_client = None
        self.collection = None

        self.load_data()
        
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
        documents, metadata = self.prepare_documents()
        document_ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Initialize ChromaDB client
        if save_path:
            self.chroma_client = chromadb.PersistentClient(path=save_path)
        else:
            self.chroma_client = chromadb.Client()
        
        # Embedding function
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )
        
        try:
            self.chroma_client.delete_collection(collection_name)
        except:
            pass
        
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
    
    os.makedirs(vector_store_dir, exist_ok=True)
    
    # Create vector store
    store = VectorStore(processed_data_path)
    store.create_chroma_collection(save_path=str(vector_store_dir / "chroma_collection"))
    
    # Test a query
    results = store.query("Show me bookings from Portugal with high ADR", top_k=3)
    print("\nQuery results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result['document'][:100]}... (Score: {result['score']})")
    
    print("\nVector store creation completed successfully.")

if __name__ == "__main__":
    main()

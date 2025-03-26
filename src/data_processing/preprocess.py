import os
import pandas as pd
import numpy as np
from pathlib import Path

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        
    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded data with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
        return self.data
    
    def clean_data(self):

        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create a copy for processing
        df = self.data.copy()
        
        # Handle missing values
        print("Missing values before cleaning:")
        print(df.isnull().sum())
        
        # Fill numeric columns with appropriate values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Convert date columns to datetime if they exist
        date_columns = [col for col in df.columns if 'date' in col.lower() or 
                        'arrival' in col.lower() or 'departure' in col.lower()]
        
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                print(f"Could not convert column '{col}' to datetime.")
        
        print("Missing values after cleaning:")
        print(df.isnull().sum())
        
        self.processed_data = df
        return self.processed_data
    
    def add_derived_features(self):
        if self.processed_data is None:
            raise ValueError("Data not cleaned. Call clean_data() first.")
        
        df = self.processed_data.copy()
        
        # Add booking lead time if not already present
        if 'reservation_status_date' in df.columns and 'arrival_date' in df.columns:
            df['booking_lead_time'] = (df['arrival_date'] - df['reservation_status_date']).dt.days
        
        # Add total revenue column if not already present
        if 'adr' in df.columns and 'stays_in_weekend_nights' in df.columns and 'stays_in_week_nights' in df.columns:
            df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
            df['total_revenue'] = df['adr'] * df['total_nights']
        
        self.processed_data = df
        return self.processed_data
    
    def save_processed_data(self, output_path):

        if self.processed_data is None:
            raise ValueError("No processed data to save. Run the preprocessing first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        file_extension = os.path.splitext(output_path)[1].lower()
        
        if file_extension == '.csv':
            self.processed_data.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output file format: {file_extension}")
            
        print(f"Processed data saved to {output_path}")

def main():
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    # data_dir = base_dir / 'data'
    raw_data_path = data_dir / 'raw' / 'hotel_bookings.csv'
    processed_data_path = base_dir / 'data' / 'processed' / 'hotel_bookings_processed.csv'
    
    # Create directories if they don't exist
    os.makedirs(data_dir / 'raw', exist_ok=True)
    os.makedirs(data_dir / 'processed', exist_ok=True)
    
    # Check if data file exists
    if not os.path.exists(raw_data_path):
        print(f"Data file not found at {raw_data_path}")
        return
    
    # Initialize and run the data preprocessor
    preprocessor = DataPreprocessor(raw_data_path)
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.add_derived_features()
    preprocessor.save_processed_data(processed_data_path)
    
    print("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()

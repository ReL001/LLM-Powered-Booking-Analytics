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
            raise ValueError("Data not loaded.")
        
        df = self.data.copy()
        
        # Handle missing values
        print("Missing values before cleaning:")
        print(df.isnull().sum())
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        if 'reservation_status_date' in df.columns:
            df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')
        else:
            print("Column 'reservation_status_date' not found.")
        
        # Convert date columns to datetime
        if all(col in df.columns for col in ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']):
            month_map = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            df['arrival_month_number'] = df['arrival_date_month'].map(month_map)
            df['arrival_date'] = pd.to_datetime(
                df['arrival_date_year'].astype(str) + '-' +
                df['arrival_month_number'].astype(str) + '-' +
                df['arrival_date_day_of_month'].astype(str),
                errors='coerce'
            )
            df.drop('arrival_month_number', axis=1, inplace=True)
        else:
            print("Required columns for arrival_date not found.")
        
        print("Missing values after cleaning:")
        print(df.isnull().sum())
        
        self.processed_data = df
        return self.processed_data
    
    def add_derived_features(self):
        if self.processed_data is None:
            raise ValueError("Data not cleaned.")
        
        df = self.processed_data.copy()
        
        # Calculate total nights and revenue
        if 'stays_in_weekend_nights' in df.columns and 'stays_in_week_nights' in df.columns:
            df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
            if 'adr' in df.columns:
                df['total_revenue'] = df['adr'] * df['total_nights']
        
        self.processed_data = df
        return self.processed_data
    
    def save_processed_data(self, output_path):
        if self.processed_data is None:
            raise ValueError("No processed data to save. Run preprocessing first.")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.processed_data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / 'data'
    raw_data_path = data_dir / 'raw' / 'hotel_bookings.csv'
    processed_data_path = data_dir / 'processed' / 'hotel_bookings_processed.csv'
    
    os.makedirs(data_dir / 'processed', exist_ok=True)
    
    if not raw_data_path.exists():
        print(f"Data not found at {raw_data_path}")
        return
    
    preprocessor = DataPreprocessor(raw_data_path)
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.add_derived_features()
    preprocessor.save_processed_data(processed_data_path)
    print("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()
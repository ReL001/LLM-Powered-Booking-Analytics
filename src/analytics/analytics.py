import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

class BookingAnalytics:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.load_data()
        
    def load_data(self):

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        try:
            self.data = pd.read_csv(self.data_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
            
        print(f"Loaded data from CSV with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
        return self.data
    
    def revenue_trends(self):
        if 'arrival_date_year' in self.data.columns and 'arrival_date_month' in self.data.columns:
            if 'total_revenue' not in self.data.columns and 'adr' in self.data.columns:
                if 'stays_in_weekend_nights' in self.data.columns and 'stays_in_week_nights' in self.data.columns:
                    self.data['total_nights'] = self.data['stays_in_weekend_nights'] + self.data['stays_in_week_nights']
                    self.data['total_revenue'] = self.data['adr'] * self.data['total_nights']
                else:
                    self.data['total_revenue'] = self.data['adr']  # Use ADR as a proxy
                    
            revenue_trends = self.data.groupby(['arrival_date_year', 'arrival_date_month'])['total_revenue'].sum().reset_index()
            revenue_trends['date'] = pd.to_datetime(revenue_trends['arrival_date_year'].astype(str) + '-' + revenue_trends['arrival_date_month'], format='%Y-%B')
            revenue_trends = revenue_trends.sort_values('date')
            
            return revenue_trends
        else:
            raise ValueError("Required columns 'arrival_date_year' and 'arrival_date_month' not found in data.")
    
    def hotel_distribution(self):
        if 'hotel' in self.data.columns:
            hotel_counts = self.data['hotel'].value_counts()
            return hotel_counts
        else:
            raise ValueError("Required column 'hotel' not found in data.")
    
    def cancellation_rates(self):
        if 'hotel' in self.data.columns and 'is_canceled' in self.data.columns:
            hotel_cancellation = self.data.groupby('hotel')['is_canceled'].agg(['count', 'sum'])
            hotel_cancellation['cancellation_rate'] = (hotel_cancellation['sum'] / hotel_cancellation['count'] * 100).round(2)
            return hotel_cancellation
        else:
            raise ValueError("Required columns 'hotel' and 'is_canceled' not found in data.")

    def geographical_distribution(self):
        if 'country' in self.data.columns:
            top_countries = self.data['country'].value_counts().head(10)
            return top_countries
        else:
            raise ValueError("Required column 'country' not found in data.")
    
    def lead_time_distribution(self):
        if 'lead_time' in self.data.columns:
            lead_filtered = self.data[self.data['lead_time'] < self.data['lead_time'].quantile(0.99)]
            return lead_filtered
        else:
            raise ValueError("Required column 'lead_time' not found in data.")
    
    def save_analytics_visualizations(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Revenue trends over time
        try:
            revenue_trends = self.revenue_trends()
            plt.figure(figsize=(14, 8))
            plt.plot(revenue_trends['date'], revenue_trends['total_revenue'], marker='o')
            plt.title('Revenue Trends Over Time')
            plt.xlabel('Date')
            plt.ylabel('Total Revenue')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'revenue_trends.png'))
            plt.close()
            print("Revenue trends visualization saved.")
        except Exception as e:
            print(f"Error: {e}")
        
        # 2. Distribution of hotel types
        try:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.data, x='hotel')
            plt.title('Distribution of Hotel Types')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'hotel_distribution.png'))
            plt.close()
            print("Hotel distribution visualization saved.")
        except Exception as e:
            print(f"Error: {e}")
        
        # 3. Cancellation rates by hotel type
        try:
            hotel_cancellation = self.cancellation_rates()
            plt.figure(figsize=(10, 6))
            bars = plt.bar(hotel_cancellation.index, hotel_cancellation['cancellation_rate'])
            
            # Percentage labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height}%',
                        ha='center', va='bottom')
            
            plt.title('Cancellation Rates')
            plt.xlabel('Hotel Type')
            plt.ylabel('Cancellation Rate (%)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cancellation_rates.png'))
            plt.close()
            print("Cancellation rates visualization saved.")
        except Exception as e:
            print(f"Error: {e}")

        # 4. Geographical distribution of users
        try:
            top_countries = self.geographical_distribution()
            plt.figure(figsize=(12, 8))
            sns.barplot(x=top_countries.values, y=top_countries.index)
            plt.title('Geographical distribution of users')
            plt.xlabel('Number of Bookings')
            plt.ylabel('Country')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'geographical_distribution.png'))
            plt.close()
            print("Geographical distribution visualization saved.")
        except Exception as e:
            print(f"Error: {e}")
        
        # 5. Lead time distribution
        try:
            lead_filtered = self.lead_time_distribution()
            plt.figure(figsize=(12, 6))
            sns.histplot(data=lead_filtered, x='lead_time', kde=True, bins=50)
            plt.title('Distribution of Lead Time (days between booking and arrival)')
            plt.xlabel('Lead Time (days)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'lead_time_distribution.png'))
            plt.close()
            print("Lead time distribution visualization saved.")
        except Exception as e:
            print(f"Error: {e}")
        
        print(f"Analytics visualizations saved to {output_dir}")

    def generate_all_analytics(self):
        results = {}
        
        try:
            results["revenue_trends"] = self.revenue_trends()
        except Exception as e:
            results["revenue_trends"] = {"error": str(e)}
            
        try:
            results["hotel_distribution"] = self.hotel_distribution().to_dict()
        except Exception as e:
            results["hotel_distribution"] = {"error": str(e)}
            
        try:
            results["cancellation_rate"] = self.cancellation_rates().to_dict()
        except Exception as e:
            results["cancellation_rate"] = {"error": str(e)}
            
        try:
            results["geographical_distribution"] = self.geographical_distribution().to_dict()
        except Exception as e:
            results["geographical_distribution"] = {"error": str(e)}
            
        try:
            lead_filtered = self.lead_time_distribution()
            results["lead_time_distribution"] = {
                "mean": lead_filtered["lead_time"].mean(),
                "median": lead_filtered["lead_time"].median(),
                "min": lead_filtered["lead_time"].min(),
                "max": lead_filtered["lead_time"].max()
            }
        except Exception as e:
            results["lead_time_distribution"] = {"error": str(e)}
            
        return results

def main():
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    processed_data_path = base_dir / 'data' / 'processed' / 'hotel_bookings_processed.csv'
    visualizations_dir = base_dir / 'reports' / 'visualizations'
    
    os.makedirs(visualizations_dir, exist_ok=True)
    
    analytics = BookingAnalytics(processed_data_path)
    analytics.save_analytics_visualizations(visualizations_dir)
    
    print("Analytics completed successfully.")

if __name__ == "__main__":
    main()

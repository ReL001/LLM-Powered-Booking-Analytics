# Hotel Booking System

A system for processing hotel booking data, extracting insights, and enabling retrieval-augmented question answering (RAG).

## Quick Start

After installation (see [Setup and Installation](#setup-and-installation)):

```bash
# Run the complete pipeline (processing, analytics, and start the API)
python main.py --all

# Access the API at http://localhost:8000
```

## Features

1. **Data Processing**
   - Cleans and preprocesses hotel booking data
   - Handles missing values and outliers
   - Adds derived features for better analysis

2. **Analytics & Reporting**
   - Revenue trends over time
   - Cancellation rates
   - Geographical distribution of users
   - Lead time distribution
   - Visualizations for key metrics

3. **Retrieval-Augmented Question Answering (RAG)**
   - Vector database for efficient retrieval
   - Natural language Q&A using Mistral API
   - Contextual answers based on booking data

4. **API**
   - FastAPI endpoints for analytics and Q&A
   - Health check endpoint
   - Query history tracking

## Setup and Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd hotel-booking-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the dataset:
   - Get the Hotel Booking Demand dataset from [Kaggle](https://www.kaggle.com/jessemostipak/hotel-booking-demand)
   - Place the CSV file in the `data/raw/` directory with the name `hotel_bookings.csv`

4. Set up environment variables:
   Create a `.env` file with:
   ```
   MISTRAL_API_KEY=your_mistral_api_key
   MISTRAL_API_URL=https://api.mistral.ai/v1/chat/completions
   MISTRAL_MODEL_NAME=mistral-medium
   ```

## Usage

### Running the full pipeline:

```bash
python main.py --all
```
This will:
- Preprocess the raw data
- Generate analytics and visualizations
- Create the vector store for RAG
- Start the API server on the default port (8000)

### Individual components:

```bash
# Preprocess data
python main.py --preprocess

# Generate analytics
python main.py --analytics

# Create vector store
python main.py --vector-store

# Run API server
python main.py --api --port 8000
```

### Accessing the API:

Once the API server is running:
- Access the Swagger UI documentation: http://localhost:8000/docs
- Access the ReDoc documentation: http://localhost:8000/redoc
- The API endpoints are available at http://localhost:8000/

### API Endpoints:

- `POST /analytics`: Get analytics data
  ```json
  {
    "metric": "revenue_trends",
    "time_period": "M"
  }
  ```

- `POST /ask`: Ask a question about the data
  ```json
  {
    "query": "Show me total revenue for July 2017"
  }
  ```

- `GET /query-history`: View history of questions asked
- `GET /health`: Check system health status

## Project Structure

```
hotel-booking-system/
├── data/
│   ├── raw/                  # Raw data
│   ├── processed/            # Processed data
│   └── vector_store/         # Vector embeddings
├── reports/
│   └── visualizations/       # Generated charts
├── src/
│   ├── analytics/            # Analytics module
│   ├── api/                  # FastAPI application
│   ├── preprocess/           # Data preprocessing
│   └── rag/                  # Retrieval Augmented Generation
├── logs/                     # Application logs
├── main.py                   # Main entry point
└── requirements.txt          # Dependencies
```

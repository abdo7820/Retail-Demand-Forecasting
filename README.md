# Retail Demand Forecasting - Ensemble ML Pipeline
![Uploading image.png…]()


An advanced Machine Learning system for retail demand forecasting using **Ensemble Techniques** (Voting, Bagging, Boosting). Built with a full Flask GUI for data preprocessing, visualization, model training, and real-time prediction.

## Features

### Machine Learning
- **Regression**: Predict exact demand quantities using ensemble regressors
- **Classification**: Categorize demand into Low/Medium/High levels
- **Time Series**: Forecast demand trends over time with lag features

### Ensemble Techniques
| Type | Models |
|------|--------|
| **Bagging** | Random Forest, Bagging (Decision Tree), Extra Trees |
| **Boosting** | Gradient Boosting, XGBoost, AdaBoost |
| **Voting** | Voting Ensemble (RF + GB + XGBoost) |

### Data Preprocessing
- Handle missing values (Mean, Median, Mode, Zero, Drop)
- Remove duplicates
- Encode categorical variables (Label Encoding, One-Hot Encoding)
- Feature scaling (Standard, MinMax, Robust)
- Automated feature engineering (date, price, competition, stock features)

### Visualization
- Interactive Plotly charts
- Column-level exploration (Histogram, Box Plot, Bar Chart)
- Scatter plots with color grouping
- Correlation heatmap
- Domain-specific: Demand over time, by category, by region
- Price vs demand analysis, discount impact study
- Model comparison charts, feature importance, residual analysis

### Flask GUI
- Modern responsive dashboard
- 6-step ML pipeline navigation
- Real-time model training with progress indicators
- Interactive prediction form

## Project Structure

```
retail-demand-forecasting/
├── app.py                      # Flask application
├── data/
│   └── retail_demand.csv       # Dataset (5000 records)
├── models/
│   ├── __init__.py
│   ├── preprocessing.py        # Data preprocessing pipeline
│   ├── ensemble_models.py      # Ensemble ML models
│   └── visualization.py        # Plotly visualizations
├── saved_models/               # Trained model files (auto-generated)
├── static/
│   └── css/
│       └── style.css           # Custom styling
├── templates/
│   ├── base.html               # Base template
│   ├── index.html              # Dashboard
│   ├── data.html               # Data exploration
│   ├── preprocessing.html      # Preprocessing controls
│   ├── visualization.html      # Visualization page
│   ├── train.html              # Model training
│   ├── results.html            # Training results
│   └── predict.html            # Prediction form
├── requirements.txt
└── README.md
```

## Installation & Setup

```bash
# Clone the repository
git clone <repo-url>
cd retail-demand-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. **Dashboard**: Overview of dataset statistics and pipeline steps
2. **Data Exploration**: Browse the dataset with pagination and statistics
3. **Preprocessing**: Clean data, encode categories, scale features, engineer new features
4. **Visualization**: Generate interactive plots for any column or domain-specific analysis
5. **Train Models**: Select task type (Regression/Classification/Time Series) and train 7 ensemble models
6. **Results**: Compare model performance with metrics, charts, and feature importance
7. **Predict**: Enter product details and get real-time demand predictions

## Dataset

The `retail_demand.csv` dataset contains 5000 records of retail product demand data with features:

| Feature | Description |
|---------|-------------|
| date | Transaction date |
| category | Product category (Electronics, Food, Clothing, Home, Sports) |
| product | Product name |
| region | Egyptian region (Cairo, Giza, Alex, Luxor, Aswan) |
| base_price | Original price |
| discount_pct | Discount percentage |
| final_price | Price after discount |
| competitor_price | Competitor's price |
| demand | Target: number of units demanded |
| stock_level | Current stock level |
| rating | Product rating (1-5) |
| day_of_week | Day of week (0-6) |
| month | Month (1-12) |
| is_weekend | Weekend flag (0/1) |

## Technologies

- **Backend**: Flask 3.0
- **ML**: scikit-learn, XGBoost
- **Data**: Pandas, NumPy
- **Visualization**: Plotly
- **Frontend**: Bootstrap 5, Font Awesome

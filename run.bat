@echo off
echo ========================================
echo   UPI Fraud Detector - Setup & Launch
echo ========================================

echo.
echo [1/4] Installing dependencies...
pip install -r requirements.txt

echo.
echo [2/4] Generating dataset...
python data\generate_data.py

echo.
echo [3/4] Running EDA (generating charts)...
python src\eda.py

echo.
echo [4/4] Training ML models...
python src\model.py

echo.
echo ========================================
echo   Launching Streamlit Dashboard...
echo   Open: http://localhost:8501
echo ========================================
streamlit run dashboard\app.py

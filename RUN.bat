@echo off
REM One-click local run: data, training, API and UI.
cd /d "%~dp0"
call .venv\Scripts\activate.bat 2>nul
if not exist models\dyslexia_cnn.pt (
  python scripts\prepare_data.py
  python -m src.train
)
start "API" cmd /k uvicorn api.main:app --host 127.0.0.1 --port 8000
timeout /t 8
streamlit run ui\app.py

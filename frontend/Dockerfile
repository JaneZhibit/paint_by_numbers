FROM python:3.10-slim

WORKDIR /app

# Копируем весь проект
COPY . .

# Устанавливаем зависимости frontend
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Expose Streamlit порт
EXPOSE 8501

# Запускаем Streamlit приложение
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

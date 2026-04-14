FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости для OpenCV и других библиотек
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем весь проект
COPY . .

# Устанавливаем зависимости frontend
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Expose Streamlit порт
EXPOSE 8501

# Запускаем Streamlit приложение
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

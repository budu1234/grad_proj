FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
COPY routes/ ./routes/
COPY models/ ./models/
COPY db.py .
COPY config.py .
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
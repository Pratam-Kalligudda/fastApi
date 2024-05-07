FROM python:3.12

WORKDIR /app

RUN apt-get update && apt-get install -y libhdf5-dev libsndfile1

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache

COPY . .

EXPOSE 8000

# CMD ["uvicorn", "fastapi:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["fastapi", "run", "main.py", "--port", "8000"]
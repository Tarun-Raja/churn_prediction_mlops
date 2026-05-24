# 1. Use an official, lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements file first (to leverage Docker layer caching)
COPY requirements.txt .

# 4. Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the necessary source code and artifacts into the container
# Notice we are NOT copying data/ or tests/, to keep the image lightweight
COPY api/ ./api/
COPY src/models/artifacts/ ./src/models/artifacts/

# 6. Expose port 8000 so the outside world can talk to the API
EXPOSE 8000

# 7. Define the command to run the FastAPI application when the container starts
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
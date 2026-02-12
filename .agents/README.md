# ML API

A FastAPI-based machine learning API for Iris flower classification and Loan approval prediction.

## Features

- **Iris Flower Classification**: Predict iris species based on sepal and petal measurements
- **Loan Approval Prediction**: Predict loan approval status based on applicant information
- **Batch Processing**: Process multiple loan applications at once
- **Health Check**: Monitor API status

## Requirements

- Docker
- Docker Compose (optional)

## Quick Start

### Using Docker

```bash
# Build the image
docker build -t ml-api .

# Run the container
docker run -p 8000:8000 ml-api
```

### Using Docker Compose

```bash
docker-compose up -d
```

The API will be available at `http://localhost:8000`

## Endpoints

### Health Check

**Endpoint**: `GET /health`

Check if the API is running.

```bash
curl -X GET http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy"
}
```

---

### Iris Prediction

**Endpoint**: `POST /iris/predict`

Predict iris species from flower measurements.

**Request Body**:
```json
{
  "data": [[5.1, 3.5, 1.4, 0.2]]
}
```

**Parameters**:
- `data`: List of 4 numeric values [sepal_length, sepal_width, petal_length, petal_width]

```bash
curl -X POST http://localhost:8000/iris/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[5.1, 3.5, 1.4, 0.2]]}'
```

**Response**:
```json
{
  "prediction": [0],
  "prediction_name": ["setosa"],
  "proba": [[0.99, 0.01, 0.0]]
}
```

---

### Loan Prediction (Single)

**Endpoint**: `POST /loan/predict`

Predict loan approval for a single application.

**Request Body**:
```json
{
  "gender": "Male",
  "married": "Yes",
  "dependents": "0",
  "education": "Graduate",
  "self_employed": "No",
  "applicant_income": 5000,
  "coapplicant_income": 2000,
  "loan_amount": 150000,
  "loan_amount_term": 360,
  "credit_history": 1.0,
  "property_area": "Urban"
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `gender` | string | Yes | "Male" or "Female" |
| `married` | string | Yes | "Yes" or "No" |
| `dependents` | string | No | "0", "1", "2", or "3+" |
| `education` | string | Yes | "Graduate" or "Not Graduate" |
| `self_employed` | string | No | "Yes" or "No" |
| `applicant_income` | number | Yes | Applicant's monthly income |
| `coapplicant_income` | number | Yes | Co-applicant's monthly income |
| `loan_amount` | number | Yes | Loan amount requested |
| `loan_amount_term` | number | Yes | Loan term in days (e.g., 360 for 30 years) |
| `credit_history` | number | Yes | 1.0 or 0.0 |
| `property_area` | string | Yes | "Urban", "Rural", or "Semiurban" |

```bash
curl -X POST http://localhost:8000/loan/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "married": "Yes",
    "dependents": "0",
    "education": "Graduate",
    "self_employed": "No",
    "applicant_income": 5000,
    "coapplicant_income": 2000,
    "loan_amount": 150000,
    "loan_amount_term": 360,
    "credit_history": 1.0,
    "property_area": "Urban"
  }'
```

**Response**:
```json
{
  "prediction": [1],
  "prediction_name": ["Approved"],
  "proba": [[0.15, 0.85]]
}
```

---

### Loan Prediction (Batch)

**Endpoint**: `POST /loan/predict/batch`

Predict loan approval for multiple applications.

**Request Body**:
```json
{
  "applications": [
    {
      "gender": "Male",
      "married": "Yes",
      "dependents": "0",
      "education": "Graduate",
      "self_employed": "No",
      "applicant_income": 5000,
      "coapplicant_income": 2000,
      "loan_amount": 150000,
      "loan_amount_term": 360,
      "credit_history": 1.0,
      "property_area": "Urban"
    },
    {
      "gender": "Female",
      "married": "No",
      "dependents": "2",
      "education": "Not Graduate",
      "self_employed": "Yes",
      "applicant_income": 3000,
      "coapplicant_income": 0,
      "loan_amount": 100000,
      "loan_amount_term": 180,
      "credit_history": 0.0,
      "property_area": "Rural"
    }
  ]
}
```

```bash
curl -X POST http://localhost:8000/loan/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "applications": [
      {
        "gender": "Male",
        "married": "Yes",
        "dependents": "0",
        "education": "Graduate",
        "self_employed": "No",
        "applicant_income": 5000,
        "coapplicant_income": 2000,
        "loan_amount": 150000,
        "loan_amount_term": 360,
        "credit_history": 1.0,
        "property_area": "Urban"
      },
      {
        "gender": "Female",
        "married": "No",
        "dependents": "2",
        "education": "Not Graduate",
        "self_employed": "Yes",
        "applicant_income": 3000,
        "coapplicant_income": 0,
        "loan_amount": 100000,
        "loan_amount_term": 180,
        "credit_history": 0.0,
        "property_area": "Rural"
      }
    ]
  }'
```

**Response**:
```json
{
  "prediction": [1, 0],
  "prediction_name": ["Approved", "Rejected"],
  "proba": [[0.15, 0.85], [0.72, 0.28]]
}
```

---

## Test Cases

### Health Check

```bash
# Test health endpoint
curl -s http://localhost:8000/health | jq .
```

Expected output:
```json
{
  "status": "healthy"
}
```

---

### Iris Prediction Test Cases

**Test Case 1: Setosa (Expected: setosa)**

```bash
curl -s -X POST http://localhost:8000/iris/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[5.1, 3.5, 1.4, 0.2]]}' | jq .
```

**Test Case 2: Versicolor (Expected: versicolor)**

```bash
curl -s -X POST http://localhost:8000/iris/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[7.0, 3.2, 4.7, 1.4]]}' | jq .
```

**Test Case 3: Virginica (Expected: virginica)**

```bash
curl -s -X POST http://localhost:8000/iris/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[6.3, 3.3, 6.0, 2.5]]}' | jq .
```

---

### Loan Prediction Test Cases

**Test Case 1: Approved Application**

```bash
curl -s -X POST http://localhost:8000/loan/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "married": "Yes",
    "dependents": "0",
    "education": "Graduate",
    "self_employed": "No",
    "applicant_income": 10000,
    "coapplicant_income": 5000,
    "loan_amount": 200000,
    "loan_amount_term": 360,
    "credit_history": 1.0,
    "property_area": "Urban"
  }' | jq .
```

**Test Case 2: Rejected Application**

```bash
curl -s -X POST http://localhost:8000/loan/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "married": "No",
    "dependents": "3+",
    "education": "Not Graduate",
    "self_employed": "Yes",
    "applicant_income": 2000,
    "coapplicant_income": 0,
    "loan_amount": 200000,
    "loan_amount_term": 360,
    "credit_history": 0.0,
    "property_area": "Rural"
  }' | jq .
```

**Test Case 3: Batch Predictions**

```bash
curl -s -X POST http://localhost:8000/loan/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "applications": [
      {
        "gender": "Male",
        "married": "Yes",
        "dependents": "0",
        "education": "Graduate",
        "self_employed": "No",
        "applicant_income": 10000,
        "coapplicant_income": 5000,
        "loan_amount": 200000,
        "loan_amount_term": 360,
        "credit_history": 1.0,
        "property_area": "Urban"
      },
      {
        "gender": "Female",
        "married": "No",
        "dependents": "2",
        "education": "Not Graduate",
        "self_employed": "Yes",
        "applicant_income": 3000,
        "coapplicant_income": 0,
        "loan_amount": 100000,
        "loan_amount_term": 180,
        "credit_history": 0.0,
        "property_area": "Rural"
      }
    ]
  }' | jq .
```

---

### Error Handling Test Cases

**Test Case 1: Invalid Credit History**

```bash
curl -s -X POST http://localhost:8000/loan/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "married": "Yes",
    "dependents": "0",
    "education": "Graduate",
    "self_employed": "No",
    "applicant_income": 5000,
    "coapplicant_income": 2000,
    "loan_amount": 150000,
    "loan_amount_term": 360,
    "credit_history": 2.0,
    "property_area": "Urban"
  }' | jq .
```

Expected error response:
```json
{
  "detail": {
    "error_code": "INVALID_INPUT",
    "message": "Input validation failed",
    "details": {
      "errors": ["Credit history must be 0.0 or 1.0"]
    }
  }
}
```

**Test Case 2: Invalid Property Area**

```bash
curl -s -X POST http://localhost:8000/loan/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "married": "Yes",
    "dependents": "0",
    "education": "Graduate",
    "self_employed": "No",
    "applicant_income": 5000,
    "coapplicant_income": 2000,
    "loan_amount": 150000,
    "loan_amount_term": 360,
    "credit_history": 1.0,
    "property_area": "InvalidArea"
  }' | jq .
```

Expected error response:
```json
{
  "detail": {
    "error_code": "INVALID_INPUT",
    "message": "Input validation failed",
    "details": {
      "errors": ["Property area must be 'Urban', 'Rural', or 'Semiurban'"]
    }
  }
}
```

---

## Development

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Building Docker Image

```bash
docker build -t ml-api:latest .
```

### Running Tests

```bash
# Run with Docker
docker run -p 8000:8000 ml-api:latest

# In another terminal, run tests
./run_tests.sh
```

---

## API Documentation

FastAPI provides automatic API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Port to run the API on |
| `MODULE_NAME` | app.main | Main application module |
| `APP_HOST` | 0.0.0.0 | Host to bind to |

---

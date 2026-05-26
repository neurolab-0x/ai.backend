# NeuroLab EEG Analysis API Documentation

## Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [Base URL](#base-url)
- [Common Response Formats](#common-response-formats)
- [Error Handling](#error-handling)
- [API Endpoints](#api-endpoints)
  - [Health & Status](#health--status)
  - [Authentication Endpoints](#authentication-endpoints)
  - [EEG Data Processing](#eeg-data-processing)
  - [Real-time Streaming](#real-time-streaming)
  - [External Training And Model Management](#external-training-and-model-management)
  - [Model Management](#model-management)
- [Data Models](#data-models)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

---

## Overview

The NeuroLab EEG Analysis API provides endpoints for processing EEG (Electroencephalogram) data, real-time mental state classification, model training, and user authentication. The API uses RESTful principles and returns JSON responses.

**Version:** 1.0.0  
**API Type:** REST  
**Content-Type:** application/json

---

## Authentication

Most endpoints require JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

### Token Expiry
- **Access Token:** 24 hours
- **Refresh Token:** 30 days

---

## Base URL

```
Production: https://model.neurolab.cc
Development: http://localhost:8000
```

---

## Common Response Formats

### Success Response
```json
{
  "status": "success",
  "data": { ... },
  "timestamp": "2025-11-13T10:30:00Z"
}
```

### Error Response
```json
{
  "status": "error",
  "detail": "Error message description",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-11-13T10:30:00Z"
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 202 | Accepted (async operation) |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 409 | Conflict |
| 422 | Validation Error |
| 429 | Too Many Requests |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

---

## API Endpoints

### Health & Status

#### GET /health
Check API health status and diagnostics.

**Authentication:** Not required

**Response:**
```json
{
  "status": "healthy",
  "diagnostics": {
    "model_loaded": true,
    "tensorflow_available": true
  }
}
```

#### GET /
Get API information and available endpoints.

**Authentication:** Not required

**Response:**
```json
{
  "name": "NeuroLab EEG Analysis API",
  "version": "1.0.0",
  "description": "API for EEG signal processing and mental state classification",
  "endpoints": {
    "health": "/health",
    "upload": "/upload",
    "analyze": "/analyze",
    "calibrate": "/calibrate",
    "recommendations": "/recommendations"
  }
}
```

---

### Authentication Endpoints

#### POST /api/auth/login
Authenticate user and receive access tokens.

**Authentication:** Not required

**Request Body:**
```json
{
  "username": "string (3-50 chars, alphanumeric)",
  "password": "string (8-100 chars)"
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "random_secure_token_string",
  "token_type": "bearer",
  "expires_in": 86400,
  "user_info": {
    "username": "user",
    "roles": ["user"]
  }
}
```

**Errors:**
- 401: Invalid username or password
- 422: Validation error

---

#### POST /api/auth/refresh
Refresh access token using refresh token.

**Authentication:** Not required

**Request Body:**
```json
{
  "refresh_token": "your_refresh_token"
}
```

**Response (200):**
```json
{
  "access_token": "new_access_token",
  "refresh_token": "new_refresh_token",
  "token_type": "bearer",
  "expires_in": 86400,
  "user_info": {
    "username": "user",
    "roles": ["user"]
  }
}
```

**Errors:**
- 401: Invalid or expired refresh token

---

#### POST /api/auth/logout
Logout and invalidate refresh tokens.

**Authentication:** Required (Bearer token)

**Headers:**
```
Authorization: Bearer <access_token>
refresh_token: <refresh_token> (optional)
```

**Response (200):**
```json
{
  "status": "success",
  "message": "Logged out successfully"
}
```

---

#### POST /api/auth/users
Create a new user (Admin only).

**Authentication:** Required (Admin role)

**Request Body:**
```json
{
  "username": "string (3-50 chars)",
  "password": "string (8-100 chars)",
  "roles": ["user"] // or ["user", "admin"]
}
```

**Response (201):**
```json
{
  "username": "newuser",
  "roles": ["user"],
  "created_at": "2025-11-13T10:30:00Z"
}
```

**Errors:**
- 403: Insufficient permissions
- 409: Username already exists

---

### EEG Data Processing

#### POST /upload
Upload and process EEG file or JSON data.

**Authentication:** Optional (recommended)

**Request (File Upload):**
```
Content-Type: multipart/form-data

file: <eeg_file.csv>
encrypt_response: false (optional, boolean)
```

**Request (JSON Data):**
```json
{
  "alpha": 0.5,
  "beta": 0.3,
  "theta": 0.2,
  "delta": 0.1,
  "gamma": 0.4,
  "subject_id": "subject_001",
  "session_id": "session_001"
}
```

**Response (200):**
```json
{
  "status": "success",
  "state_label": "Relaxed",
  "dominant_state": 0,
  "confidence": 85.5,
  "state_percentages": {
    "0": 60.0,
    "1": 25.0,
    "2": 15.0
  },
  "cognitive_metrics": {
    "attention_score": 0.75,
    "stress_level": 0.15,
    "relaxation_index": 0.85
  },
  "recommendations": [
    {
      "type": "relaxation",
      "message": "Maintain current relaxation state",
      "priority": "low"
    }
  ],
  "temporal_analysis": {
    "total_samples": 100,
    "state_transitions": 5,
    "average_state_duration": 20.0
  }
}
```

---

#### POST /analyze
Analyze EEG data and return detailed results.

**Authentication:** Optional

**Request Body:**
```json
{
  "alpha": 0.5,
  "beta": 0.3,
  "theta": 0.2,
  "delta": 0.1,
  "gamma": 0.4,
  "subject_id": "subject_001",
  "session_id": "session_001"
}
```

**Response (200):**
```json
{
  "status": "success",
  "state_label": "Focused",
  "dominant_state": 1,
  "confidence": 92.3,
  "state_percentages": {
    "0": 15.0,
    "1": 70.0,
    "2": 15.0
  },
  "cognitive_metrics": {
    "attention_score": 0.92,
    "stress_level": 0.20,
    "relaxation_index": 0.30
  },
  "recommendations": [
    {
      "type": "attention",
      "message": "Excellent focus detected",
      "priority": "info"
    }
  ]
}
```

---

### Real-time Streaming

#### POST /api/stream
Stream EEG data for real-time processing.

**Authentication:** Required (User role)

**Headers:**
```
Authorization: Bearer <access_token>
X-Client-ID: <client_identifier> (optional)
```

**Request Body:**
```json
{
  "eeg_data": [
    [0.5, 0.3, 0.2, 0.1, 0.4],
    [0.6, 0.4, 0.3, 0.2, 0.5],
    [0.4, 0.2, 0.1, 0.05, 0.3]
  ],
  "client_id": "client_001",
  "model_type": "enhanced_cnn_lstm",
  "clean_artifacts": true,
  "encrypt_response": false,
  "include_interpretability": false
}
```

**Validation Rules:**
- `eeg_data`: Max 64 channels, max 10,000 samples per channel
- Max amplitude: ±1000 μV
- No NaN or Inf values allowed

**Response (200):**
```json
{
  "predicted_states": [0, 1, 0],
  "dominant_state": 0,
  "confidence": 87.5,
  "processing_time_ms": 45.2,
  "timestamp": "2025-11-13T10:30:00Z",
  "encrypted": false,
  "interpretability": {
    "method": "lime",
    "feature_importance": {
      "alpha": 0.35,
      "beta": 0.25,
      "theta": 0.20,
      "delta": 0.10,
      "gamma": 0.10
    },
    "predicted_class": 0
  }
}
```

**Encrypted Response (when encrypt_response=true):**
```json
{
  "encrypted": true,
  "metadata": {
    "dominant_state": 0,
    "confidence": 87.5,
    "timestamp": "2025-11-13T10:30:00Z"
  },
  "data": "base64_encoded_encrypted_data"
}
```

**Errors:**
- 400: Invalid EEG data format or values
- 401: Authentication required
- 429: Rate limit exceeded

---

#### POST /api/stream/clear
Clear client stream buffer.

**Authentication:** Required (User role)

**Request Body:**
```json
{
  "client_id": "client_001"
}
```

**Response (200):**
```json
{
  "status": "success",
  "message": "Buffer cleared for client client_001"
}
```

---

### External Training And Model Management

Backend training and backend model-management endpoints have been removed.

Use:
- [`../../training_system`](</home/polo/Documents/Neurolab/AI Service/training_system>) for training runs, status tracking, and model comparison
- [`../../model_platform`](</home/polo/Documents/Neurolab/AI Service/model_platform>) for promotion, sync, calibration, and serving control
- [`../../preprocessor`](</home/polo/Documents/Neurolab/AI Service/preprocessor>) for dataset preparation and publication

This backend now documents inference-facing application APIs only.

---

#### GET /recommendations
Get recommendations based on analysis.

**Authentication:** Optional

**Query Parameters:**
- `session_id`: Session ID (required)
- `subject_id`: Subject ID (required)

**Response (200):**
```json
{
  "session_id": "session_001",
  "subject_id": "subject_001",
  "recommendations": [
    {
      "type": "stress_management",
      "severity": "medium",
      "message": "Consider stress reduction techniques",
      "confidence": 0.85
    },
    {
      "type": "attention_improvement",
      "severity": "low",
      "message": "Attention levels are good",
      "confidence": 0.90
    }
  ]
}
```

---

## Data Models

### EEGFeatures
```typescript
{
  alpha: number,    // Alpha band power (8-13 Hz)
  beta: number,     // Beta band power (13-30 Hz)
  theta: number,    // Theta band power (4-8 Hz)
  delta: number,    // Delta band power (0.5-4 Hz)
  gamma: number     // Gamma band power (30-45 Hz)
}
```

### EEGDataPoint
```typescript
{
  timestamp: string (ISO 8601),
  features: EEGFeatures,
  state: number (0-2),
  confidence: number (0-100),
  metadata: {
    signal_quality: number (0-1),
    device: string,
    session_id: string,
    subject_id: string
  }
}
```

### Mental States
```typescript
enum MentalState {
  RELAXED = 0,      // Relaxation/calm state
  FOCUSED = 1,      // Attention/focus state
  STRESSED = 2      // Stress/anxiety state
}
```

### TrainingConfig
```typescript
{
  model_type: "original" | "enhanced_cnn_lstm" | "resnet_lstm" | "transformer",
  epochs: number (1-200),
  batch_size: number (1-256),
  learning_rate: number (0-1),
  dropout_rate: number (0-0.9),
  use_separable: boolean,
  use_relative_pos: boolean,
  l1_reg: number (>=0),
  l2_reg: number (>=0),
  subject_id?: string,
  session_id?: string
}
```

---

## Rate Limiting

**Default Limits:**
- 60 requests per minute per client
- Identified by IP address or X-Client-ID header

**Rate Limit Headers:**
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1699876800
```

**Rate Limit Exceeded Response (429):**
```json
{
  "status": "error",
  "detail": "Too many requests. Please try again later.",
  "retry_after": 30
}
```

---

## Examples

### Example 1: Complete Authentication Flow

```python
import requests

BASE_URL = "https://model.neurolab.cc"

# 1. Login
login_response = requests.post(
    f"{BASE_URL}/api/auth/login",
    json={
        "username": "user",
        "password": "secure_password"
    }
)
tokens = login_response.json()
access_token = tokens["access_token"]

# 2. Use authenticated endpoint
headers = {"Authorization": f"Bearer {access_token}"}
stream_response = requests.post(
    f"{BASE_URL}/api/stream",
    headers=headers,
    json={
        "eeg_data": [[0.5, 0.3, 0.2, 0.1, 0.4]],
        "client_id": "python_client"
    }
)
print(stream_response.json())

# 3. Logout
requests.post(
    f"{BASE_URL}/api/auth/logout",
    headers=headers
)
```

### Example 2: Real-time Streaming

```python
import requests
import time

BASE_URL = "https://model.neurolab.cc"
headers = {"Authorization": f"Bearer {access_token}"}

# Stream multiple data points
for i in range(10):
    eeg_data = generate_eeg_sample()  # Your function
    
    response = requests.post(
        f"{BASE_URL}/api/stream",
        headers=headers,
        json={
            "eeg_data": [eeg_data],
            "client_id": "streaming_client",
            "include_interpretability": True
        }
    )
    
    result = response.json()
    print(f"State: {result['dominant_state']}, "
          f"Confidence: {result['confidence']}%")
    
    time.sleep(1)  # 1 Hz sampling
```

### Example 3: File Upload and Analysis

```python
import requests

BASE_URL = "https://model.neurolab.cc"

# Upload CSV file
with open("eeg_data.csv", "rb") as f:
    files = {"file": f}
    response = requests.post(
        f"{BASE_URL}/upload",
        files=files
    )

result = response.json()
print(f"Analysis complete:")
print(f"  State: {result['state_label']}")
print(f"  Confidence: {result['confidence']}%")
print(f"  Recommendations: {result['recommendations']}")
```

### Example 4: Model Training (Admin)

```python
import requests
import time

BASE_URL = "https://model.neurolab.cc"
headers = {"Authorization": f"Bearer {admin_token}"}

# Start training
train_response = requests.post(
    f"{BASE_URL}/api/train",
    headers=headers,
    json={
        "X_train": training_features,
        "y_train": training_labels,
        "X_test": test_features,
        "y_test": test_labels,
        "config": {
            "model_type": "enhanced_cnn_lstm",
            "epochs": 50,
            "batch_size": 32
        }
    }
)

job_id = train_response.json()["job_id"]

# Poll for status
while True:
    status_response = requests.get(
        f"{BASE_URL}/api/train/status/{job_id}",
        headers=headers
    )
    status = status_response.json()
    
    print(f"Progress: {status['progress']*100:.1f}% - {status['message']}")
    
    if status["status"] in ["completed", "failed"]:
        break
    
    time.sleep(5)

# Get final metrics
if status["status"] == "completed":
    print(f"Training completed!")
    print(f"Accuracy: {status['metrics']['final_val_accuracy']:.2%}")
```

---

## Security Best Practices

1. **Always use HTTPS** in production
2. **Store tokens securely** (never in localStorage for web apps)
3. **Implement token refresh** before expiry
4. **Validate all inputs** on client side
5. **Handle rate limits** gracefully with exponential backoff
6. **Use encryption** for sensitive EEG data
7. **Implement proper error handling**

---

## Support

For API support and questions:
- **Email:** nelsonprox92@gmail.com
- **Documentation:** https://neurolab.cc/docs
- **GitHub:** https://github.com/asimwe1/eeg-ds

---

**Last Updated:** November 13, 2025  
**API Version:** 1.0.0

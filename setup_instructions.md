# Cloud SQL Utilization Monitor Setup

## Prerequisites
1. Python 3.8 or higher
2. Google Cloud SDK installed
3. Appropriate GCP permissions

## Installation Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Authentication Setup
Choose one of these methods:

#### Option A: Default Application Credentials (Recommended)
```bash
gcloud auth application-default login
```

#### Option B: Service Account (if you have a service account JSON)
- Download your service account JSON file
- Set environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
```

### 3. Required GCP Permissions
Your account/service account needs these IAM roles:
- `Cloud SQL Viewer` (to list instances)
- `Monitoring Viewer` (to read metrics)
- `Browser` (to list projects)

## Usage

### Step 1: Collect Data
```bash
python cloudsql_utilization_monitor.py
```
This will:
- Auto-discover all projects you have access to
- Find all Cloud SQL instances across those projects
- Collect 9 months of CPU utilization data
- Generate `cloudsql_utilization_results.xlsx`

### Step 2: Launch Interactive Dashboard
```bash
streamlit run dashboard.py
```
This will open a web browser with your interactive dashboard.

## Dashboard Features
- 📊 Overview metrics and KPIs
- 🎯 Interactive scatter plot (CPU vs Instances)
- 📈 Project-wise summary charts
- ⏱️ Time series analysis for selected instances
- 📋 Filterable data table
- 📥 Export functionality

## Troubleshooting
- If you get permission errors, check your IAM roles
- If no instances are found, verify you have Cloud SQL instances in your projects
- For large datasets, the script may take 10-30 minutes to complete 
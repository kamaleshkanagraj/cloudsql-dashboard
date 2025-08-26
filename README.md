# 🔍 Cloud SQL Resource Utilization Dashboard

Advanced Cloud SQL instance optimization analysis with **Zero-Spike Algorithm** for performance-safe rightsizing.

## 🎯 **Key Features**

- **Zero-Spike Analysis**: Prevents performance issues by detecting critical CPU spikes (>90%)
- **Dual-Threshold Monitoring**: Tracks both critical (>90%) and moderate (>50%) usage spikes
- **Real-Time Data**: 30-day analysis with 5-minute granularity (8,640+ data points per instance)
- **Interactive Dashboard**: Professional Streamlit interface with drill-down capabilities
- **Multi-Project Support**: Analyzes all accessible GCP projects automatically

## 🚀 **Quick Start**

### 1. **Setup Environment**
```bash
# Create virtual environment
python -m venv cloudsql_monitor_env
source cloudsql_monitor_env/bin/activate  # Linux/Mac
# or
cloudsql_monitor_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. **Configure GCP Authentication**
```bash
# Install gcloud CLI and authenticate
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 3. **Run Analysis**
```bash
# Collect data and generate analysis
python cloudsql_utilization_monitor.py

# Launch interactive dashboard
streamlit run dashboard.py
```

## 📊 **Algorithm Logic**

### **Zero-Spike Approach**
```
✅ SAFE TO OPTIMIZE = (Avg CPU < 50%) AND (Zero spikes > 90%)
❌ KEEP CURRENT SIZE = Any spike > 90% detected  
ℹ️ WELL UTILIZED = Average CPU >= 50%
```

### **Why This Works**
- **Performance Safety**: Even one spike >90% indicates the application needs current capacity
- **Conservative Thresholds**: 50% average (not 40%) prevents performance risks
- **Real Usage Patterns**: Based on actual monitoring data, not estimates

## 🏗️ **Architecture**

```
cloudsql_utilization_monitor.py  → Data Collection & Analysis
├── Google Cloud Monitoring API   → CPU/Memory/Disk metrics
├── Google Cloud SQL Admin API    → Instance specifications  
├── Statistical Analysis          → Averages, P95, Max, Spike counting
└── Excel Export                  → Results for dashboard

dashboard.py                      → Interactive Visualization
├── Executive Overview            → High-level metrics & trends
├── Project Analysis              → Resource efficiency by project
├── Instance Deep Dive            → Individual instance analysis
└── Professional Charts           → Plotly-based visualizations
```

## 🌐 **Deployment Options**

### **Option 1: Streamlit Community Cloud (Recommended)**
1. Push code to GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io/)
3. Connect GitHub and deploy
4. Get public URL: `https://yourapp.streamlit.app/`

### **Option 2: Local Network Sharing**
```bash
# Run on local network
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
# Access via: http://YOUR_IP:8501
```

### **Option 3: Cloud Platforms**
- **Heroku**: `git push heroku main`
- **Railway**: `railway up`
- **Google Cloud Run**: Container deployment

## 📋 **Required GCP Permissions**

```yaml
# Service Account Roles:
- Cloud SQL Viewer
- Monitoring Viewer  
- Browser (for project listing)

# APIs to Enable:
- Cloud SQL Admin API
- Cloud Monitoring API
- Cloud Resource Manager API
```

## 📈 **Sample Output**

### **Executive Metrics**
- **47 instances** analyzed across **12 projects**
- **23% optimization opportunity** (11 instances safe to optimize)
- **67.2% CPU waste** detected across portfolio
- **$2,400/month potential savings**

### **Instance Analysis Example**
```
Instance: production-db-01
├── Specifications: 16 vCPU, 104 GB RAM, 1.5 TB SSD
├── 30-Day Averages: 12.3% CPU, 28.1% Memory, 45.2% Storage  
├── Performance Spikes: 3 critical (>90%), 247 moderate (>50%)
└── Decision: KEEP CURRENT SIZE ❌ (Performance risk detected)
```

## 🔧 **Configuration**

### **Analysis Parameters** (`cloudsql_utilization_monitor.py`)
```python
ANALYSIS_MONTHS = 1          # 30 days analysis period
CPU_AVG_THRESHOLD = 50       # Average CPU threshold  
SPIKE_THRESHOLD = 90         # Critical spike threshold
MAX_SPIKES_ALLOWED = 0       # Zero-spike policy
```

### **Dashboard Settings** (`dashboard.py`)
```python
instances_per_page = 10      # Pagination size
color_themes = "professional" # Chart styling
export_formats = ["CSV", "Excel"] # Export options
```

## 🛠️ **Troubleshooting**

### **Common Issues**
```bash
# Permission errors
gcloud auth application-default login
gcloud projects list  # Verify access

# Missing data file
python cloudsql_utilization_monitor.py  # Generate data first

# Module import errors  
pip install -r requirements.txt  # Reinstall dependencies
```

### **Performance Optimization**
- Use `--server.runOnSave false` for large datasets
- Enable caching with `@st.cache_data` for repeated queries
- Paginate results for 100+ instances

## 📊 **Business Value**

### **Risk Mitigation**
- **Zero performance degradation** from rightsizing
- **Application-first approach** prevents outages
- **Data-driven decisions** based on real usage patterns

### **Cost Optimization**  
- **Identify over-provisioned instances** safely
- **Quantify potential savings** with confidence levels
- **Prioritize optimization efforts** by impact

### **Operational Excellence**
- **Automated analysis** reduces manual effort
- **Professional reporting** for stakeholder communication  
- **Continuous monitoring** capabilities

## 📞 **Support**

For questions, issues, or feature requests:
- Create GitHub issues for bugs/features
- Review dashboard logs for troubleshooting
- Check GCP quotas and permissions for API errors

---

**Built with ❤️ for Cloud Infrastructure Optimization** 
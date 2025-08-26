#!/usr/bin/env python3
"""
Cloud SQL Resource Utilization Dashboard
Main application entry point for deployment
"""

import streamlit as st
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import the dashboard
from dashboard import CloudSQLDashboard

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Cloud SQL Optimization Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize and run dashboard
    dashboard = CloudSQLDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 
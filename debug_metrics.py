#!/usr/bin/env python3
"""
Debug script to check raw Cloud SQL metric values
This will help identify if the conversion factor is correct
"""

import pandas as pd
from google.cloud import monitoring_v3
from google.auth import default
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_single_instance_metrics():
    """Debug a single instance to check raw metric values"""
    
    # Use your actual project and instance
    project_id = "balsambrands-20022025"  # Replace with your actual project
    instance_id = "balsam-dev"            # Replace with your actual instance
    
    try:
        credentials, _ = default()
        monitoring_client = monitoring_v3.MetricServiceClient(credentials=credentials)
        
        # Get last 1 hour of data for debugging
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        interval = monitoring_v3.TimeInterval({
            "end_time": {"seconds": int(end_time.timestamp())},
            "start_time": {"seconds": int(start_time.timestamp())},
        })
        
        project_name = f"projects/{project_id}"
        
        # CPU metric filter
        cpu_filter = f'resource.type="cloudsql_database" AND resource.labels.database_id="{project_id}:{instance_id}" AND metric.type="cloudsql.googleapis.com/database/cpu/utilization"'
        
        request = monitoring_v3.ListTimeSeriesRequest(
            name=project_name,
            filter=cpu_filter,
            interval=interval,
            view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            aggregation=monitoring_v3.Aggregation(
                alignment_period={"seconds": 300},  # 5-minute intervals
                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MAX,
            ),
        )
        
        results = monitoring_client.list_time_series(request=request)
        
        print(f"\n🔍 DEBUGGING METRICS FOR: {project_id}/{instance_id}")
        print(f"=" * 60)
        
        raw_values = []
        converted_values = []
        
        for result in results:
            print(f"Metric: {result.metric.type}")
            print(f"Resource: {result.resource.type}")
            
            for i, point in enumerate(result.points):
                if i >= 10:  # Show only first 10 points
                    break
                    
                raw_value = point.value.double_value
                
                # Test different conversion approaches
                converted_100x = raw_value * 100      # Current approach
                converted_as_is = raw_value           # No conversion
                
                raw_values.append(raw_value)
                converted_values.append(converted_100x)
                
                timestamp = datetime.fromtimestamp(point.interval.end_time.timestamp())
                
                print(f"Point {i+1}:")
                print(f"  Timestamp: {timestamp}")
                print(f"  Raw Value: {raw_value:.6f}")
                print(f"  × 100: {converted_100x:.2f}%")
                print(f"  As-is: {converted_as_is:.2f}%")
                print()
        
        if raw_values:
            import numpy as np
            
            print(f"📊 SUMMARY STATISTICS:")
            print(f"Raw values range: {min(raw_values):.6f} to {max(raw_values):.6f}")
            print(f"Raw average: {np.mean(raw_values):.6f}")
            print(f"Converted (×100) average: {np.mean(converted_values):.2f}%")
            print(f"As-is average: {np.mean(raw_values):.2f}%")
            
            print(f"\n🤔 ANALYSIS:")
            if max(raw_values) > 1.0:
                print("❌ Raw values > 1.0 → Already in percentage format!")
                print("✅ Correct conversion: Use raw values as-is (no ×100)")
            else:
                print("✅ Raw values ≤ 1.0 → In decimal format (0-1 range)")
                print("✅ Correct conversion: Multiply by 100")
                
        else:
            print("❌ No data points found. Check project/instance names.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you're authenticated with: gcloud auth application-default login")

if __name__ == "__main__":
    debug_single_instance_metrics() 
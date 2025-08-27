#!/usr/bin/env python3
"""
Debug script to match exact Google Console time period
Test different aggregation methods to match Console results
"""

from google.cloud import monitoring_v3
from google.auth import default
from datetime import datetime, timedelta
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_exact_console_period():
    """Test different aggregation methods to match Google Console"""
    
    project_id = "balsambrands-20022025"
    instance_id = "balsam-dev"
    
    try:
        credentials, _ = default()
        monitoring_client = monitoring_v3.MetricServiceClient(credentials=credentials)
        
        # Test different time periods
        end_time = datetime.now()
        
        # Exactly 30 days (720 hours)
        start_time_30d = end_time - timedelta(days=30)
        
        print(f"\n🔍 TESTING DIFFERENT AGGREGATION METHODS")
        print(f"=" * 60)
        print(f"Instance: {project_id}/{instance_id}")
        print(f"Period: {start_time_30d.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
        
        project_name = f"projects/{project_id}"
        cpu_filter = f'resource.type="cloudsql_database" AND resource.labels.database_id="{project_id}:{instance_id}" AND metric.type="cloudsql.googleapis.com/database/cpu/utilization"'
        
        # Test different alignment periods
        test_configs = [
            {"period": 300, "aligner": monitoring_v3.Aggregation.Aligner.ALIGN_MEAN, "name": "5-min MEAN"},
            {"period": 300, "aligner": monitoring_v3.Aggregation.Aligner.ALIGN_MAX, "name": "5-min MAX"},
            {"period": 3600, "aligner": monitoring_v3.Aggregation.Aligner.ALIGN_MEAN, "name": "1-hour MEAN"},
            {"period": 3600, "aligner": monitoring_v3.Aggregation.Aligner.ALIGN_MAX, "name": "1-hour MAX"},
            {"period": 86400, "aligner": monitoring_v3.Aggregation.Aligner.ALIGN_MEAN, "name": "1-day MEAN"},
        ]
        
        for config in test_configs:
            try:
                interval = monitoring_v3.TimeInterval({
                    "end_time": {"seconds": int(end_time.timestamp())},
                    "start_time": {"seconds": int(start_time_30d.timestamp())},
                })
                
                request = monitoring_v3.ListTimeSeriesRequest(
                    name=project_name,
                    filter=cpu_filter,
                    interval=interval,
                    view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                    aggregation=monitoring_v3.Aggregation(
                        alignment_period={"seconds": config["period"]},
                        per_series_aligner=config["aligner"],
                    ),
                )
                
                results = monitoring_client.list_time_series(request=request)
                
                values = []
                for result in results:
                    for point in result.points:
                        raw_value = point.value.double_value
                        converted_value = raw_value * 100
                        values.append(converted_value)
                
                if values:
                    avg_val = np.mean(values)
                    max_val = np.max(values)
                    min_val = np.min(values)
                    p50_val = np.percentile(values, 50)
                    p95_val = np.percentile(values, 95)
                    p99_val = np.percentile(values, 99)
                    
                    print(f"\n📊 {config['name']} ({len(values)} points):")
                    print(f"  Average: {avg_val:.2f}%")
                    print(f"  P50 (Median): {p50_val:.2f}%")
                    print(f"  P95: {p95_val:.2f}%")
                    print(f"  P99: {p99_val:.2f}%")
                    print(f"  Max: {max_val:.2f}%")
                    print(f"  Min: {min_val:.2f}%")
                    
                    # Compare with Console values
                    console_p50 = 3.15
                    console_p99 = 21.69
                    
                    p50_diff = abs(p50_val - console_p50)
                    p99_diff = abs(p99_val - console_p99)
                    
                    if p50_diff < 1.0 and p99_diff < 3.0:
                        print(f"  ✅ CLOSE MATCH to Console! (P50 diff: {p50_diff:.2f}, P99 diff: {p99_diff:.2f})")
                    else:
                        print(f"  ❌ Different from Console (P50 diff: {p50_diff:.2f}, P99 diff: {p99_diff:.2f})")
                        
                else:
                    print(f"\n❌ {config['name']}: No data")
                    
            except Exception as e:
                print(f"\n❌ {config['name']}: Error - {e}")
        
        print(f"\n🎯 CONSOLE REFERENCE VALUES:")
        print(f"  P50: 3.15%")
        print(f"  P99: 21.69%")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_exact_console_period() 
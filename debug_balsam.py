#!/usr/bin/env python3
"""
Debug script specifically for balsam-dev instance
Trace the exact calculation path that leads to 5.6% vs expected ~15-20%
"""

from cloudsql_utilization_monitor import CloudSQLMonitor
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_balsam_calculation():
    """Debug the exact calculation for balsam-dev"""
    
    print(f"\n🔍 DEBUGGING BALSAM-DEV CALCULATION")
    print(f"=" * 60)
    
    monitor = CloudSQLMonitor()
    
    project_id = "balsambrands-20022025"
    instance_id = "balsam-dev"
    
    # Get 1 month of data (same as production)
    print(f"Getting 1 month of comprehensive metrics...")
    metrics_data = monitor.get_comprehensive_metrics(project_id, instance_id, months=1)
    
    # Check what we got
    cpu_data = metrics_data.get('cpu', [])
    print(f"CPU data points retrieved: {len(cpu_data)}")
    
    if cpu_data:
        # Extract values and show sample
        values = [d['value'] for d in cpu_data]
        raw_values = [d['raw_value'] for d in cpu_data]
        
        print(f"\n📊 RAW DATA SAMPLE (first 10 points):")
        for i in range(min(10, len(cpu_data))):
            point = cpu_data[i]
            print(f"  Point {i+1}: Raw={point['raw_value']:.6f}, Converted={point['value']:.2f}%")
        
        print(f"\n📈 STATISTICAL ANALYSIS:")
        import numpy as np
        avg_pct = np.mean(values)
        max_pct = np.max(values) 
        min_pct = np.min(values)
        p95_pct = np.percentile(values, 95)
        
        print(f"  Total data points: {len(values)}")
        print(f"  Average: {avg_pct:.2f}%")
        print(f"  Maximum: {max_pct:.2f}%") 
        print(f"  Minimum: {min_pct:.2f}%")
        print(f"  95th Percentile: {p95_pct:.2f}%")
        
        print(f"\n🔍 VALUE RANGE ANALYSIS:")
        print(f"  Raw values range: {min(raw_values):.6f} to {max(raw_values):.6f}")
        print(f"  Converted range: {min(values):.2f}% to {max(values):.2f}%")
        
        # Now test the full calculation pipeline
        print(f"\n🧮 FULL CALCULATION PIPELINE:")
        
        # Get instance specs (dummy for this debug)
        vcpu_count = 16
        memory_gb = 102.4
        disk_size_gb = 1507
        
        utilization_stats = monitor.calculate_resource_utilization(
            metrics_data, vcpu_count, memory_gb, disk_size_gb
        )
        
        cpu_stats = utilization_stats.get('cpu', {})
        print(f"\n🎯 FINAL RESULTS:")
        print(f"  avg_cpu_utilization: {cpu_stats.get('avg_percentage', 0):.2f}%")
        print(f"  max_cpu_utilization: {cpu_stats.get('max_percentage', 0):.2f}%")
        print(f"  p95_cpu_utilization: {cpu_stats.get('p95_percentage', 0):.2f}%")
        print(f"  data_points: {cpu_stats.get('data_points', 0)}")
        
        print(f"\n🤔 EXPECTED vs ACTUAL:")
        print(f"  Google Console P99: ~21.69%")
        print(f"  Google Console P50: ~3.15%")
        print(f"  Our Average: {cpu_stats.get('avg_percentage', 0):.2f}%")
        print(f"  Our Maximum: {cpu_stats.get('max_percentage', 0):.2f}%")
        
        if abs(cpu_stats.get('avg_percentage', 0) - 15.6) > 5:
            print(f"  ❌ CALCULATION ERROR DETECTED!")
            print(f"     Expected: ~15.6% (from 1-hour debug)")
            print(f"     Got: {cpu_stats.get('avg_percentage', 0):.2f}%")
        else:
            print(f"  ✅ Calculation appears correct")
            
    else:
        print("❌ No CPU data retrieved!")

if __name__ == "__main__":
    debug_balsam_calculation() 
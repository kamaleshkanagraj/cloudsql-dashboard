#!/usr/bin/env python3
"""
Cloud SQL Utilization Calculation Example
==========================================

This script demonstrates the exact calculation methodology used to determine
if a Cloud SQL instance is underutilized.
"""

import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(days=90):
    """Generate realistic sample CPU utilization data"""
    
    # Simulate 3 months of 5-minute interval data
    total_points = days * 24 * 12  # 90 days × 24 hours × 12 (5-min intervals)
    
    print(f"Generating {total_points:,} data points for {days} days...")
    
    # Create realistic usage pattern
    cpu_data = []
    base_time = datetime.now() - timedelta(days=days)
    
    for i in range(total_points):
        # Time progression
        timestamp = base_time + timedelta(minutes=i * 5)
        
        # Simulate realistic usage patterns
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        # Base usage varies by time of day and week
        if 9 <= hour <= 17 and day_of_week < 5:  # Business hours, weekdays
            base_usage = random.uniform(25, 45)   # Higher during business hours
        elif 18 <= hour <= 22:  # Evening
            base_usage = random.uniform(15, 35)   # Moderate evening usage
        else:  # Night/weekend
            base_usage = random.uniform(5, 25)    # Lower at night/weekends
        
        # Add occasional spikes (batch jobs, reports, etc.)
        if random.random() < 0.05:  # 5% chance of spike
            spike = random.uniform(60, 90)
            usage = min(spike, 95)  # Cap at 95%
        else:
            # Add some random variation
            variation = random.uniform(-5, 5)
            usage = max(0, min(95, base_usage + variation))
        
        cpu_data.append({
            'timestamp': timestamp.timestamp(),
            'value': usage,
            'datetime': timestamp
        })
    
    return cpu_data

def analyze_instance_utilization(cpu_data, vcpu_count=8, threshold=40):
    """Analyze instance utilization using our algorithm"""
    
    print(f"\n{'='*60}")
    print(f"INSTANCE UTILIZATION ANALYSIS")
    print(f"{'='*60}")
    
    # Extract CPU percentages
    cpu_values = [d['value'] for d in cpu_data]
    
    print(f"Instance Specifications:")
    print(f"  • Allocated vCPUs: {vcpu_count}")
    print(f"  • Analysis Period: {len(cpu_data):,} data points")
    print(f"  • Time Span: {len(cpu_data) * 5 / (60 * 24):.0f} days")
    print(f"  • Data Granularity: 5-minute intervals")
    
    # Step 1: Calculate Statistical Metrics
    print(f"\nStep 1: Statistical Analysis")
    print(f"-" * 30)
    
    avg_cpu = np.mean(cpu_values)
    max_cpu = np.max(cpu_values)
    min_cpu = np.min(cpu_values)
    p50_cpu = np.percentile(cpu_values, 50)  # Median
    p95_cpu = np.percentile(cpu_values, 95)  # 95th percentile
    p99_cpu = np.percentile(cpu_values, 99)  # 99th percentile
    
    print(f"  Average CPU: {avg_cpu:.2f}%")
    print(f"  Median CPU: {p50_cpu:.2f}%")
    print(f"  Maximum CPU: {max_cpu:.2f}%")
    print(f"  Minimum CPU: {min_cpu:.2f}%")
    print(f"  95th Percentile: {p95_cpu:.2f}% (regular peak usage)")
    print(f"  99th Percentile: {p99_cpu:.2f}% (extreme peaks)")
    
    # Step 2: Calculate Actual Resource Usage
    print(f"\nStep 2: Actual Resource Usage Calculation")
    print(f"-" * 45)
    
    avg_cpu_actual = (avg_cpu / 100) * vcpu_count
    p95_cpu_actual = (p95_cpu / 100) * vcpu_count
    max_cpu_actual = (max_cpu / 100) * vcpu_count
    
    print(f"  Average Usage: {avg_cpu_actual:.2f} vCPUs ({avg_cpu:.1f}%)")
    print(f"  Peak Usage (P95): {p95_cpu_actual:.2f} vCPUs ({p95_cpu:.1f}%)")
    print(f"  Maximum Usage: {max_cpu_actual:.2f} vCPUs ({max_cpu:.1f}%)")
    print(f"  Unused Capacity (Avg): {vcpu_count - avg_cpu_actual:.2f} vCPUs ({100 - avg_cpu:.1f}%)")
    
    # Step 3: Apply Underutilization Logic
    print(f"\nStep 3: Underutilization Assessment")
    print(f"-" * 38)
    
    peak_threshold = threshold + 20
    
    print(f"  Thresholds:")
    print(f"    • Average CPU threshold: {threshold}%")
    print(f"    • Peak CPU threshold: {peak_threshold}%")
    
    avg_below_threshold = avg_cpu < threshold
    peak_below_threshold = p95_cpu < peak_threshold
    
    print(f"\n  Assessment:")
    print(f"    • Average CPU ({avg_cpu:.1f}%) < {threshold}%: {'✓' if avg_below_threshold else '✗'}")
    print(f"    • Peak CPU ({p95_cpu:.1f}%) < {peak_threshold}%: {'✓' if peak_below_threshold else '✗'}")
    
    # Final Decision
    is_underutilized = avg_below_threshold and peak_below_threshold
    
    print(f"\n  ALGORITHM LOGIC:")
    print(f"    is_underutilized = (avg < {threshold}%) AND (p95 < {peak_threshold}%)")
    print(f"    is_underutilized = {avg_below_threshold} AND {peak_below_threshold}")
    print(f"    is_underutilized = {is_underutilized}")
    
    # Result and Reasoning
    print(f"\n{'='*60}")
    if is_underutilized:
        print(f"🔴 RESULT: UNDERUTILIZED")
        print(f"{'='*60}")
        print(f"Reasoning:")
        print(f"  • Average usage ({avg_cpu:.1f}%) is consistently below {threshold}%")
        print(f"  • Peak usage ({p95_cpu:.1f}%) is also below {peak_threshold}%")
        print(f"  • Instance rarely experiences high usage spikes")
        print(f"  • Using only {avg_cpu_actual:.1f} out of {vcpu_count} vCPUs on average")
        
        # Rightsizing recommendation
        recommended_vcpu = max(2, int(np.ceil(p95_cpu_actual * 1.2)))  # 20% buffer above P95
        potential_savings = vcpu_count - recommended_vcpu
        savings_pct = (potential_savings / vcpu_count) * 100
        
        print(f"\n💡 OPTIMIZATION RECOMMENDATION:")
        print(f"  • Current allocation: {vcpu_count} vCPUs")
        print(f"  • Recommended allocation: {recommended_vcpu} vCPUs")
        print(f"  • Potential savings: {potential_savings} vCPUs ({savings_pct:.1f}%)")
        
    else:
        print(f"🟢 RESULT: WELL UTILIZED")
        print(f"{'='*60}")
        if avg_below_threshold and not peak_below_threshold:
            print(f"Reasoning:")
            print(f"  • Average usage ({avg_cpu:.1f}%) is below {threshold}%")
            print(f"  • BUT peak usage ({p95_cpu:.1f}%) exceeds {peak_threshold}%")
            print(f"  • Instance has regular high usage spikes")
            print(f"  • Rightsizing could cause performance issues during peaks")
            print(f"  • Current allocation is appropriate for workload pattern")
        else:
            print(f"Reasoning:")
            print(f"  • Average usage ({avg_cpu:.1f}%) meets or exceeds {threshold}%")
            print(f"  • Instance is well-utilized for its workload")
    
    # Usage Pattern Analysis
    print(f"\n📊 USAGE PATTERN ANALYSIS:")
    
    # Calculate time spent in different usage ranges
    low_usage = len([x for x in cpu_values if x < 20]) / len(cpu_values) * 100
    medium_usage = len([x for x in cpu_values if 20 <= x < 60]) / len(cpu_values) * 100
    high_usage = len([x for x in cpu_values if x >= 60]) / len(cpu_values) * 100
    
    print(f"  • Low usage (<20%): {low_usage:.1f}% of time")
    print(f"  • Medium usage (20-60%): {medium_usage:.1f}% of time")
    print(f"  • High usage (≥60%): {high_usage:.1f}% of time")
    
    return {
        'is_underutilized': is_underutilized,
        'avg_cpu': avg_cpu,
        'p95_cpu': p95_cpu,
        'max_cpu': max_cpu,
        'avg_cpu_actual': avg_cpu_actual,
        'p95_cpu_actual': p95_cpu_actual,
        'data_points': len(cpu_data)
    }

def main():
    """Run the calculation example"""
    
    print("🔍 CLOUD SQL UTILIZATION CALCULATION EXAMPLE")
    print("=" * 80)
    print("This example demonstrates how we determine if an instance is underutilized.")
    print("We'll use simulated data that mimics real Cloud SQL usage patterns.")
    
    # Generate sample data
    cpu_data = generate_sample_data(days=90)  # 3 months
    
    # Analyze the instance
    result = analyze_instance_utilization(cpu_data, vcpu_count=8, threshold=40)
    
    # Show some sample data points
    print(f"\n📋 SAMPLE DATA POINTS (first 10 of {len(cpu_data):,}):")
    print(f"{'Timestamp':<20} {'CPU %':<8} {'DateTime'}")
    print("-" * 50)
    for i in range(10):
        data_point = cpu_data[i]
        dt_str = data_point['datetime'].strftime('%Y-%m-%d %H:%M')
        print(f"{data_point['timestamp']:<20.0f} {data_point['value']:<8.1f} {dt_str}")
    
    print(f"\n🎯 KEY TAKEAWAYS:")
    print(f"  1. We collect data every 5 minutes using MAX aggregation")
    print(f"  2. Statistical analysis considers averages AND peak usage")
    print(f"  3. Underutilization requires BOTH avg<40% AND peak<60%")
    print(f"  4. This prevents flagging instances with regular usage spikes")
    print(f"  5. Actual resource usage helps with rightsizing decisions")
    
    print(f"\n📊 To see this analysis on your real data:")
    print(f"     python cloudsql_utilization_monitor.py")

if __name__ == "__main__":
    main() 
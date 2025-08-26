#!/usr/bin/env python3
"""
Statistical Logic Behind Cloud SQL Utilization Thresholds
=========================================================

This script explains why we use 40%, 60%, 95th percentile, and 100% thresholds
for determining Cloud SQL instance underutilization.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

def generate_realistic_cpu_data(pattern_type="underutilized"):
    """Generate realistic CPU usage patterns for different instance types"""
    
    # 3 months = 90 days = 25,920 data points (5-minute intervals)
    hours = 90 * 24  # 2160 hours
    data_points = hours * 12  # 12 data points per hour (5-minute intervals)
    
    if pattern_type == "underutilized":
        # Low baseline with occasional small spikes
        baseline = np.random.normal(15, 5, data_points)  # 15% average
        spikes = np.random.exponential(2, data_points)   # Occasional spikes
        cpu_data = np.clip(baseline + spikes, 0, 45)     # Max 45%
        
    elif pattern_type == "borderline":
        # Very low average but significant peaks (like your carters-dev-new)
        baseline = np.random.normal(12, 4, data_points)  # 12% baseline
        
        # Add regular business hour spikes (8 AM - 6 PM)
        for i in range(data_points):
            hour_of_day = (i // 12) % 24
            if 8 <= hour_of_day <= 18:  # Business hours
                if np.random.random() < 0.1:  # 10% chance of spike
                    baseline[i] += np.random.uniform(50, 85)  # Big spikes
        
        cpu_data = np.clip(baseline, 0, 100)
        
    elif pattern_type == "well_utilized":
        # Consistent moderate usage with regular peaks
        baseline = np.random.normal(55, 15, data_points)  # 55% average
        cpu_data = np.clip(baseline, 20, 90)  # 20-90% range
        
    elif pattern_type == "high_utilization":
        # High consistent usage
        baseline = np.random.normal(75, 10, data_points)  # 75% average
        cpu_data = np.clip(baseline, 50, 100)  # 50-100% range
    
    return cpu_data

def analyze_cpu_pattern(cpu_data, instance_name, vcpu_count=24):
    """Analyze CPU pattern and apply our threshold logic"""
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {instance_name}")
    print(f"{'='*60}")
    
    # Calculate statistics
    avg_cpu = np.mean(cpu_data)
    min_cpu = np.min(cpu_data)
    max_cpu = np.max(cpu_data)
    p50_cpu = np.percentile(cpu_data, 50)  # Median
    p95_cpu = np.percentile(cpu_data, 95)  # 95th percentile
    p99_cpu = np.percentile(cpu_data, 99)  # 99th percentile
    
    # Calculate actual vCPU usage
    avg_vcpu_used = (avg_cpu / 100) * vcpu_count
    p95_vcpu_used = (p95_cpu / 100) * vcpu_count
    max_vcpu_used = (max_cpu / 100) * vcpu_count
    
    print(f"📊 STATISTICAL BREAKDOWN:")
    print(f"  Data Points: {len(cpu_data):,}")
    print(f"  Time Period: 90 days (3 months)")
    print(f"  Instance Capacity: {vcpu_count} vCPUs")
    
    print(f"\n📈 CPU UTILIZATION STATISTICS:")
    print(f"  Average (Mean):     {avg_cpu:.1f}% ({avg_vcpu_used:.1f}/{vcpu_count} vCPU)")
    print(f"  Minimum:            {min_cpu:.1f}%")
    print(f"  Median (P50):       {p50_cpu:.1f}%")
    print(f"  95th Percentile:    {p95_cpu:.1f}% ({p95_vcpu_used:.1f}/{vcpu_count} vCPU)")
    print(f"  99th Percentile:    {p99_cpu:.1f}%")
    print(f"  Maximum:            {max_cpu:.1f}% ({max_vcpu_used:.1f}/{vcpu_count} vCPU)")
    
    # Apply our threshold logic
    print(f"\n🎯 THRESHOLD ANALYSIS:")
    print(f"  Average < 40%:      {'✓' if avg_cpu < 40 else '✗'} ({avg_cpu:.1f}%)")
    print(f"  P95 < 60%:          {'✓' if p95_cpu < 60 else '✗'} ({p95_cpu:.1f}%)")
    
    # Decision logic
    avg_below = avg_cpu < 40
    peak_below = p95_cpu < 60
    
    if avg_below and peak_below:
        decision = "🔴 UNDERUTILIZED"
        reason = "Both average and peaks are low - clear underutilization"
        recommended_vcpu = max(2, int(np.ceil(p95_vcpu_used * 1.3)))
    elif avg_below and not peak_below:
        if avg_cpu < 20 and max_cpu >= 90:
            decision = "🟡 BORDERLINE (Underutilized with caution)"
            reason = "Very low average but high peaks - careful rightsizing possible"
            recommended_vcpu = max(4, int(np.ceil(p95_vcpu_used * 1.2)))
        else:
            decision = "🟢 WELL UTILIZED"
            reason = "Low average but regular high usage spikes"
            recommended_vcpu = vcpu_count
    else:
        decision = "🟢 WELL UTILIZED"
        reason = "Average usage above threshold"
        recommended_vcpu = vcpu_count
    
    print(f"\n🏁 FINAL DECISION: {decision}")
    print(f"  Reasoning: {reason}")
    
    if recommended_vcpu < vcpu_count:
        savings = vcpu_count - recommended_vcpu
        print(f"  💡 Recommendation: Reduce to {recommended_vcpu} vCPU (save {savings} vCPU)")
        print(f"  💰 Potential Savings: ~{(savings/vcpu_count)*100:.0f}% cost reduction")
    else:
        print(f"  ✅ Recommendation: Keep current {vcpu_count} vCPU sizing")
    
    # Time-based analysis
    above_95_time = np.sum(cpu_data > p95_cpu) / len(cpu_data) * 100
    above_avg_time = np.sum(cpu_data > avg_cpu) / len(cpu_data) * 100
    
    print(f"\n⏰ TIME DISTRIBUTION:")
    print(f"  Time above average: {above_avg_time:.1f}%")
    print(f"  Time above P95:     {above_95_time:.1f}% (~{above_95_time/100*90*24:.1f} hours/month)")
    print(f"  Time at/near max:   {np.sum(cpu_data > 90)/len(cpu_data)*100:.1f}%")
    
    return {
        'avg': avg_cpu, 'p95': p95_cpu, 'max': max_cpu,
        'decision': decision, 'recommended_vcpu': recommended_vcpu,
        'data': cpu_data
    }

def create_visualization(results):
    """Create visualization showing why these thresholds make sense"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Statistical Logic Behind Cloud SQL Utilization Thresholds', fontsize=16)
    
    patterns = ['underutilized', 'borderline', 'well_utilized', 'high_utilization']
    colors = ['red', 'orange', 'green', 'blue']
    
    for i, (pattern, color) in enumerate(zip(patterns, colors)):
        ax = axes[i//2, i%2]
        data = results[pattern]['data']
        
        # Histogram
        ax.hist(data, bins=50, alpha=0.7, color=color, density=True)
        
        # Add threshold lines
        ax.axvline(results[pattern]['avg'], color='blue', linestyle='-', 
                  label=f"Average: {results[pattern]['avg']:.1f}%")
        ax.axvline(results[pattern]['p95'], color='red', linestyle='--', 
                  label=f"P95: {results[pattern]['p95']:.1f}%")
        ax.axvline(40, color='orange', linestyle=':', alpha=0.8, label='40% Threshold')
        ax.axvline(60, color='purple', linestyle=':', alpha=0.8, label='60% Threshold')
        
        ax.set_title(f'{pattern.replace("_", " ").title()}\n{results[pattern]["decision"]}')
        ax.set_xlabel('CPU Utilization (%)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cpu_utilization_patterns.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'cpu_utilization_patterns.png'")

def main():
    """Demonstrate the statistical logic with different instance patterns"""
    
    print("🔍 STATISTICAL LOGIC BEHIND CLOUD SQL UTILIZATION THRESHOLDS")
    print("=" * 80)
    
    print("""
📚 BACKGROUND:
The thresholds (40%, 60%, 95th percentile) are based on:
1. Industry best practices (AWS, Azure, GCP)
2. Statistical significance (95% confidence)
3. Risk management (safety buffers)
4. Cost optimization principles

Let's analyze different instance patterns:
""")
    
    # Generate and analyze different patterns
    patterns = {
        'underutilized': 'Clearly underutilized instance',
        'borderline': 'Your carters-dev-new pattern (low avg, high peaks)',
        'well_utilized': 'Properly sized instance',
        'high_utilization': 'Heavily utilized instance'
    }
    
    results = {}
    
    for pattern, description in patterns.items():
        print(f"\n{description.upper()}")
        cpu_data = generate_realistic_cpu_data(pattern)
        results[pattern] = analyze_cpu_pattern(cpu_data, description)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("📊 SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Pattern':<20} {'Avg CPU':<10} {'P95 CPU':<10} {'Decision':<30} {'Recommended':<12}")
    print("-" * 80)
    
    for pattern in patterns.keys():
        r = results[pattern]
        print(f"{pattern:<20} {r['avg']:<10.1f} {r['p95']:<10.1f} {r['decision']:<30} {r['recommended_vcpu']:<12}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"1. 40% threshold catches instances wasting >60% of resources")
    print(f"2. P95 threshold prevents performance issues from rightsizing")
    print(f"3. Borderline cases need careful analysis (your instance!)")
    print(f"4. Statistical approach is more reliable than simple averages")
    
    # Create visualization
    try:
        create_visualization(results)
    except ImportError:
        print("\n📊 Install matplotlib to see visualizations: pip install matplotlib")

if __name__ == "__main__":
    main() 
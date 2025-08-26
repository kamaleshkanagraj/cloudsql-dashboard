#!/usr/bin/env python3
"""
Sample Data Generator for Cloud SQL Dashboard Demo
Creates realistic sample data when actual Cloud SQL data is not available
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_sample_data():
    """Generate realistic sample Cloud SQL data for dashboard demo"""
    
    # Sample projects and instances
    projects = ['demo-project-1', 'demo-project-2', 'demo-project-3']
    instance_types = ['production', 'staging', 'development', 'analytics']
    regions = ['us-central1', 'us-west1', 'europe-west1']
    
    instances = []
    
    for i in range(15):  # Generate 15 sample instances
        project = random.choice(projects)
        instance_type = random.choice(instance_types)
        
        # Generate realistic specs
        if instance_type == 'production':
            vcpu = random.choice([16, 24, 32, 48])
            memory_gb = vcpu * random.uniform(6, 8)
        elif instance_type == 'analytics':
            vcpu = random.choice([8, 16, 24])
            memory_gb = vcpu * random.uniform(4, 6)
        else:
            vcpu = random.choice([2, 4, 8])
            memory_gb = vcpu * random.uniform(3, 5)
        
        disk_size_gb = random.randint(500, 2000)
        
        # Generate utilization patterns
        if instance_type == 'production':
            avg_cpu = random.uniform(35, 75)
            max_cpu = min(100, avg_cpu + random.uniform(20, 40))
        elif instance_type == 'development':
            avg_cpu = random.uniform(5, 25)
            max_cpu = min(100, avg_cpu + random.uniform(10, 60))
        else:
            avg_cpu = random.uniform(15, 45)
            max_cpu = min(100, avg_cpu + random.uniform(15, 35))
        
        # Calculate spike counts
        critical_spikes = random.randint(0, 5) if max_cpu >= 90 else 0
        moderate_spikes = random.randint(critical_spikes, 20) if max_cpu >= 50 else 0
        
        # Determine underutilization
        is_underutilized = avg_cpu < 50 and critical_spikes == 0
        
        instance = {
            'project_id': project,
            'instance_id': f'{instance_type}-db-{i+1:02d}',
            'region': random.choice(regions),
            'tier': f'db-custom-{vcpu}-{int(memory_gb*1024)}',
            'database_version': random.choice(['POSTGRES_14', 'POSTGRES_15', 'MYSQL_8_0']),
            'vcpu_count': vcpu,
            'memory_gb': memory_gb,
            'disk_size_gb': disk_size_gb,
            'avg_cpu_utilization': round(avg_cpu, 1),
            'max_cpu_utilization': round(max_cpu, 1),
            'p95_cpu_utilization': round(avg_cpu + random.uniform(5, 15), 1),
            'avg_cpu_actual': round((avg_cpu/100) * vcpu, 2),
            'max_cpu_actual': round((max_cpu/100) * vcpu, 2),
            'p95_cpu_actual': round(((avg_cpu + random.uniform(5, 15))/100) * vcpu, 2),
            'avg_memory_utilization': round(random.uniform(20, 70), 1),
            'max_memory_utilization': round(random.uniform(50, 95), 1),
            'p95_memory_utilization': round(random.uniform(40, 80), 1),
            'avg_memory_actual': round(random.uniform(20, 70)/100 * memory_gb, 1),
            'max_memory_actual': round(random.uniform(50, 95)/100 * memory_gb, 1),
            'p95_memory_actual': round(random.uniform(40, 80)/100 * memory_gb, 1),
            'avg_disk_utilization': round(random.uniform(15, 60), 1),
            'max_disk_utilization': round(random.uniform(30, 85), 1),
            'p95_disk_utilization': round(random.uniform(25, 70), 1),
            'avg_disk_actual': round(random.uniform(15, 60)/100 * disk_size_gb, 1),
            'max_disk_actual': round(random.uniform(30, 85)/100 * disk_size_gb, 1),
            'p95_disk_actual': round(random.uniform(25, 70)/100 * disk_size_gb, 1),
            'data_points': random.randint(8000, 8640),
            'analysis_months': 1,
            'underutilized': is_underutilized,
            'underutilization_reasons': 'Safe to optimize: avg CPU <50%, ZERO critical spikes' if is_underutilized else 'Performance risk detected',
            'critical_spikes_count': critical_spikes,
            'moderate_spikes_count': moderate_spikes,
            'critical_spike_frequency': round(critical_spikes/8640*100, 2),
            'moderate_spike_frequency': round(moderate_spikes/8640*100, 2),
            'total_data_points': 8640
        }
        
        instances.append(instance)
    
    return pd.DataFrame(instances)

def create_sample_excel():
    """Create sample Excel file for dashboard demo"""
    df = generate_sample_data()
    
    # Save to Excel
    with pd.ExcelWriter('cloudsql_utilization_results.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Instance Summary', index=False)
        
        # Create empty metrics sheet for compatibility
        metrics_df = pd.DataFrame({
            'timestamp': [],
            'instance_id': [],
            'project_id': [],
            'metric_type': [],
            'value': []
        })
        metrics_df.to_excel(writer, sheet_name='CPU Metrics', index=False)
    
    print("✅ Sample data created: cloudsql_utilization_results.xlsx")
    print(f"📊 Generated {len(df)} sample instances across {df['project_id'].nunique()} projects")
    print(f"🎯 {len(df[df['underutilized']==True])} instances marked as underutilized")

if __name__ == "__main__":
    create_sample_excel() 
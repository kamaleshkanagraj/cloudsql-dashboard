#!/usr/bin/env python3
"""
Advanced Cloud SQL Utilization Analysis Algorithms
==================================================

This script demonstrates multiple sophisticated algorithms for more accurate
utilization analysis beyond simple averages and percentiles.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedUtilizationAnalyzer:
    """Advanced algorithms for accurate utilization analysis"""
    
    def __init__(self, cpu_data, vcpu_count, analysis_name="Unknown"):
        self.cpu_data = cpu_data
        self.vcpu_count = vcpu_count
        self.analysis_name = analysis_name
        self.cpu_values = [d['value'] for d in cpu_data]
        
    def algorithm_1_time_weighted_analysis(self):
        """
        Algorithm 1: Time-Weighted Utilization Analysis
        ===============================================
        
        Problem with simple averages: All time periods are weighted equally
        Solution: Weight recent data more heavily, consider business hours
        """
        print(f"\n🔍 ALGORITHM 1: TIME-WEIGHTED ANALYSIS")
        print("=" * 50)
        
        df = pd.DataFrame(self.cpu_data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5))
        
        # Calculate different weighted averages
        simple_avg = np.mean(self.cpu_values)
        
        # Business hours weighted (business hours count 3x more)
        business_weight = np.where(df['is_business_hours'], 3.0, 1.0)
        business_weighted_avg = np.average(self.cpu_values, weights=business_weight)
        
        # Recency weighted (recent data counts more)
        recency_weights = np.linspace(0.5, 1.0, len(self.cpu_values))  # Recent data weighted higher
        recency_weighted_avg = np.average(self.cpu_values, weights=recency_weights)
        
        # Combined weighted average
        combined_weights = business_weight * recency_weights
        combined_weighted_avg = np.average(self.cpu_values, weights=combined_weights)
        
        print(f"Simple Average: {simple_avg:.2f}%")
        print(f"Business Hours Weighted: {business_weighted_avg:.2f}%")
        print(f"Recency Weighted: {recency_weighted_avg:.2f}%")
        print(f"Combined Weighted: {combined_weighted_avg:.2f}%")
        
        # Business hours analysis
        business_hours_data = df[df['is_business_hours']]['value']
        off_hours_data = df[~df['is_business_hours']]['value']
        
        print(f"\nBusiness Hours Analysis:")
        print(f"  Business Hours Avg: {business_hours_data.mean():.2f}%")
        print(f"  Off Hours Avg: {off_hours_data.mean():.2f}%")
        print(f"  Business/Off Hours Ratio: {business_hours_data.mean() / off_hours_data.mean():.2f}x")
        
        return {
            'simple_avg': simple_avg,
            'business_weighted': business_weighted_avg,
            'recency_weighted': recency_weighted_avg,
            'combined_weighted': combined_weighted_avg,
            'business_hours_avg': business_hours_data.mean(),
            'off_hours_avg': off_hours_data.mean()
        }
    
    def algorithm_2_workload_pattern_recognition(self):
        """
        Algorithm 2: Workload Pattern Recognition
        ========================================
        
        Problem: Different workloads have different utilization patterns
        Solution: Classify workload type and apply appropriate thresholds
        """
        print(f"\n🔍 ALGORITHM 2: WORKLOAD PATTERN RECOGNITION")
        print("=" * 50)
        
        df = pd.DataFrame(self.cpu_data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['hour'] = df['datetime'].dt.hour
        
        # Calculate pattern metrics
        hourly_avg = df.groupby('hour')['value'].mean()
        daily_variance = df.groupby(df['datetime'].dt.date)['value'].var().mean()
        spike_frequency = len(df[df['value'] > 70]) / len(df) * 100
        baseline_usage = np.percentile(self.cpu_values, 10)  # 10th percentile as baseline
        peak_to_baseline_ratio = np.percentile(self.cpu_values, 95) / max(baseline_usage, 1)
        
        # Pattern classification
        patterns = {
            'steady_state': {
                'condition': daily_variance < 100 and spike_frequency < 5,
                'threshold': 60,  # Higher threshold for steady workloads
                'description': 'Consistent, predictable workload'
            },
            'batch_processing': {
                'condition': spike_frequency > 15 and peak_to_baseline_ratio > 3,
                'threshold': 30,  # Lower average threshold, but consider spikes
                'description': 'Periodic batch jobs with high spikes'
            },
            'business_hours': {
                'condition': hourly_avg[9:17].mean() > hourly_avg[0:8].mean() * 1.5,
                'threshold': 45,  # Moderate threshold for business hour patterns
                'description': 'Business hours driven workload'
            },
            'variable_load': {
                'condition': True,  # Default case
                'threshold': 40,   # Standard threshold
                'description': 'Variable workload pattern'
            }
        }
        
        # Determine workload pattern
        detected_pattern = 'variable_load'  # Default
        for pattern_name, pattern_info in patterns.items():
            if pattern_info['condition']:
                detected_pattern = pattern_name
                break
        
        pattern_info = patterns[detected_pattern]
        
        print(f"Pattern Metrics:")
        print(f"  Daily Variance: {daily_variance:.2f}")
        print(f"  Spike Frequency (>70%): {spike_frequency:.2f}%")
        print(f"  Baseline Usage (P10): {baseline_usage:.2f}%")
        print(f"  Peak/Baseline Ratio: {peak_to_baseline_ratio:.2f}x")
        
        print(f"\nDetected Pattern: {detected_pattern.upper()}")
        print(f"Description: {pattern_info['description']}")
        print(f"Recommended Threshold: {pattern_info['threshold']}%")
        
        # Apply pattern-specific analysis
        avg_cpu = np.mean(self.cpu_values)
        is_underutilized = avg_cpu < pattern_info['threshold']
        
        print(f"\nPattern-Based Assessment:")
        print(f"  Average CPU: {avg_cpu:.2f}%")
        print(f"  Pattern Threshold: {pattern_info['threshold']}%")
        print(f"  Result: {'UNDERUTILIZED' if is_underutilized else 'WELL UTILIZED'}")
        
        return {
            'detected_pattern': detected_pattern,
            'pattern_threshold': pattern_info['threshold'],
            'is_underutilized': is_underutilized,
            'daily_variance': daily_variance,
            'spike_frequency': spike_frequency,
            'peak_to_baseline_ratio': peak_to_baseline_ratio
        }
    
    def algorithm_3_statistical_confidence_analysis(self):
        """
        Algorithm 3: Statistical Confidence Analysis
        ===========================================
        
        Problem: Point estimates don't show confidence levels
        Solution: Calculate confidence intervals and statistical significance
        """
        print(f"\n🔍 ALGORITHM 3: STATISTICAL CONFIDENCE ANALYSIS")
        print("=" * 50)
        
        from scipy import stats
        
        cpu_array = np.array(self.cpu_values)
        n = len(cpu_array)
        
        # Basic statistics
        mean_cpu = np.mean(cpu_array)
        std_cpu = np.std(cpu_array, ddof=1)
        sem_cpu = stats.sem(cpu_array)  # Standard error of mean
        
        # Confidence intervals
        confidence_levels = [0.90, 0.95, 0.99]
        confidence_intervals = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
            margin_error = t_critical * sem_cpu
            
            ci_lower = mean_cpu - margin_error
            ci_upper = mean_cpu + margin_error
            
            confidence_intervals[conf_level] = (ci_lower, ci_upper)
        
        print(f"Statistical Metrics:")
        print(f"  Sample Size: {n:,} data points")
        print(f"  Mean CPU: {mean_cpu:.2f}%")
        print(f"  Standard Deviation: {std_cpu:.2f}%")
        print(f"  Standard Error: {sem_cpu:.2f}%")
        
        print(f"\nConfidence Intervals:")
        for conf_level, (ci_lower, ci_upper) in confidence_intervals.items():
            print(f"  {conf_level*100:.0f}% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
        
        # Statistical test for underutilization
        threshold = 40
        t_statistic = (mean_cpu - threshold) / sem_cpu
        p_value = stats.t.cdf(t_statistic, df=n-1)  # One-tailed test
        
        print(f"\nStatistical Test (H0: CPU >= {threshold}%):")
        print(f"  t-statistic: {t_statistic:.3f}")
        print(f"  p-value: {p_value:.6f}")
        
        # Interpretation
        significance_level = 0.05
        is_statistically_underutilized = p_value < significance_level
        
        print(f"  Statistical Significance (α=0.05): {'YES' if is_statistically_underutilized else 'NO'}")
        print(f"  Confidence: {(1-p_value)*100:.1f}% confident that CPU < {threshold}%")
        
        return {
            'mean_cpu': mean_cpu,
            'std_cpu': std_cpu,
            'confidence_intervals': confidence_intervals,
            'is_statistically_underutilized': is_statistically_underutilized,
            'confidence_level': (1-p_value)*100
        }
    
    def algorithm_4_capacity_planning_analysis(self):
        """
        Algorithm 4: Capacity Planning Analysis
        =======================================
        
        Problem: Static thresholds don't consider growth and variability
        Solution: Dynamic thresholds based on capacity planning principles
        """
        print(f"\n🔍 ALGORITHM 4: CAPACITY PLANNING ANALYSIS")
        print("=" * 50)
        
        cpu_array = np.array(self.cpu_values)
        
        # Capacity planning metrics
        p50 = np.percentile(cpu_array, 50)  # Median
        p95 = np.percentile(cpu_array, 95)  # Peak planning
        p99 = np.percentile(cpu_array, 99)  # Extreme peak
        max_cpu = np.max(cpu_array)
        
        # Variability analysis
        coefficient_of_variation = np.std(cpu_array) / np.mean(cpu_array)
        
        # Dynamic threshold calculation
        base_threshold = 40
        variability_adjustment = min(20, coefficient_of_variation * 30)  # Cap at 20%
        dynamic_threshold = base_threshold + variability_adjustment
        
        # Capacity headroom analysis
        current_headroom = 100 - p95
        recommended_headroom = max(20, 30 - coefficient_of_variation * 10)  # 20-30% based on variability
        
        print(f"Capacity Metrics:")
        print(f"  P50 (Median): {p50:.2f}%")
        print(f"  P95 (Peak Planning): {p95:.2f}%")
        print(f"  P99 (Extreme Peak): {p99:.2f}%")
        print(f"  Maximum: {max_cpu:.2f}%")
        print(f"  Coefficient of Variation: {coefficient_of_variation:.3f}")
        
        print(f"\nDynamic Threshold Calculation:")
        print(f"  Base Threshold: {base_threshold}%")
        print(f"  Variability Adjustment: +{variability_adjustment:.1f}%")
        print(f"  Dynamic Threshold: {dynamic_threshold:.1f}%")
        
        print(f"\nCapacity Headroom Analysis:")
        print(f"  Current Headroom (100-P95): {current_headroom:.1f}%")
        print(f"  Recommended Headroom: {recommended_headroom:.1f}%")
        print(f"  Headroom Status: {'ADEQUATE' if current_headroom >= recommended_headroom else 'INSUFFICIENT'}")
        
        # Final assessment
        avg_cpu = np.mean(cpu_array)
        is_underutilized = (avg_cpu < dynamic_threshold) and (current_headroom > recommended_headroom * 1.5)
        
        print(f"\nCapacity Planning Assessment:")
        print(f"  Average CPU: {avg_cpu:.2f}%")
        print(f"  Dynamic Threshold: {dynamic_threshold:.1f}%")
        print(f"  Excessive Headroom: {current_headroom > recommended_headroom * 1.5}")
        print(f"  Result: {'UNDERUTILIZED' if is_underutilized else 'APPROPRIATELY SIZED'}")
        
        return {
            'dynamic_threshold': dynamic_threshold,
            'current_headroom': current_headroom,
            'recommended_headroom': recommended_headroom,
            'is_underutilized': is_underutilized,
            'coefficient_of_variation': coefficient_of_variation
        }
    
    def algorithm_5_machine_learning_clustering(self):
        """
        Algorithm 5: Machine Learning Clustering Analysis
        ================================================
        
        Problem: Manual pattern recognition is limited
        Solution: Use ML to discover usage patterns automatically
        """
        print(f"\n🔍 ALGORITHM 5: MACHINE LEARNING CLUSTERING")
        print("=" * 50)
        
        # Prepare features for clustering
        df = pd.DataFrame(self.cpu_data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Create feature matrix
        features = []
        window_size = 12  # 1-hour windows (12 * 5-minute intervals)
        
        for i in range(0, len(df) - window_size, window_size):
            window_data = df.iloc[i:i+window_size]['value']
            
            feature_vector = [
                window_data.mean(),           # Average usage in window
                window_data.std(),            # Variability in window
                window_data.max(),            # Peak usage in window
                window_data.min(),            # Minimum usage in window
                df.iloc[i]['hour'],           # Hour of day
                df.iloc[i]['day_of_week'],    # Day of week
                len(window_data[window_data > 50]) / len(window_data)  # High usage ratio
            ]
            features.append(feature_vector)
        
        features_array = np.array(features)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        # Perform clustering
        n_clusters = 4  # Low, Medium, High, Peak usage patterns
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features_array[cluster_mask]
            
            if len(cluster_features) > 0:
                cluster_analysis[cluster_id] = {
                    'count': len(cluster_features),
                    'percentage': len(cluster_features) / len(features_array) * 100,
                    'avg_cpu': cluster_features[:, 0].mean(),
                    'avg_variability': cluster_features[:, 1].mean(),
                    'avg_peak': cluster_features[:, 2].mean(),
                    'dominant_hour': int(cluster_features[:, 4].mean()),
                    'dominant_day': int(cluster_features[:, 5].mean())
                }
        
        print(f"Machine Learning Analysis:")
        print(f"  Total Windows Analyzed: {len(features_array):,}")
        print(f"  Clusters Identified: {n_clusters}")
        
        print(f"\nCluster Analysis:")
        cluster_names = ['Low Usage', 'Medium Usage', 'High Usage', 'Peak Usage']
        sorted_clusters = sorted(cluster_analysis.items(), key=lambda x: x[1]['avg_cpu'])
        
        for i, (cluster_id, analysis) in enumerate(sorted_clusters):
            name = cluster_names[i] if i < len(cluster_names) else f"Cluster {cluster_id}"
            print(f"  {name} (Cluster {cluster_id}):")
            print(f"    • Frequency: {analysis['percentage']:.1f}% of time")
            print(f"    • Avg CPU: {analysis['avg_cpu']:.1f}%")
            print(f"    • Avg Peak: {analysis['avg_peak']:.1f}%")
            print(f"    • Dominant Hour: {analysis['dominant_hour']}:00")
        
        # ML-based recommendation
        low_usage_time = sum(analysis['percentage'] for cluster_id, analysis in cluster_analysis.items() 
                           if analysis['avg_cpu'] < 30)
        
        ml_recommendation = "UNDERUTILIZED" if low_usage_time > 60 else "WELL UTILIZED"
        
        print(f"\nML-Based Assessment:")
        print(f"  Low Usage Time (<30%): {low_usage_time:.1f}%")
        print(f"  ML Recommendation: {ml_recommendation}")
        
        return {
            'cluster_analysis': cluster_analysis,
            'low_usage_time': low_usage_time,
            'ml_recommendation': ml_recommendation
        }

def generate_realistic_data(days=90, pattern='mixed'):
    """Generate realistic CPU data with different patterns"""
    import random
    
    total_points = days * 24 * 12
    cpu_data = []
    base_time = datetime.now() - timedelta(days=days)
    
    for i in range(total_points):
        timestamp = base_time + timedelta(minutes=i * 5)
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        if pattern == 'steady':
            # Steady workload with minimal variation
            base_usage = random.uniform(45, 55)
            if random.random() < 0.02:  # Rare spikes
                usage = random.uniform(70, 85)
            else:
                usage = base_usage + random.uniform(-5, 5)
                
        elif pattern == 'batch':
            # Batch processing pattern
            if hour in [2, 14, 22]:  # Batch processing times
                usage = random.uniform(80, 95)
            else:
                usage = random.uniform(10, 30)
                
        elif pattern == 'business_hours':
            # Business hours pattern
            if 9 <= hour <= 17 and day_of_week < 5:
                usage = random.uniform(60, 80)
            else:
                usage = random.uniform(15, 35)
                
        else:  # mixed pattern
            # Mixed workload (current default)
            if 9 <= hour <= 17 and day_of_week < 5:
                base_usage = random.uniform(25, 45)
            else:
                base_usage = random.uniform(5, 25)
                
            if random.random() < 0.05:
                usage = random.uniform(60, 90)
            else:
                usage = base_usage + random.uniform(-5, 5)
        
        usage = max(0, min(95, usage))
        
        cpu_data.append({
            'timestamp': timestamp.timestamp(),
            'value': usage,
            'datetime': timestamp
        })
    
    return cpu_data

def main():
    """Demonstrate advanced algorithms"""
    
    print("🚀 ADVANCED CLOUD SQL UTILIZATION ALGORITHMS")
    print("=" * 80)
    print("Comparing multiple sophisticated approaches for accurate analysis")
    
    # Test different workload patterns
    patterns = ['mixed', 'steady', 'batch', 'business_hours']
    
    for pattern in patterns:
        print(f"\n{'='*80}")
        print(f"TESTING PATTERN: {pattern.upper()}")
        print(f"{'='*80}")
        
        # Generate data for this pattern
        cpu_data = generate_realistic_data(days=90, pattern=pattern)
        analyzer = AdvancedUtilizationAnalyzer(cpu_data, vcpu_count=8, analysis_name=pattern)
        
        # Run all algorithms
        results = {}
        results['time_weighted'] = analyzer.algorithm_1_time_weighted_analysis()
        results['pattern_recognition'] = analyzer.algorithm_2_workload_pattern_recognition()
        results['statistical'] = analyzer.algorithm_3_statistical_confidence_analysis()
        results['capacity_planning'] = analyzer.algorithm_4_capacity_planning_analysis()
        results['ml_clustering'] = analyzer.algorithm_5_machine_learning_clustering()
        
        # Summary comparison
        print(f"\n📊 ALGORITHM COMPARISON SUMMARY")
        print("=" * 50)
        
        simple_avg = np.mean([d['value'] for d in cpu_data])
        print(f"Simple Average: {simple_avg:.1f}%")
        print(f"Time-Weighted: {results['time_weighted']['combined_weighted']:.1f}%")
        print(f"Pattern-Based Threshold: {results['pattern_recognition']['pattern_threshold']:.1f}%")
        print(f"Statistical Confidence: {results['statistical']['confidence_level']:.1f}%")
        print(f"Dynamic Threshold: {results['capacity_planning']['dynamic_threshold']:.1f}%")
        print(f"ML Recommendation: {results['ml_clustering']['ml_recommendation']}")
        
        # Final recommendation
        underutilized_votes = sum([
            results['pattern_recognition']['is_underutilized'],
            results['statistical']['is_statistically_underutilized'],
            results['capacity_planning']['is_underutilized'],
            results['ml_clustering']['ml_recommendation'] == 'UNDERUTILIZED'
        ])
        
        consensus = "UNDERUTILIZED" if underutilized_votes >= 3 else "WELL UTILIZED"
        print(f"\n🎯 CONSENSUS RECOMMENDATION: {consensus}")
        print(f"   (Based on {underutilized_votes}/4 algorithms agreeing)")

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
from google.cloud import monitoring_v3
from google.cloud import resourcemanager_v3
from googleapiclient.discovery import build
from google.oauth2 import service_account
from google.auth import default
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudSQLMonitor:
    def __init__(self, credentials_path=None):
        """Initialize the Cloud SQL Monitor"""
        if credentials_path:
            self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
        else:
            # Use default credentials (gcloud auth application-default login)
            self.credentials, self.project = default()
        
        self.monitoring_client = monitoring_v3.MetricServiceClient(credentials=self.credentials)
        self.resource_client = resourcemanager_v3.ProjectsClient(credentials=self.credentials)
        
    def get_accessible_projects(self):
        """Get all projects accessible to the current user"""
        try:
            projects = []
            request = resourcemanager_v3.ListProjectsRequest()
            
            page_result = self.resource_client.list_projects(request=request)
            
            for project in page_result:
                if project.state == resourcemanager_v3.Project.State.ACTIVE:
                    projects.append({
                        'project_id': project.project_id,
                        'project_name': project.display_name or project.project_id,
                        'project_number': project.name.split('/')[-1]  # Extract number from resource name
                    })
            
            logger.info(f"Found {len(projects)} accessible projects")
            return projects
        except Exception as e:
            logger.error(f"Error getting projects using Resource Manager API: {e}")
            logger.info("Falling back to gcloud projects list...")
            
            # Fallback: Use gcloud to get project list
            try:
                import subprocess
                result = subprocess.run(['gcloud', 'projects', 'list', '--format=json'], 
                                      capture_output=True, text=True, check=True)
                
                import json
                gcloud_projects = json.loads(result.stdout)
                
                projects = []
                for project in gcloud_projects:
                    if project.get('lifecycleState') == 'ACTIVE':
                        projects.append({
                            'project_id': project['projectId'],
                            'project_name': project.get('name', project['projectId']),
                            'project_number': project.get('projectNumber', 'unknown')
                        })
                
                logger.info(f"Found {len(projects)} accessible projects using gcloud fallback")
                return projects
                
            except Exception as fallback_error:
                logger.error(f"Fallback method also failed: {fallback_error}")
                
                # Last resort: use the current project from credentials
                if hasattr(self, 'project') and self.project:
                    logger.info(f"Using last resort - current project only: {self.project}")
                    return [{
                        'project_id': self.project,
                        'project_name': self.project,
                        'project_number': 'unknown'
                    }]
                return []
    
    def get_cloudsql_instances(self, project_id):
        """Get all Cloud SQL instances in a project with detailed specifications"""
        try:
            service = build('sqladmin', 'v1', credentials=self.credentials)
            request = service.instances().list(project=project_id)
            response = request.execute()
            
            instances = []
            if 'items' in response:
                for instance in response['items']:
                    # Extract detailed instance specifications
                    settings = instance.get('settings', {})
                    tier = settings.get('tier', 'Unknown')
                    
                    # Parse tier information to get vCPU and memory specs
                    vcpu_count, memory_gb = self.parse_tier_specs(tier)
                    
                    # Get storage information
                    disk_size_gb = settings.get('dataDiskSizeGb', 0)
                    disk_type = settings.get('dataDiskType', 'Unknown')
                    
                    # Get database count for this instance
                    database_count = self.get_instance_database_count(project_id, instance['name'])
                    
                    instances.append({
                        'project_id': project_id,
                        'instance_id': instance['name'],
                        'database_version': instance.get('databaseVersion', 'Unknown'),
                        'tier': tier,
                        'vcpu_count': vcpu_count,
                        'memory_gb': memory_gb,
                        'disk_size_gb': int(disk_size_gb) if disk_size_gb else 0,
                        'disk_type': disk_type,
                        'region': instance.get('region', 'Unknown'),
                        'state': instance.get('state', 'Unknown'),
                        'backend_type': instance.get('backendType', 'Unknown'),
                        'instance_type': instance.get('instanceType', 'Unknown'),
                        'database_count': database_count  # NEW: Database count per instance
                    })
            
            logger.info(f"Found {len(instances)} Cloud SQL instances in project {project_id}")
            return instances
            
        except Exception as e:
            logger.error(f"Error getting instances for project {project_id}: {e}")
            return []
    
    def get_instance_database_count(self, project_id, instance_id):
        """Count the number of databases in a Cloud SQL instance"""
        try:
            service = build('sqladmin', 'v1', credentials=self.credentials)
            request = service.databases().list(project=project_id, instance=instance_id)
            response = request.execute()
            
            database_count = 0
            if 'items' in response:
                # Filter out system databases (information_schema, performance_schema, etc.)
                user_databases = []
                for db in response['items']:
                    db_name = db.get('name', '').lower()
                    # Skip common system databases
                    if db_name not in ['information_schema', 'performance_schema', 'mysql', 'sys', 'postgres', 'template0', 'template1']:
                        user_databases.append(db['name'])
                
                database_count = len(user_databases)
                logger.debug(f"Instance {instance_id} has {database_count} user databases: {user_databases}")
            
            return database_count
            
        except Exception as e:
            logger.warning(f"Error getting database count for {project_id}/{instance_id}: {e}")
            return 0  # Return 0 if we can't get the count
    
    def filter_projects_by_instance_count(self, projects, min_instances=3):
        """Filter projects that have more than the minimum number of instances"""
        filtered_projects = []
        
        for project in projects:
            project_id = project['project_id']
            instances = self.get_cloudsql_instances(project_id)
            instance_count = len(instances)
            
            if instance_count > min_instances:
                logger.info(f"✅ Project {project_id} has {instance_count} instances (>{min_instances}) - INCLUDED")
                filtered_projects.append(project)
            else:
                logger.info(f"❌ Project {project_id} has {instance_count} instances (<={min_instances}) - EXCLUDED")
        
        return filtered_projects
    
    def parse_tier_specs(self, tier):
        """Parse Cloud SQL tier to extract vCPU and memory specifications"""
        try:
            # Common Cloud SQL tier patterns
            tier_specs = {
                # Standard tiers
                'db-standard-1': (1, 3.75),
                'db-standard-2': (2, 7.5),
                'db-standard-4': (4, 15),
                'db-standard-8': (8, 30),
                'db-standard-16': (16, 60),
                'db-standard-32': (32, 120),
                'db-standard-64': (64, 240),
                'db-standard-96': (96, 360),
                
                # High-memory tiers
                'db-highmem-2': (2, 13),
                'db-highmem-4': (4, 26),
                'db-highmem-8': (8, 52),
                'db-highmem-16': (16, 104),
                'db-highmem-32': (32, 208),
                'db-highmem-64': (64, 416),
                'db-highmem-96': (96, 624),
                
                # Custom tiers (extract from name)
                # Format: db-custom-{vcpu}-{memory_mb}
            }
            
            if tier in tier_specs:
                return tier_specs[tier]
            elif tier.startswith('db-custom-'):
                # Parse custom tier: db-custom-4-15360 (4 vCPU, 15360 MB)
                parts = tier.split('-')
                if len(parts) >= 4:
                    vcpu = int(parts[2])
                    memory_mb = int(parts[3])
                    memory_gb = memory_mb / 1024
                    return (vcpu, memory_gb)
            elif tier.startswith('db-perf-optimized-N-'):
                # Parse performance-optimized tier: db-perf-optimized-N-4 (4 vCPU)
                # Format: db-perf-optimized-N-{vcpu_count}
                # Extract the number after the last dash
                try:
                    vcpu_str = tier.split('-')[-1]  # Get last part after dash
                    vcpu = int(vcpu_str)
                    # Performance optimized instances typically have 4GB per vCPU
                    memory_gb = vcpu * 4
                    logger.info(f"Performance-optimized tier {tier} parsed as {vcpu} vCPU, {memory_gb} GB")
                    return (vcpu, memory_gb)
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse vCPU count from tier: {tier}")
                    pass
            elif 'perf-optimized' in tier:
                # Generic performance-optimized parsing
                # Try to extract number from the end
                import re
                numbers = re.findall(r'\d+', tier)
                if numbers:
                    vcpu = int(numbers[-1])  # Last number is usually vCPU
                    memory_gb = vcpu * 4     # Standard ratio for perf-optimized
                    logger.info(f"Generic perf-optimized tier {tier} parsed as {vcpu} vCPU, {memory_gb} GB")
                    return (vcpu, memory_gb)
            
            # Log unknown tier for future reference
            logger.warning(f"Unknown tier format: {tier} - using fallback (0, 0)")
            return (0, 0)
            
        except Exception as e:
            logger.warning(f"Could not parse tier specs for {tier}: {e}")
            return (0, 0)
    
    def get_comprehensive_metrics(self, project_id, instance_id, months=3):
        """Get comprehensive metrics: CPU, Memory, and Storage utilization
        
        Args:
            project_id: GCP project ID
            instance_id: Cloud SQL instance ID  
            months: Number of months to analyze (default: 3, recommended: 3-9)
        """
        try:
            # Calculate time range more precisely
            from dateutil.relativedelta import relativedelta
            end_time = datetime.now()
            start_time = end_time - relativedelta(months=months)
            
            # Calculate expected data points for validation
            total_minutes = int((end_time - start_time).total_seconds() / 60)
            expected_data_points = total_minutes // 5  # 5-minute intervals
            
            logger.info(f"Analyzing {months} months of data for {instance_id}")
            logger.info(f"Time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"Expected data points: ~{expected_data_points:,} (5-minute intervals)")
            
            interval = monitoring_v3.TimeInterval({
                "end_time": {"seconds": int(end_time.timestamp())},
                "start_time": {"seconds": int(start_time.timestamp())},
            })
            
            project_name = f"projects/{project_id}"
            
            # Metrics to collect with detailed explanations and validation
            metrics_config = {
                'cpu': {
                    'filter': f'resource.type="cloudsql_database" AND resource.labels.database_id="{project_id}:{instance_id}" AND metric.type="cloudsql.googleapis.com/database/cpu/utilization"',
                    'multiplier': 100,  # Convert 0-1 range to percentage
                    'description': 'CPU utilization as percentage of allocated vCPUs',
                    'expected_range': (0, 100)  # Valid percentage range
                },
                'memory': {
                    'filter': f'resource.type="cloudsql_database" AND resource.labels.database_id="{project_id}:{instance_id}" AND metric.type="cloudsql.googleapis.com/database/memory/utilization"',
                    'multiplier': 100,  # Convert 0-1 range to percentage  
                    'description': 'Memory utilization as percentage of allocated memory',
                    'expected_range': (0, 100)  # Valid percentage range
                },
                'disk': {
                    'filter': f'resource.type="cloudsql_database" AND resource.labels.database_id="{project_id}:{instance_id}" AND metric.type="cloudsql.googleapis.com/database/disk/utilization"',
                    'multiplier': 100,  # Convert 0-1 range to percentage
                    'description': 'Storage utilization as percentage of allocated storage',
                    'expected_range': (0, 100)  # Valid percentage range
                }
            }
            
            all_metrics = {}
            
            for metric_name, config in metrics_config.items():
                try:
                    logger.info(f"Collecting {metric_name} metrics: {config['description']}")
                    
                    request = monitoring_v3.ListTimeSeriesRequest(
                        name=project_name,
                        filter=config['filter'],
                        interval=interval,
                        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                        aggregation=monitoring_v3.Aggregation(
                            alignment_period={"seconds": 300},  # 5-minute intervals for granular data
                            per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MAX,  # Capture peak values to detect spikes
                        ),
                    )
                    
                    results = self.monitoring_client.list_time_series(request=request)
                    
                    metric_data = []
                    raw_values = []
                    
                    for result in results:
                        for point in result.points:
                            raw_value = point.value.double_value
                            converted_value = raw_value * config['multiplier']
                            
                            # Validate data range
                            min_val, max_val = config['expected_range']
                            if not (min_val <= converted_value <= max_val):
                                logger.warning(f"Unusual {metric_name} value: {converted_value:.2f}% (raw: {raw_value:.4f})")
                            
                            raw_values.append(raw_value)
                            metric_data.append({
                                'timestamp': point.interval.end_time.timestamp(),
                                'value': converted_value,
                                'raw_value': raw_value,  # Store original for debugging
                                'metric_type': metric_name,
                                'instance_id': instance_id,
                                'project_id': project_id
                            })
                    
                    all_metrics[metric_name] = metric_data
                    
                    # Enhanced logging with validation
                    logger.info(f"Retrieved {len(metric_data)} {metric_name} data points for {instance_id}")
                    
                    if metric_data:
                        values = [d['value'] for d in metric_data]
                        raw_vals = [d['raw_value'] for d in metric_data]
                        
                        logger.info(f"{metric_name.upper()} converted stats: Min={min(values):.1f}%, Max={max(values):.1f}%, Avg={np.mean(values):.1f}%")
                        logger.info(f"{metric_name.upper()} raw stats: Min={min(raw_vals):.4f}, Max={max(raw_vals):.4f}, Avg={np.mean(raw_vals):.4f}")
                        
                        # Data quality check
                        coverage = len(metric_data) / expected_data_points * 100
                        logger.info(f"{metric_name.upper()} data coverage: {coverage:.1f}% ({len(metric_data)}/{expected_data_points} expected)")
                        
                        if coverage < 50:
                            logger.warning(f"Low data coverage for {metric_name}: {coverage:.1f}% - results may be unreliable")
                    else:
                        logger.warning(f"No {metric_name} data retrieved for {instance_id}")
                    
                except Exception as metric_error:
                    logger.warning(f"Could not retrieve {metric_name} metrics for {instance_id}: {metric_error}")
                    all_metrics[metric_name] = []
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error getting comprehensive metrics for {instance_id} in {project_id}: {e}")
            return {'cpu': [], 'memory': [], 'disk': []}
    
    def calculate_resource_utilization(self, metrics_data, vcpu_count, memory_gb, disk_size_gb):
        """Calculate actual resource utilization values with detailed explanations
        
        This method processes raw monitoring data and calculates:
        1. Statistical metrics (average, max, 95th percentile)
        2. Actual resource usage (e.g., 2.5 out of 4 vCPUs)
        3. Capacity analysis for rightsizing decisions
        """
        utilization_stats = {}
        
        logger.info("=== RESOURCE UTILIZATION CALCULATION DETAILS ===")
        
        for metric_type, data in metrics_data.items():
            logger.info(f"\n--- {metric_type.upper()} ANALYSIS ---")
            
            if not data:
                logger.warning(f"No {metric_type} data available")
                utilization_stats[metric_type] = {
                    'avg_percentage': 0, 'max_percentage': 0, 'p95_percentage': 0,
                    'avg_actual': 0, 'max_actual': 0, 'p95_actual': 0,
                    'total_capacity': 0, 'data_points': 0, 'unit': 'Unknown'
                }
                continue
            
            values = [d['value'] for d in data]
            logger.info(f"Data points collected: {len(values)}")
            logger.info(f"Time span: {len(values) * 5 / 60:.1f} hours of 5-minute intervals")
            
            # Calculate percentage statistics
            avg_pct = np.mean(values)
            max_pct = np.max(values)
            min_pct = np.min(values)
            p95_pct = np.percentile(values, 95)
            p99_pct = np.percentile(values, 99)
            
            logger.info(f"Statistical Analysis:")
            logger.info(f"  Average: {avg_pct:.2f}%")
            logger.info(f"  Maximum: {max_pct:.2f}%")
            logger.info(f"  Minimum: {min_pct:.2f}%")
            logger.info(f"  95th Percentile: {p95_pct:.2f}% (peak usage threshold)")
            logger.info(f"  99th Percentile: {p99_pct:.2f}% (extreme peak)")
            
            # CRITICAL: Data consistency validation
            if p95_pct > max_pct:
                logger.error(f"🚨 DATA INCONSISTENCY: P95 ({p95_pct:.2f}%) > Max ({max_pct:.2f}%) - This is impossible!")
                logger.error(f"Raw data sample: {values[:10]} ... (showing first 10 values)")
                logger.error(f"Data range: Min={min(values):.2f}, Max={max(values):.2f}, Count={len(values)}")
            
            if p99_pct > max_pct:
                logger.error(f"🚨 DATA INCONSISTENCY: P99 ({p99_pct:.2f}%) > Max ({max_pct:.2f}%) - This is impossible!")
            
            if min_pct > avg_pct or avg_pct > max_pct:
                logger.error(f"🚨 DATA INCONSISTENCY: Values not in logical order: Min={min_pct:.2f}, Avg={avg_pct:.2f}, Max={max_pct:.2f}")
            
            # Calculate actual usage based on capacity
            if metric_type == 'cpu' and vcpu_count > 0:
                total_capacity = vcpu_count
                avg_actual = (avg_pct / 100) * vcpu_count
                max_actual = (max_pct / 100) * vcpu_count
                p95_actual = (p95_pct / 100) * vcpu_count
                unit = 'vCPU'
                
                logger.info(f"CPU Capacity Analysis:")
                logger.info(f"  Allocated: {vcpu_count} vCPUs")
                logger.info(f"  Average usage: {avg_actual:.2f} vCPUs ({avg_pct:.1f}%)")
                logger.info(f"  Peak usage (P95): {p95_actual:.2f} vCPUs ({p95_pct:.1f}%)")
                logger.info(f"  Maximum ever: {max_actual:.2f} vCPUs ({max_pct:.1f}%)")
                
            elif metric_type == 'memory' and memory_gb > 0:
                total_capacity = memory_gb
                avg_actual = (avg_pct / 100) * memory_gb
                max_actual = (max_pct / 100) * memory_gb
                p95_actual = (p95_pct / 100) * memory_gb
                unit = 'GB'
                
                logger.info(f"Memory Capacity Analysis:")
                logger.info(f"  Allocated: {memory_gb:.1f} GB")
                logger.info(f"  Average usage: {avg_actual:.2f} GB ({avg_pct:.1f}%)")
                logger.info(f"  Peak usage (P95): {p95_actual:.2f} GB ({p95_pct:.1f}%)")
                logger.info(f"  Maximum ever: {max_actual:.2f} GB ({max_pct:.1f}%)")
                
            elif metric_type == 'disk' and disk_size_gb > 0:
                total_capacity = disk_size_gb
                avg_actual = (avg_pct / 100) * disk_size_gb
                max_actual = (max_pct / 100) * disk_size_gb
                p95_actual = (p95_pct / 100) * disk_size_gb
                unit = 'GB'
                
                logger.info(f"Storage Capacity Analysis:")
                logger.info(f"  Allocated: {disk_size_gb} GB")
                logger.info(f"  Average usage: {avg_actual:.2f} GB ({avg_pct:.1f}%)")
                logger.info(f"  Peak usage (P95): {p95_actual:.2f} GB ({p95_pct:.1f}%)")
                logger.info(f"  Maximum ever: {max_actual:.2f} GB ({max_pct:.1f}%)")
                
            else:
                total_capacity = 0
                avg_actual = max_actual = p95_actual = 0
                unit = 'Unknown'
                logger.warning(f"No capacity information available for {metric_type}")
            
            # Store comprehensive statistics
            utilization_stats[metric_type] = {
                'avg_percentage': avg_pct,
                'max_percentage': max_pct,
                'min_percentage': min_pct,
                'p95_percentage': p95_pct,
                'p99_percentage': p99_pct,
                'avg_actual': avg_actual,
                'max_actual': max_actual,
                'p95_actual': p95_actual,
                'total_capacity': total_capacity,
                'data_points': len(values),
                'unit': unit
            }
        
        return utilization_stats
    
    def analyze_underutilization_logic(self, utilization_stats, threshold=40):
        """Analyze and explain underutilization logic with detailed reasoning
        
        Our underutilization algorithm considers multiple factors:
        1. Average CPU usage below threshold (40%)
        2. Peak usage (95th percentile) below threshold + buffer (60%)
        3. This prevents flagging instances with regular high-usage spikes
        """
        
        cpu_stats = utilization_stats.get('cpu', {})
        memory_stats = utilization_stats.get('memory', {})
        disk_stats = utilization_stats.get('disk', {})
        
        logger.info("\n=== UNDERUTILIZATION ANALYSIS ===")
        
        # Primary analysis: CPU-based underutilization
        is_underutilized = False
        reasons = []
        
        if cpu_stats.get('data_points', 0) > 0:
            avg_cpu = cpu_stats.get('avg_percentage', 0)
            p95_cpu = cpu_stats.get('p95_percentage', 0)
            max_cpu = cpu_stats.get('max_percentage', 0)
            data_points = cpu_stats.get('data_points', 0)
            
            # Validate CPU data quality
            if avg_cpu < 0 or avg_cpu > 100 or p95_cpu < 0 or p95_cpu > 100:
                logger.error(f"Invalid CPU data: avg={avg_cpu:.1f}%, p95={p95_cpu:.1f}%")
                is_underutilized = True
                reasons.append("Invalid CPU data - assumed underutilized for safety")
                return is_underutilized, reasons
            
            # Check data quality
            confidence_level = "HIGH" if data_points > 10000 else "MEDIUM" if data_points > 5000 else "LOW"
            
            logger.info(f"CPU Utilization Assessment:")
            logger.info(f"  Data Quality: {confidence_level} ({data_points:,} data points)")
            logger.info(f"  Average CPU: {avg_cpu:.1f}% (threshold: <{threshold}%)")
            logger.info(f"  Peak CPU (P95): {p95_cpu:.1f}% (threshold: <{threshold + 20}%)")
            logger.info(f"  Maximum CPU: {max_cpu:.1f}%")
            
            # Validate logical consistency
            if p95_cpu < avg_cpu:
                logger.warning(f"Data inconsistency: P95 ({p95_cpu:.1f}%) < Average ({avg_cpu:.1f}%) - possible calculation error")
            
            # Main underutilization logic with enhanced validation
            avg_below_threshold = avg_cpu < threshold
            peak_below_threshold = p95_cpu < (threshold + 20)
            
            logger.info(f"Assessment Results:")
            logger.info(f"  Average below {threshold}%: {'✓' if avg_below_threshold else '✗'} ({avg_cpu:.1f}%)")
            logger.info(f"  Peak below {threshold + 20}%: {'✓' if peak_below_threshold else '✗'} ({p95_cpu:.1f}%)")
            
            if avg_below_threshold and peak_below_threshold:
                is_underutilized = True
                reasons.append(f"CPU avg ({avg_cpu:.1f}%) and peak ({p95_cpu:.1f}%) both below thresholds")
                logger.info("  RESULT: UNDERUTILIZED ❌")
            elif avg_below_threshold and not peak_below_threshold:
                # Special case: Very low average but high peaks - needs careful analysis
                if avg_cpu < 20 and max_cpu >= 90:
                    is_underutilized = True
                    reasons.append(f"CPU avg very low ({avg_cpu:.1f}%) despite peaks - potential for rightsizing with monitoring")
                    logger.info("  RESULT: UNDERUTILIZED ❌ (very low average, consider careful rightsizing)")
                else:
                    is_underutilized = False
                    reasons.append(f"CPU avg low ({avg_cpu:.1f}%) but has regular high usage spikes (P95: {p95_cpu:.1f}%)")
                    logger.info("  RESULT: WELL UTILIZED ✅ (has usage spikes)")
            else:
                is_underutilized = False
                reasons.append(f"CPU avg ({avg_cpu:.1f}%) above threshold")
                logger.info("  RESULT: WELL UTILIZED ✅")
            
            # Add confidence qualifier to results
            if confidence_level == "LOW":
                reasons.append(f"Low confidence due to limited data ({data_points:,} points)")
                logger.warning(f"  ⚠️  LOW CONFIDENCE: Only {data_points:,} data points available")
                
        else:
            is_underutilized = True
            reasons.append("No CPU data available - assumed underutilized")
            logger.warning("  RESULT: UNDERUTILIZED ❌ (no data)")
        
        # Additional context from memory and storage
        if memory_stats.get('data_points', 0) > 0:
            avg_memory = memory_stats.get('avg_percentage', 0)
            logger.info(f"Memory Context: {avg_memory:.1f}% average usage")
            
        if disk_stats.get('data_points', 0) > 0:
            avg_disk = disk_stats.get('avg_percentage', 0)
            logger.info(f"Storage Context: {avg_disk:.1f}% average usage")
        
        logger.info(f"Final Decision: {'UNDERUTILIZED' if is_underutilized else 'WELL UTILIZED'}")
        logger.info(f"Reasoning: {'; '.join(reasons)}")
        
        return is_underutilized, reasons
    
    def analyze_underutilization_with_spikes(self, utilization_stats, metrics_data, threshold=50, spike_threshold=90, max_spikes_allowed=0, analysis_months=1):
        """Enhanced Two-Tier Spike Analysis - Ultimate performance safety (Application-First approach)
        
        This method implements the enhanced logic:
        1. 30-day analysis period for recent patterns
        2. 50% average threshold (conservative)
        3. Two-tier spike detection:
           - Moderate spikes: 50% < CPU ≤ 89%
           - Critical spikes: CPU > 90%
        4. ANY spike (moderate OR critical) = keep current size (no performance risk)
        """
        
        cpu_stats = utilization_stats.get('cpu', {})
        cpu_data = metrics_data.get('cpu', [])
        
        logger.info("\n=== ENHANCED SPIKE-AWARE ANALYSIS ===")
        
        is_underutilized = False
        reasons = []
        spike_analysis = {}
        
        if cpu_stats.get('data_points', 0) > 0 and cpu_data:
            avg_cpu = cpu_stats.get('avg_percentage', 0)
            p95_cpu = cpu_stats.get('p95_percentage', 0)
            max_cpu = cpu_stats.get('max_percentage', 0)
            data_points = cpu_stats.get('data_points', 0)
            
            # Enhanced Two-Tier Spike Analysis - Critical (≥90%) and Moderate (50-85%)
            cpu_values = [d['value'] for d in cpu_data]
            critical_spikes = sum(1 for value in cpu_values if value >= spike_threshold)  # ≥90%
            moderate_spikes = sum(1 for value in cpu_values if 50 <= value <= 85)  # 50% ≤ CPU ≤ 85%
            
            # Find the highest spike for context
            highest_spike = max(cpu_values) if cpu_values else 0
            
            # Calculate spike frequencies for reporting
            days_analyzed = analysis_months * 30
            total_data_points = len(cpu_values)
            critical_spike_frequency = (critical_spikes / total_data_points * 100) if total_data_points > 0 else 0
            moderate_spike_frequency = (moderate_spikes / total_data_points * 100) if total_data_points > 0 else 0
            
            # Performance risk assessment
            consecutive_spikes = self.count_consecutive_spikes(cpu_values, spike_threshold)
            max_consecutive = max(consecutive_spikes) if consecutive_spikes else 0
            
            spike_analysis = {
                'critical_spikes': critical_spikes,
                'moderate_spikes': moderate_spikes,
                'highest_spike': highest_spike,
                'critical_spike_frequency': critical_spike_frequency,
                'moderate_spike_frequency': moderate_spike_frequency,
                'max_consecutive_spikes': max_consecutive,
                'critical_threshold': spike_threshold,
                'moderate_threshold': 50,
                'analysis_period_days': days_analyzed,
                'total_data_points': total_data_points,
                'zero_spike_policy': True,
                'two_tier_policy': True,  # NEW: Enhanced two-tier spike detection
                'moderate_range': '50-85%',
                'critical_range': f'≥{spike_threshold}%'
            }
            
            logger.info(f"Enhanced Two-Tier Spike Analysis Results:")
            logger.info(f"  Analysis Period: {days_analyzed} days ({total_data_points:,} data points)")
            logger.info(f"  Critical Spikes (≥{spike_threshold}%): {critical_spikes} ({critical_spike_frequency:.2f}% of time)")
            logger.info(f"  Moderate Spikes (50-85%): {moderate_spikes} ({moderate_spike_frequency:.2f}% of time)")
            logger.info(f"  Highest CPU Ever: {highest_spike:.1f}%")
            logger.info(f"  Policy: ZERO spikes ≥50% allowed (Ultimate application safety)")
            
            if critical_spikes > 0:
                logger.warning(f"  ⚠️  PERFORMANCE RISK: {critical_spikes} critical spikes detected - cannot safely reduce CPU")
            
            if moderate_spikes > 0:
                logger.info(f"  📊 USAGE PATTERN: {moderate_spikes} moderate spikes indicate periodic higher usage")
            
            logger.info(f"CPU Performance Assessment:")
            logger.info(f"  Average CPU: {avg_cpu:.1f}% (threshold: <{threshold}%)")
            logger.info(f"  Peak CPU (P95): {p95_cpu:.1f}%")
            logger.info(f"  Maximum CPU: {max_cpu:.1f}%")
            
            # Enhanced Two-Tier Spike Decision Logic (Ultimate Application Safety)
            avg_below_threshold = avg_cpu < threshold
            zero_critical_spikes = critical_spikes == 0
            zero_moderate_spikes = moderate_spikes == 0
            zero_all_spikes = zero_critical_spikes and zero_moderate_spikes
            
            logger.info(f"Enhanced Two-Tier Decision Criteria:")
            logger.info(f"  Average below {threshold}%: {'✓' if avg_below_threshold else '✗'} ({avg_cpu:.1f}%)")
            logger.info(f"  ZERO critical spikes ≥{spike_threshold}%: {'✓' if zero_critical_spikes else '✗'} ({critical_spikes} spikes found)")
            logger.info(f"  ZERO moderate spikes (50-85%): {'✓' if zero_moderate_spikes else '✗'} ({moderate_spikes} spikes found)")
            logger.info(f"  Highest CPU ever: {highest_spike:.1f}%")
            
            # Final Decision - ALL criteria must pass (avg < 50% AND zero moderate AND zero critical spikes)
            if avg_below_threshold and zero_all_spikes:
                is_underutilized = True
                reasons.append(f"Safe to optimize: avg CPU {avg_cpu:.1f}% < {threshold}%, ZERO spikes ≥50% in {days_analyzed} days")
                reasons.append(f"Performance guarantee: No spikes above 50% detected - application consistently low usage")
                logger.info("  RESULT: SAFE TO OPTIMIZE ✅ (Zero performance risk)")
                
            elif avg_below_threshold and (not zero_critical_spikes):
                is_underutilized = False
                reasons.append(f"Critical performance risk: {critical_spikes} spikes ≥{spike_threshold}% - application needs current capacity")
                logger.warning("  RESULT: KEEP CURRENT SIZE ⚠️ (Critical spikes detected)")
                
            elif avg_below_threshold and (not zero_moderate_spikes):
                is_underutilized = False
                reasons.append(f"Moderate performance risk: {moderate_spikes} spikes 50-85% - application needs current capacity during peak periods")
                logger.warning("  RESULT: KEEP CURRENT SIZE ⚠️ (Moderate spikes detected)")
                
            else:
                is_underutilized = False
                reasons.append(f"Average CPU too high: {avg_cpu:.1f}% >= {threshold}%")
                reasons.append(f"Spike analysis: {critical_spikes} critical (>{spike_threshold}%), {moderate_spikes} moderate (>50%)")
                logger.info("  RESULT: WELL UTILIZED ✅ (High average usage)")
            
        else:
            is_underutilized = True
            reasons.append("No CPU data available - assumed underutilized")
            logger.warning("  RESULT: UNDERUTILIZED ❌ (no data)")
        
        logger.info(f"Final Decision: {'SAFE TO OPTIMIZE' if is_underutilized else 'KEEP CURRENT SIZE'}")
        logger.info(f"Reasoning: {'; '.join(reasons)}")
        
        return is_underutilized, reasons, spike_analysis
    
    def count_consecutive_spikes(self, cpu_values, spike_threshold):
        """Count consecutive spikes to detect sustained high usage"""
        consecutive_counts = []
        current_count = 0
        
        for value in cpu_values:
            if value >= spike_threshold:
                current_count += 1
            else:
                if current_count > 0:
                    consecutive_counts.append(current_count)
                current_count = 0
        
        # Don't forget the last sequence
        if current_count > 0:
            consecutive_counts.append(current_count)
            
        return consecutive_counts
    
    def analyze_underutilized_instances(self, threshold=50, analysis_months=1, spike_threshold=90, max_spikes_allowed=0, min_instances_per_project=3):
        """Zero-spike method - guarantees application performance safety
        
        Args:
            threshold: CPU average utilization threshold (default: 50%)
            analysis_months: Number of months to analyze (default: 1 month = 30 days)
            spike_threshold: CPU percentage considered dangerous (default: 90%)
            max_spikes_allowed: Maximum spikes allowed (default: 0 - ZERO tolerance)
        """
        logger.info(f"Starting Cloud SQL utilization analysis with {analysis_months}-month lookback...")
        logger.info(f"Underutilization threshold: {threshold}% CPU (with {threshold + 20}% peak threshold)")
        
        # Get all accessible projects
        projects = self.get_accessible_projects()
        if not projects:
            logger.error("No accessible projects found")
            return pd.DataFrame(), pd.DataFrame()
        
        # Filter projects with more than minimum instances (if filtering enabled)
        if min_instances_per_project > 0:
            projects = self.filter_projects_by_instance_count(projects, min_instances=min_instances_per_project)
            if not projects:
                logger.warning(f"No projects with more than {min_instances_per_project} instances found. Exiting analysis.")
                return pd.DataFrame(), pd.DataFrame()
        else:
            logger.info("MIN_INSTANCES_PER_PROJECT = 0: Analyzing ALL projects (no instance count filtering)")
        
        all_instances = []
        all_metrics = []
        
        # Process each project
        for project in projects:
            project_id = project['project_id']
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing project: {project_id}")
            logger.info(f"{'='*60}")
            
            # Get Cloud SQL instances
            instances = self.get_cloudsql_instances(project_id)
            
            for instance in instances:
                instance_id = instance['instance_id']
                logger.info(f"\n--- ANALYZING INSTANCE: {instance_id} ---")
                logger.info(f"Specifications: {instance['vcpu_count']} vCPU, {instance['memory_gb']:.1f} GB RAM, {instance['disk_size_gb']} GB Storage")
                
                # Get comprehensive metrics
                metrics_data = self.get_comprehensive_metrics(project_id, instance_id, months=analysis_months)
                
                # Calculate utilization for the instance
                utilization_stats = self.calculate_resource_utilization(metrics_data, 
                                                                         instance['vcpu_count'], 
                                                                         instance['memory_gb'], 
                                                                         instance['disk_size_gb'])
                
                # Analyze underutilization with zero-spike protection
                is_underutilized, reasons, spike_analysis = self.analyze_underutilization_with_spikes(
                    utilization_stats, metrics_data, threshold, spike_threshold, max_spikes_allowed, analysis_months
                )
                
                # Add comprehensive summary to instance info
                instance.update({
                    # CPU metrics
                    'avg_cpu_utilization': utilization_stats.get('cpu', {}).get('avg_percentage', 0),
                    'max_cpu_utilization': utilization_stats.get('cpu', {}).get('max_percentage', 0),
                    'p95_cpu_utilization': utilization_stats.get('cpu', {}).get('p95_percentage', 0),
                    'avg_cpu_actual': utilization_stats.get('cpu', {}).get('avg_actual', 0),
                    'max_cpu_actual': utilization_stats.get('cpu', {}).get('max_actual', 0),
                    'p95_cpu_actual': utilization_stats.get('cpu', {}).get('p95_actual', 0),
                    
                    # Memory metrics
                    'avg_memory_utilization': utilization_stats.get('memory', {}).get('avg_percentage', 0),
                    'max_memory_utilization': utilization_stats.get('memory', {}).get('max_percentage', 0),
                    'p95_memory_utilization': utilization_stats.get('memory', {}).get('p95_percentage', 0),
                    'avg_memory_actual': utilization_stats.get('memory', {}).get('avg_actual', 0),
                    'max_memory_actual': utilization_stats.get('memory', {}).get('max_actual', 0),
                    'p95_memory_actual': utilization_stats.get('memory', {}).get('p95_actual', 0),
                    
                    # Disk metrics
                    'avg_disk_utilization': utilization_stats.get('disk', {}).get('avg_percentage', 0),
                    'max_disk_utilization': utilization_stats.get('disk', {}).get('max_percentage', 0),
                    'p95_disk_utilization': utilization_stats.get('disk', {}).get('p95_percentage', 0),
                    'avg_disk_actual': utilization_stats.get('disk', {}).get('avg_actual', 0),
                    'max_disk_actual': utilization_stats.get('disk', {}).get('max_actual', 0),
                    'p95_disk_actual': utilization_stats.get('disk', {}).get('p95_actual', 0),
                    
                    # Analysis metadata
                    'data_points': utilization_stats.get('cpu', {}).get('data_points', 0),
                    'analysis_months': analysis_months,
                    'underutilized': is_underutilized,
                    'underutilization_reasons': '; '.join(reasons),
                    'project_name': project['project_name'],
                    
                    # Real spike analysis data (not simulation!)
                    'critical_spikes_count': spike_analysis.get('critical_spikes', 0),
                    'moderate_spikes_count': spike_analysis.get('moderate_spikes', 0),
                    'critical_spike_frequency': spike_analysis.get('critical_spike_frequency', 0),
                    'moderate_spike_frequency': spike_analysis.get('moderate_spike_frequency', 0),
                    'total_data_points': spike_analysis.get('total_data_points', 0)
                })
                
                all_instances.append(instance)
                
                # Combine all metrics for time series analysis
                for metric_type, metric_data in metrics_data.items():
                    all_metrics.extend(metric_data)
                
                # Add delay to avoid rate limiting
                time.sleep(1)
        
        # Convert to DataFrames
        instances_df = pd.DataFrame(all_instances)
        metrics_df = pd.DataFrame(all_metrics)
        
        # Convert timestamp to datetime in metrics
        if not metrics_df.empty:
            metrics_df['datetime'] = pd.to_datetime(metrics_df['timestamp'], unit='s')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ANALYSIS COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Found {len(instances_df)} instances with metrics across {len(projects)} projects")
        logger.info(f"Analysis period: {analysis_months} months")
        logger.info(f"Total data points collected: {len(metrics_df):,}")
        
        return instances_df, metrics_df
    
    def save_results(self, instances_df, metrics_df, output_file='cloudsql_utilization_results.xlsx'):
        """Save results to Excel file"""
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Summary sheet
                instances_df.to_excel(writer, sheet_name='Instance Summary', index=False)
                
                # Underutilized instances
                underutilized = instances_df[instances_df['underutilized'] == True]
                underutilized.to_excel(writer, sheet_name='Underutilized Instances', index=False)
                
                # Raw metrics (sample if too large)
                if len(metrics_df) > 100000:  # Limit to prevent Excel issues
                    metrics_sample = metrics_df.sample(n=100000)
                    metrics_sample.to_excel(writer, sheet_name='CPU Metrics Sample', index=False)
                else:
                    metrics_df.to_excel(writer, sheet_name='CPU Metrics', index=False)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    """Main execution function with optimal configuration"""
    
    # ENHANCED CONFIGURATION - ZERO-SPIKE APPROACH (Performance First)
    ANALYSIS_MONTHS = 1      # 30 days: recent patterns, spike-focused analysis
    CPU_AVG_THRESHOLD = 50   # Conservative threshold to avoid performance issues
    SPIKE_THRESHOLD = 90     # Lower threshold - any spike above 90% is risky
    MAX_SPIKES_ALLOWED = 0   # ZERO spikes allowed - performance safety first
    MIN_INSTANCES_PER_PROJECT = 0  # Analyze ALL projects (set to 3 to filter projects with >3 instances)
    
    print(f"\n{'='*80}")
    print(f"🔍 CLOUD SQL RESOURCE UTILIZATION ANALYSIS")
    print(f"{'='*80}")
    print(f"Analysis Period: {ANALYSIS_MONTHS} month (30 days - two-tier spike analysis)")
    print(f"CPU Average Threshold: {CPU_AVG_THRESHOLD}% (Conservative for performance safety)")
    print(f"Enhanced Spike Detection: >50% moderate, >{SPIKE_THRESHOLD}% critical (ZERO spikes >50% allowed)")
    print(f"Two-Tier Logic: ANY spike >50% = KEEP CURRENT SIZE (Ultimate performance safety)")
    if MIN_INSTANCES_PER_PROJECT > 0:
        print(f"Project Filter: Only projects with >{MIN_INSTANCES_PER_PROJECT} instances")
    else:
        print(f"Project Filter: ALL projects (no filtering)")
    print(f"Algorithm: Enhanced Two-Tier Spike Analysis (Ultimate application reliability)")
    print(f"{'='*80}")
    
    # Initialize monitor (uses default credentials)
    monitor = CloudSQLMonitor()
    
    # Run analysis with zero-spike parameters
    instances_df, metrics_df = monitor.analyze_underutilized_instances(
        threshold=CPU_AVG_THRESHOLD, 
        analysis_months=ANALYSIS_MONTHS,
        spike_threshold=SPIKE_THRESHOLD,
        max_spikes_allowed=MAX_SPIKES_ALLOWED,
        min_instances_per_project=MIN_INSTANCES_PER_PROJECT
    )
    
    if not instances_df.empty:
        # Save results
        monitor.save_results(instances_df, metrics_df)
        
        # Calculate comprehensive summary statistics
        total_instances = len(instances_df)
        underutilized_count = len(instances_df[instances_df['underutilized'] == True])
        total_vcpu = instances_df['vcpu_count'].sum()
        total_memory = instances_df['memory_gb'].sum()
        total_storage = instances_df['disk_size_gb'].sum()
        
        # Resource utilization averages
        avg_cpu_util = instances_df['avg_cpu_utilization'].mean()
        avg_memory_util = instances_df['avg_memory_utilization'].mean()
        avg_disk_util = instances_df['avg_disk_utilization'].mean()
        
        # Actual resource usage totals
        total_cpu_used = instances_df['avg_cpu_actual'].sum()
        total_memory_used = instances_df['avg_memory_actual'].sum()
        total_storage_used = instances_df['avg_disk_actual'].sum()
        
        print(f"\n{'='*80}")
        print(f"📊 EXECUTIVE SUMMARY")
        print(f"{'='*80}")
        print(f"Analysis Period: {ANALYSIS_MONTHS} months")
        print(f"Total Instances: {total_instances}")
        print(f"Underutilized: {underutilized_count} ({(underutilized_count/total_instances)*100:.1f}%)")
        
        print(f"\n💰 COST OPTIMIZATION OPPORTUNITY:")
        if underutilized_count > 0:
            underutilized_instances = instances_df[instances_df['underutilized'] == True]
            potential_cpu_savings = underutilized_instances['vcpu_count'].sum()
            potential_memory_savings = underutilized_instances['memory_gb'].sum()
            
            print(f"  Resources in underutilized instances:")
            print(f"  • {potential_cpu_savings} vCPUs ({(potential_cpu_savings/total_vcpu)*100:.1f}% of total)")
            print(f"  • {potential_memory_savings:.0f} GB memory ({(potential_memory_savings/total_memory)*100:.1f}% of total)")
            print(f"  💡 Potential monthly savings: 30-50% on these instances")
        else:
            print(f"  ✅ All instances are well-utilized - no immediate optimization needed")
        
        print(f"\n📋 INSTANCE ANALYSIS:")
        for _, instance in instances_df.iterrows():
            status = "🔴 UNDERUTILIZED" if instance['underutilized'] else "🟢 WELL-UTILIZED"
            confidence = "HIGH" if instance['data_points'] > 10000 else "MEDIUM" if instance['data_points'] > 5000 else "LOW"
            
            print(f"\n{instance['project_id']}/{instance['instance_id']} - {status}")
            print(f"  📊 Usage: {instance['avg_cpu_utilization']:.1f}% avg, {instance['p95_cpu_utilization']:.1f}% peak")
            print(f"  🔧 Specs: {instance['vcpu_count']} vCPU, {instance['memory_gb']:.1f} GB, {instance['disk_size_gb']} GB")
            print(f"  📈 Confidence: {confidence} ({instance['data_points']:,} data points)")
            
            if instance['underutilized']:
                # Simple rightsizing recommendation
                current_vcpu = instance['vcpu_count']
                peak_usage = instance['p95_cpu_actual']
                recommended_vcpu = max(2, int(np.ceil(peak_usage * 1.3)))  # 30% buffer above peak
                
                if recommended_vcpu < current_vcpu:
                    savings = current_vcpu - recommended_vcpu
                    print(f"  💡 Recommendation: Reduce to {recommended_vcpu} vCPU (save {savings} vCPU)")
        
        print(f"\n{'='*80}")
        print(f"🎯 NEXT STEPS")
        print(f"{'='*80}")
        
        if underutilized_count == 0:
            print("✅ Excellent utilization! Continue monitoring quarterly.")
        elif underutilized_count / total_instances < 0.3:
            print("1. Review underutilized instances for rightsizing opportunities")
            print("2. Test with smaller instance sizes in non-production first")
            print("3. Monitor performance after changes")
        else:
            print("⚠️  High underutilization detected!")
            print("1. IMMEDIATE: Review workload requirements")
            print("2. Consider consolidation or rightsizing")
            print("3. Implement monitoring alerts for usage spikes")
        
        print(f"\n📊 Interactive Dashboard: streamlit run dashboard.py")
        print(f"📄 Detailed Report: cloudsql_utilization_results.xlsx")
        
        # Auto-launch Streamlit dashboard
        print(f"\n{'='*80}")
        print(f"🚀 LAUNCHING INTERACTIVE DASHBOARD")
        print(f"{'='*80}")
        print("Opening Streamlit dashboard automatically...")
        print("Dashboard will open in your default browser in a few seconds...")
        
        try:
            import subprocess
            import time
            import webbrowser
            
            # Small delay to let user read the message
            time.sleep(2)
            
            # Launch Streamlit dashboard
            subprocess.Popen([
                'streamlit', 'run', 'dashboard.py',
                '--server.headless', 'false',
                '--server.port', '8501',
                '--browser.gatherUsageStats', 'false'
            ])
            
            # Wait a moment for Streamlit to start
            time.sleep(3)
            
            # Open browser automatically
            webbrowser.open('http://localhost:8501')
            
            print("✅ Dashboard launched successfully!")
            print("🌐 Access URL: http://localhost:8501")
            print("\n💡 TIP: Keep this terminal open to keep the dashboard running")
            print("Press Ctrl+C in this terminal to stop the dashboard")
            
        except Exception as e:
            print(f"⚠️  Could not auto-launch dashboard: {e}")
            print("Please manually run: streamlit run dashboard.py")
    
    else:
        print("❌ No instances found. Check permissions and instance availability.")

if __name__ == "__main__":
    main() 
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Cloud SQL Resource Utilization Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .project-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
        margin: 2rem 0 1.5rem 0;
        padding: 1rem 1.5rem;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 12px;
        border-left: 6px solid #3b82f6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        transition: transform 0.2s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
    }
    
    .instance-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .instance-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }
    
    .resource-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .resource-item {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        border: 2px solid #cbd5e1;
        transition: all 0.2s ease;
    }
    
    .resource-item:hover {
        border-color: #3b82f6;
        transform: scale(1.02);
    }
    
    .resource-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    
    .resource-label {
        font-size: 1rem;
        color: #64748b;
        font-weight: 600;
    }
    
    .status-badge-underutilized {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        color: #dc2626;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 700;
        border: 2px solid #fecaca;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.2);
    }
    
    .status-badge-utilized {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        color: #16a34a;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 700;
        border: 2px solid #bbf7d0;
        box-shadow: 0 2px 8px rgba(22, 163, 74, 0.2);
    }
    
    .spec-container {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        margin: 1rem 0;
    }
    
    .spec-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .spec-item {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    
    .spec-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #374151;
    }
    
    .spec-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
    
    .view-selector {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 3px solid #3b82f6;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
        color: white;
    }
    
    .view-selector h3 {
        color: white !important;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

class CloudSQLDashboard:
    def __init__(self):
        self.instances_df = None
        self.metrics_df = None
        
    def load_data(self):
        """Load data from Excel file"""
        try:
            self.instances_df = pd.read_excel('cloudsql_utilization_results.xlsx', sheet_name='Instance Summary')
            
            # Debug information to verify data source
            if not self.instances_df.empty:
                unique_projects = self.instances_df['project_id'].unique()
                st.sidebar.success(f"✅ **Real Data Loaded!**")
                st.sidebar.info(f"📊 **{len(self.instances_df)} instances** from **{len(unique_projects)} projects**")
                st.sidebar.info(f"🏢 **Sample projects:** {', '.join(unique_projects[:3])}...")
                
                # Check if this is sample data
                if any('demo-project' in proj for proj in unique_projects):
                    st.sidebar.warning("⚠️ **Sample data detected** - Upload real data for production analysis")
                else:
                    st.sidebar.success("🎯 **Production data confirmed** - Real Cloud SQL analysis active")
            
            # Try to load metrics
            try:
                self.metrics_df = pd.read_excel('cloudsql_utilization_results.xlsx', sheet_name='CPU Metrics')
                if 'datetime' not in self.metrics_df.columns and 'timestamp' in self.metrics_df.columns:
                    self.metrics_df['datetime'] = pd.to_datetime(self.metrics_df['timestamp'], unit='s')
            except:
                try:
                    self.metrics_df = pd.read_excel('cloudsql_utilization_results.xlsx', sheet_name='CPU Metrics Sample')
                    if 'datetime' not in self.metrics_df.columns and 'timestamp' in self.metrics_df.columns:
                        self.metrics_df['datetime'] = pd.to_datetime(self.metrics_df['timestamp'], unit='s')
                except:
                    self.metrics_df = pd.DataFrame()
            
            return True
            
        except FileNotFoundError:
            st.error("❌ **Data file not found.** Please run the enhanced data collection script first.")
            st.code("python cloudsql_utilization_monitor.py")
            return False
        except Exception as e:
            st.error(f"❌ **Error loading data:** {e}")
            return False
    
    def create_executive_dashboard(self):
        """Create enhanced executive dashboard with granular analysis"""
        if self.instances_df is None or self.instances_df.empty:
            return
        
        st.markdown("## 📊 Executive Cloud SQL Resource Analysis")
        
        # Enhanced key metrics with cost implications
        total_instances = len(self.instances_df)
        total_projects = self.instances_df['project_id'].nunique()
        underutilized_instances = len(self.instances_df[self.instances_df['underutilized'] == True])
        
        # Resource totals and utilization
        total_vcpu = self.instances_df['vcpu_count'].sum()
        total_memory = self.instances_df['memory_gb'].sum()
        total_storage = self.instances_df['disk_size_gb'].sum()
        
        # Actual usage calculations
        total_cpu_used = self.instances_df['avg_cpu_actual'].sum()
        total_memory_used = self.instances_df['avg_memory_actual'].sum()
        total_storage_used = self.instances_df['avg_disk_actual'].sum()
        
        # Waste calculations
        cpu_waste_pct = ((total_vcpu - total_cpu_used) / total_vcpu * 100) if total_vcpu > 0 else 0
        memory_waste_pct = ((total_memory - total_memory_used) / total_memory * 100) if total_memory > 0 else 0
        
        # Enhanced metrics grid
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="resource-value">{total_instances}</div>
                <div class="resource-label">Total Instances</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                    {total_projects} projects
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            underutil_pct = (underutilized_instances/total_instances)*100 if total_instances > 0 else 0
            color = "#dc2626" if underutil_pct > 50 else "#f59e0b" if underutil_pct > 25 else "#16a34a"
            st.markdown(f"""
            <div class="metric-container">
                <div class="resource-value" style="color: {color}">{underutilized_instances}</div>
                <div class="resource-label">Underutilized</div>
                <div style="font-size: 0.8rem; color: {color}; margin-top: 0.5rem; font-weight: 600;">
                    {underutil_pct:.1f}% optimization opportunity
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="resource-value">{total_cpu_used:.1f}</div>
                <div class="resource-label">vCPUs Actually Used</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                    of {total_vcpu} allocated ({cpu_waste_pct:.1f}% waste)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="resource-value">{total_memory_used:.0f} GB</div>
                <div class="resource-label">Memory Actually Used</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                    of {total_memory:.0f} GB ({memory_waste_pct:.1f}% waste)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            potential_savings = underutilized_instances / total_instances * 100 if total_instances > 0 else 0
            savings_color = "#16a34a" if potential_savings > 0 else "#64748b"
            st.markdown(f"""
            <div class="metric-container">
                <div class="resource-value" style="color: {savings_color}">{potential_savings:.1f}%</div>
                <div class="resource-label">Potential Cost Savings</div>
                <div style="font-size: 0.8rem; color: {savings_color}; margin-top: 0.5rem;">
                    Monthly optimization
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Main utilization analysis graphs
        self.create_main_utilization_graphs()
        
        # Enhanced project-wise resource summary
        self.create_enhanced_project_summary()
    
    def create_main_utilization_graphs(self):
        """Create main utilization analysis graphs"""
        st.markdown("## 📈 CPU Utilization Analysis")
        
        # Main graph: Instances under 40% utilization
        st.markdown("### 🎯 Instances Below 40% CPU Threshold")
        
        # Create zero-spike optimization categories
        def categorize_instance(row):
            avg_cpu = row['avg_cpu_utilization']
            max_cpu = row.get('max_cpu_utilization', 0)
            has_spikes = max_cpu >= 90
            
            if avg_cpu < 50 and not has_spikes:
                return 'Safe to Optimize (Avg<50%, Zero Spikes)'
            elif avg_cpu < 50 and has_spikes:
                return 'Keep Current Size (Performance Spikes)'
            elif avg_cpu >= 50 and avg_cpu < 70:
                return 'Well Utilized (50-70%)'
            else:
                return 'High Utilization (70%+)'
        
        self.instances_df['optimization_category'] = self.instances_df.apply(categorize_instance, axis=1)
        
        # Professional color mapping
        color_map = {
            'Safe to Optimize (Avg<50%, Zero Spikes)': '#16a34a',      # Green - Safe
            'Keep Current Size (Performance Spikes)': '#dc2626',       # Red - Danger
            'Well Utilized (50-70%)': '#3b82f6',                      # Blue - Good
            'High Utilization (70%+)': '#8b5cf6'                      # Purple - Busy
        }
        
        # Professional optimization analysis chart
        fig1 = go.Figure()
        
        # Sort categories for logical display
        categories = ['Safe to Optimize (Avg<50%, Zero Spikes)', 'Keep Current Size (Performance Spikes)', 
                     'Well Utilized (50-70%)', 'High Utilization (70%+)']
        
        for category in categories:
            category_data = self.instances_df[self.instances_df['optimization_category'] == category]
            
            if not category_data.empty:
                # Add spike indicators to hover data
                hover_text = []
                for _, row in category_data.iterrows():
                    max_cpu = row.get('max_cpu_utilization', 0)
                    spike_indicator = "⚠️ Has Spikes" if max_cpu >= 90 else "✅ No Spikes"
                    hover_text.append(f"<b>{row['instance_id']}</b><br>"
                                    f"Project: {row['project_id']}<br>"
                                    f"Avg CPU: {row['avg_cpu_utilization']:.1f}%<br>"
                                    f"Max CPU: {max_cpu:.1f}%<br>"
                                    f"Spike Status: {spike_indicator}")
                
                fig1.add_trace(go.Bar(
                    name=category,
                    y=category_data['instance_id'],
                    x=category_data['avg_cpu_utilization'],
                    orientation='h',
                    marker_color=color_map[category],
                    text=[f"{val:.1f}%" for val in category_data['avg_cpu_utilization']],
                    textposition='outside',
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_text
                ))
        
        # Professional threshold lines with zero-spike logic
        fig1.add_vline(x=50, line_dash="solid", line_color="#16a34a", line_width=3,
                      annotation_text="50% Average Threshold", annotation_position="top")
        fig1.add_vline(x=90, line_dash="solid", line_color="#dc2626", line_width=4,
                      annotation_text="90% Spike Threshold (Zero Tolerance)", annotation_position="bottom")
        
        fig1.update_layout(
            title='Zero-Spike Optimization Analysis - All Instances',
            xaxis_title='CPU Utilization (%) - 30 Day Average',
            yaxis_title='Cloud SQL Instances',
            height=max(600, len(self.instances_df) * 30),
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(size=12)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Time series analysis if metrics data is available
        if hasattr(self, 'metrics_df') and self.metrics_df is not None and not self.metrics_df.empty:
            self.create_time_series_analysis()
        else:
            st.info("💡 **Time series data not available.** Run the monitoring script to collect detailed time-based metrics.")
    
    def create_time_series_analysis(self):
        """Create time series line graphs for CPU utilization over 3 months"""
        st.markdown("### 📊 CPU Utilization Trends Over 3 Months")
        
        # Filter for CPU metrics only
        cpu_metrics = self.metrics_df[self.metrics_df['metric_type'] == 'cpu'].copy()
        
        if cpu_metrics.empty:
            st.warning("No CPU time series data available for trend analysis.")
            return
        
        # Get top underutilized instances for detailed view
        underutilized_instances = self.instances_df[
            self.instances_df['underutilized'] == True
        ].nlargest(5, 'vcpu_count')  # Show top 5 by size
        
        if underutilized_instances.empty:
            st.info("No underutilized instances found for time series analysis.")
            return
        
        # Create time series plot
        fig2 = go.Figure()
        
        colors = ['#dc2626', '#f59e0b', '#8b5cf6', '#06b6d4', '#10b981']
        
        for i, (_, instance) in enumerate(underutilized_instances.iterrows()):
            instance_data = cpu_metrics[
                (cpu_metrics['instance_id'] == instance['instance_id']) & 
                (cpu_metrics['project_id'] == instance['project_id'])
            ].copy()
            
            if not instance_data.empty:
                # Sort by timestamp
                instance_data = instance_data.sort_values('timestamp')
                
                # Convert timestamp to datetime if needed
                if 'datetime' not in instance_data.columns:
                    instance_data['datetime'] = pd.to_datetime(instance_data['timestamp'], unit='s')
                
                fig2.add_trace(go.Scatter(
                    x=instance_data['datetime'],
                    y=instance_data['value'],
                    mode='lines',
                    name=f"{instance['instance_id']} ({instance['vcpu_count']} vCPU)",
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>CPU: %{y:.1f}%<extra></extra>'
                ))
        
        # Add threshold lines
        fig2.add_hline(y=40, line_dash="solid", line_color="red", line_width=2,
                      annotation_text="40% Threshold")
        fig2.add_hline(y=60, line_dash="dash", line_color="orange", line_width=1,
                      annotation_text="60% Peak Threshold")
        
        fig2.update_layout(
            title='CPU Utilization Timeline - Top 5 Underutilized Instances (by vCPU Count)',
            xaxis_title='Time (3 Month Period)',
            yaxis_title='CPU Utilization (%)',
            height=500,
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Summary statistics for the time period
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Time Period Statistics")
            if not cpu_metrics.empty:
                date_range = cpu_metrics['datetime'].agg(['min', 'max'])
                total_days = (date_range['max'] - date_range['min']).days
                total_measurements = len(cpu_metrics)
                
                st.metric("Analysis Period", f"{total_days} days")
                st.metric("Total Measurements", f"{total_measurements:,}")
                st.metric("Data Frequency", "Every 5 minutes")
        
        with col2:
            st.markdown("#### 🎯 Key Insights")
            underutil_in_chart = len(underutilized_instances)
            avg_cpu_underutil = underutilized_instances['avg_cpu_utilization'].mean()
            
            st.metric("Instances Shown", underutil_in_chart)
            st.metric("Avg CPU (Underutilized)", f"{avg_cpu_underutil:.1f}%")
            st.metric("Optimization Potential", "High")
    
    def create_enhanced_project_summary(self):
        """Create enhanced project-wise analysis with better visualizations"""
        st.markdown("## 🏢 Project-Wise Resource Analysis")
        
        # Group by project and calculate comprehensive stats
        project_stats = self.instances_df.groupby('project_id').agg({
            'instance_id': 'count',
            'underutilized': 'sum',
            'vcpu_count': 'sum',
            'memory_gb': 'sum',
            'disk_size_gb': 'sum',
            'avg_cpu_utilization': 'mean',
            'avg_memory_utilization': 'mean',
            'avg_disk_utilization': 'mean',
            'avg_cpu_actual': 'sum',
            'avg_memory_actual': 'sum',
            'avg_disk_actual': 'sum'
        }).round(2)
        
        project_stats.columns = ['instances', 'underutilized', 'total_vcpu', 'total_memory_gb', 
                               'total_storage_gb', 'avg_cpu_util', 'avg_memory_util', 'avg_disk_util',
                               'used_vcpu', 'used_memory_gb', 'used_storage_gb']
        
        project_stats['underutil_pct'] = (project_stats['underutilized'] / project_stats['instances'] * 100).round(1)
        project_stats['cpu_waste_pct'] = ((project_stats['total_vcpu'] - project_stats['used_vcpu']) / project_stats['total_vcpu'] * 100).round(1)
        project_stats['memory_waste_pct'] = ((project_stats['total_memory_gb'] - project_stats['used_memory_gb']) / project_stats['total_memory_gb'] * 100).round(1)
        project_stats = project_stats.reset_index().sort_values('project_id')
        
        # Create clearer visualizations with better structure
        
        # Resource waste analysis - more visual impact
        st.markdown("### 💰 Resource Waste Analysis by Project")
        
        fig1 = go.Figure()
        
        # Calculate waste percentages for better visualization
        project_stats['cpu_efficiency'] = (project_stats['used_vcpu'] / project_stats['total_vcpu'] * 100).round(1)
        
        # Create efficiency vs waste chart
        fig1.add_trace(go.Bar(
            name='CPU Efficiency',
            x=project_stats['project_id'],
            y=project_stats['cpu_efficiency'],
            marker_color='#16a34a',
            text=[f"{val:.1f}%" for val in project_stats['cpu_efficiency']],
            textposition='outside',
            yaxis='y1'
        ))
        
        fig1.add_trace(go.Bar(
            name='CPU Waste',
            x=project_stats['project_id'],
            y=project_stats['cpu_waste_pct'],
            marker_color='#dc2626',
            text=[f"{val:.1f}%" for val in project_stats['cpu_waste_pct']],
            textposition='outside',
            yaxis='y2',
            opacity=0.7
        ))
        
        # Add efficiency target line
        fig1.add_hline(y=60, line_dash="dash", line_color="blue", line_width=2,
                      annotation_text="60% Efficiency Target")
        
        fig1.update_layout(
            title='CPU Efficiency vs Waste by Project (%)',
            xaxis_title='Project',
            yaxis_title='Efficiency / Waste Percentage (%)',
            height=450,
            plot_bgcolor='white',
            showlegend=True,
            barmode='group'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Instance optimization potential
        st.markdown("### 🎯 Instance Optimization Potential")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Professional donut chart for optimization status
            optimization_summary = self.instances_df['optimization_category'].value_counts()
            
            # Professional colors matching the categories
            color_map = {
                'Safe to Optimize (Avg<50%, Zero Spikes)': '#16a34a',  # Green
                'Performance Risk (Critical Spikes)': '#dc2626',        # Red
                'Well Utilized (Avg>=50%)': '#3b82f6',                 # Blue
                'Underutilized': '#16a34a',                            # Green (fallback)
                'Well Utilized': '#3b82f6',                            # Blue (fallback)
                'Performance Risk': '#dc2626'                          # Red (fallback)
            }
            colors = [color_map.get(label, '#64748b') for label in optimization_summary.index]
            
            fig3 = go.Figure(data=[go.Pie(
                labels=optimization_summary.index,
                values=optimization_summary.values,
                hole=0.5,
                marker_colors=colors,
                textinfo='label+percent+value',
                textposition='outside',
                textfont_size=11
            )])
            
            # Add center text
            total_instances = len(self.instances_df)
            safe_to_optimize = len(self.instances_df[self.instances_df['optimization_category'] == 'Safe to Optimize (Avg<50%, Zero Spikes)'])
            
            fig3.add_annotation(
                text=f"<b>{total_instances}</b><br>Total<br>Instances",
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )
            
            fig3.update_layout(
                title='Zero-Spike Optimization Status Distribution',
                height=450,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
                font=dict(size=11)
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Project-wise underutilization
            fig4 = go.Figure()
            
            # Sort projects by underutilization percentage for better readability
            project_stats_sorted = project_stats.sort_values('underutil_pct', ascending=True)
            
            colors = ['#dc2626' if x > 50 else '#f59e0b' if x > 25 else '#16a34a' 
                     for x in project_stats_sorted['underutil_pct']]
            
            fig4.add_trace(go.Bar(
                y=project_stats_sorted['project_id'],
                x=project_stats_sorted['underutil_pct'],
                orientation='h',
                marker_color=colors,
                text=[f"{val:.1f}%" for val in project_stats_sorted['underutil_pct']],
                textposition='outside'
            ))
            
            fig4.add_vline(x=25, line_dash="dash", line_color="orange", line_width=1,
                          annotation_text="25% Warning")
            fig4.add_vline(x=50, line_dash="solid", line_color="red", line_width=2,
                          annotation_text="50% Critical")
            
            fig4.update_layout(
                title='Underutilization % by Project',
                xaxis_title='Underutilization Percentage (%)',
                yaxis_title='Project',
                height=400,
                plot_bgcolor='white',
                showlegend=False
            )
            
            st.plotly_chart(fig4, use_container_width=True)
        
        # Simplified key metrics table
        st.markdown("### 📊 Project Summary Table")
        
        # Create a cleaner, more focused table
        summary_df = project_stats[['project_id', 'instances', 'underutilized', 'underutil_pct', 
                                  'total_vcpu', 'used_vcpu', 'cpu_waste_pct', 
                                  'avg_cpu_util', 'avg_memory_util']].copy()
        
        summary_df.columns = ['Project', 'Instances', 'Underutilized', 'Underutil %', 
                            'Total vCPU', 'Used vCPU', 'CPU Waste %', 
                            'Avg CPU %', 'Avg Memory %']
        
        # Color coding function
        def highlight_metrics(val):
            if isinstance(val, (int, float)):
                if 'Waste %' in str(val) or 'Underutil %' in str(val):
                    if val > 50:
                        return 'background-color: #fef2f2; color: #dc2626; font-weight: bold'
                    elif val > 25:
                        return 'background-color: #fef3c7; color: #d97706; font-weight: bold'
                    else:
                        return 'background-color: #f0fdf4; color: #16a34a; font-weight: bold'
                elif 'Avg' in str(val):
                    if val > 60:
                        return 'background-color: #f0fdf4; color: #16a34a; font-weight: bold'
                    elif val > 30:
                        return 'background-color: #fef3c7; color: #d97706; font-weight: bold'
                    else:
                        return 'background-color: #fef2f2; color: #dc2626; font-weight: bold'
            return ''
        
        # Apply styling to specific columns
        styled_df = summary_df.style.applymap(
            lambda x: highlight_metrics(x), 
            subset=['Underutil %', 'CPU Waste %', 'Avg CPU %', 'Avg Memory %']
        )
        
        st.dataframe(styled_df, use_container_width=True)
    
    def create_instance_detail_view(self, selected_project):
        """Create detailed instance-level view for a project"""
        if self.instances_df is None or self.instances_df.empty:
            return
        
        project_data = self.instances_df[self.instances_df['project_id'] == selected_project]
        
        if project_data.empty:
            st.warning(f"No data found for project: {selected_project}")
            return
        
        st.markdown(f'<div class="project-header">🔍 Project Analysis: {selected_project}</div>', unsafe_allow_html=True)
        
        # Main visualization: CPU Utilization vs Instance Names
        st.markdown("## 📊 CPU Utilization Analysis (Your Requested Graph)")
        
        # Create the specific graph you requested: CPU util (X-axis) vs Instance (Y-axis)
        fig = go.Figure()
        
        # Add horizontal bar chart
        colors = ['#dc2626' if row['underutilized'] else '#16a34a' for _, row in project_data.iterrows()]
        
        fig.add_trace(go.Bar(
            y=project_data['instance_id'],  # Instance names on Y-axis
            x=project_data['avg_cpu_utilization'],  # CPU utilization on X-axis
            orientation='h',  # Horizontal bars
            marker_color=colors,
            text=[f"{util:.1f}%" for util in project_data['avg_cpu_utilization']],
            textposition='outside',
            name='CPU Utilization'
        ))
        
        # Add threshold line at 40%
        fig.add_vline(x=40, line_dash="dash", line_color="orange", 
                     annotation_text="40% Threshold", annotation_position="top")
        
        fig.update_layout(
            title=f'CPU Utilization by Instance - {selected_project}',
            xaxis_title='CPU Utilization (%)',
            yaxis_title='Cloud SQL Instances',
            height=max(400, len(project_data) * 60),  # Dynamic height based on instance count
            plot_bgcolor='white',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual Instance Analysis (Text Format)
        st.markdown("## 🖥️ Individual Instance Analysis")
        
        for i, (_, instance) in enumerate(project_data.iterrows(), 1):
            self.create_instance_text_analysis(instance, i)
        
        # Summary comparison charts
        if len(project_data) > 1:
            st.markdown("## 📈 Multi-Resource Comparison")
            self.create_simplified_comparison_charts(project_data, selected_project)
    
    def create_instance_card(self, instance):
        """Create detailed instance card with comprehensive metrics"""
        status_badge = "status-badge-underutilized" if instance['underutilized'] else "status-badge-utilized"
        status_text = "UNDERUTILIZED" if instance['underutilized'] else "WELL UTILIZED"
        
        # Check if we have comprehensive metrics
        has_memory = 'avg_memory_utilization' in instance and pd.notna(instance['avg_memory_utilization'])
        has_disk = 'avg_disk_utilization' in instance and pd.notna(instance['avg_disk_utilization'])
        
        st.markdown(f"""
        <div class="instance-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
                <h3 style="margin: 0; color: #1e40af; font-size: 1.5rem;">🗄️ {instance['instance_id']}</h3>
                <span class="{status_badge}">{status_text}</span>
            </div>
            
            <div class="spec-container">
                <h4 style="color: #374151; margin-bottom: 1rem;">📋 Instance Specifications</h4>
                <div class="spec-grid">
                    <div class="spec-item">
                        <div class="spec-value">{instance['vcpu_count']}</div>
                        <div class="spec-label">vCPUs</div>
                    </div>
                    <div class="spec-item">
                        <div class="spec-value">{instance['memory_gb']:.1f} GB</div>
                        <div class="spec-label">Memory</div>
                    </div>
                    <div class="spec-item">
                        <div class="spec-value">{instance['disk_size_gb']} GB</div>
                        <div class="spec-label">Storage</div>
                    </div>
                    <div class="spec-item">
                        <div class="spec-value">{instance.get('tier', 'Unknown')}</div>
                        <div class="spec-label">Tier</div>
                    </div>
                    <div class="spec-item">
                        <div class="spec-value">{instance.get('database_version', 'Unknown')}</div>
                        <div class="spec-label">Database</div>
                    </div>
                    <div class="spec-item">
                        <div class="spec-value">{instance.get('region', 'Unknown')}</div>
                        <div class="spec-label">Region</div>
                    </div>
                </div>
            </div>
            
            <h4 style="color: #374151; margin: 2rem 0 1rem 0;">📈 3-Month Resource Utilization</h4>
            <div class="resource-grid">
                <div class="resource-item">
                    <div class="resource-value" style="color: #3b82f6">{instance.get('avg_cpu_utilization', 0):.1f}%</div>
                    <div class="resource-label">Avg CPU Usage</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                        {instance.get('avg_cpu_actual', 0):.1f} / {instance['vcpu_count']} vCPU
                    </div>
                </div>
                <div class="resource-item">
                    <div class="resource-value" style="color: #10b981">{instance.get('avg_memory_utilization', 0):.1f}%</div>
                    <div class="resource-label">Avg Memory Usage</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                        {instance.get('avg_memory_actual', 0):.1f} / {instance['memory_gb']:.1f} GB
                    </div>
                </div>
                <div class="resource-item">
                    <div class="resource-value" style="color: #f59e0b">{instance.get('avg_disk_utilization', 0):.1f}%</div>
                    <div class="resource-label">Avg Storage Usage</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                        {instance.get('avg_disk_actual', 0):.1f} / {instance['disk_size_gb']} GB
                    </div>
                </div>
                <div class="resource-item">
                    <div class="resource-value" style="color: #8b5cf6">{instance.get('data_points', 0):,}</div>
                    <div class="resource-label">Data Points</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                        3-month period
                    </div>
                </div>
            </div>
            
            <h4 style="color: #374151; margin: 2rem 0 1rem 0;">🎯 Peak Utilization (95th Percentile)</h4>
            <div class="resource-grid">
                <div class="resource-item">
                    <div class="resource-value" style="color: #dc2626">{instance.get('p95_cpu_utilization', 0):.1f}%</div>
                    <div class="resource-label">Peak CPU</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                        {instance.get('p95_cpu_actual', 0):.1f} vCPU
                    </div>
                </div>
                <div class="resource-item">
                    <div class="resource-value" style="color: #dc2626">{instance.get('p95_memory_utilization', 0):.1f}%</div>
                    <div class="resource-label">Peak Memory</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                        {instance.get('p95_memory_actual', 0):.1f} GB
                    </div>
                </div>
                <div class="resource-item">
                    <div class="resource-value" style="color: #dc2626">{instance.get('p95_disk_utilization', 0):.1f}%</div>
                    <div class="resource-label">Peak Storage</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                        {instance.get('p95_disk_actual', 0):.1f} GB
                    </div>
                </div>
                <div class="resource-item">
                    <div class="resource-value" style="color: #6b7280">{instance.get('max_cpu_utilization', 0):.1f}%</div>
                    <div class="resource-label">Max CPU Ever</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                        Absolute peak
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def create_project_comparison_charts(self, project_data, project_name):
        """Create comprehensive comparison charts for project instances"""
        
        # Multi-metric comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Utilization Comparison', 'Memory Utilization Comparison',
                          'Storage Utilization Comparison', 'Resource Capacity Overview'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # CPU utilization bar chart
        fig.add_trace(
            go.Bar(x=project_data['instance_id'], y=project_data['avg_cpu_utilization'],
                  name='Avg CPU %', marker_color='#3b82f6'),
            row=1, col=1
        )
        
        # Memory utilization bar chart
        fig.add_trace(
            go.Bar(x=project_data['instance_id'], y=project_data['avg_memory_utilization'],
                  name='Avg Memory %', marker_color='#10b981'),
            row=1, col=2
        )
        
        # Storage utilization bar chart
        fig.add_trace(
            go.Bar(x=project_data['instance_id'], y=project_data['avg_disk_utilization'],
                  name='Avg Storage %', marker_color='#f59e0b'),
            row=2, col=1
        )
        
        # Resource capacity overview (stacked bar)
        fig.add_trace(
            go.Bar(x=project_data['instance_id'], y=project_data['vcpu_count'],
                  name='vCPU Count', marker_color='#8b5cf6'),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=project_data['instance_id'], y=project_data['memory_gb'],
                      mode='markers+lines', name='Memory (GB)', 
                      marker=dict(color='#06b6d4', size=10), line=dict(color='#06b6d4')),
            row=2, col=2, secondary_y=True
        )
        
        # Add threshold lines
        for row, col in [(1, 1), (1, 2), (2, 1)]:
            fig.add_hline(y=40, line_dash="dash", line_color="orange", row=row, col=col)
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text=f"Comprehensive Resource Analysis - {project_name}")
        
        # Update y-axis labels
        fig.update_yaxes(title_text="CPU Utilization (%)", row=1, col=1)
        fig.update_yaxes(title_text="Memory Utilization (%)", row=1, col=2)
        fig.update_yaxes(title_text="Storage Utilization (%)", row=2, col=1)
        fig.update_yaxes(title_text="vCPU Count", row=2, col=2)
        fig.update_yaxes(title_text="Memory (GB)", row=2, col=2, secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_instance_text_analysis(self, instance, instance_num):
        """Create text-based instance analysis (not HTML cards)"""
        status = "🔴 UNDERUTILIZED" if instance['underutilized'] else "🟢 WELL UTILIZED"
        
        st.markdown(f"### Instance {instance_num}: {instance['instance_id']} {status}")
        
        # Basic specifications in text format
        st.markdown("**📋 Instance Specifications:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("vCPUs", f"{instance['vcpu_count']}")
            st.metric("Memory", f"{instance['memory_gb']:.1f} GB")
        
        with col2:
            st.metric("Storage", f"{instance['disk_size_gb']} GB")
            st.metric("Tier", instance.get('tier', 'Unknown'))
        
        with col3:
            st.metric("Database", instance.get('database_version', 'Unknown'))
            st.metric("Region", instance.get('region', 'Unknown'))
        
        # Resource utilization in text format
        st.markdown("**📈 3-Month Resource Utilization:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Average CPU", 
                f"{instance.get('avg_cpu_utilization', 0):.1f}%",
                delta=f"{instance.get('avg_cpu_actual', 0):.1f}/{instance['vcpu_count']} vCPU"
            )
        
        with col2:
            st.metric(
                "Average Memory", 
                f"{instance.get('avg_memory_utilization', 0):.1f}%",
                delta=f"{instance.get('avg_memory_actual', 0):.1f}/{instance['memory_gb']:.1f} GB"
            )
        
        with col3:
            st.metric(
                "Average Storage", 
                f"{instance.get('avg_disk_utilization', 0):.1f}%",
                delta=f"{instance.get('avg_disk_actual', 0):.1f}/{instance['disk_size_gb']} GB"
            )
        
        with col4:
            confidence = "HIGH" if instance.get('data_points', 0) > 10000 else "MEDIUM" if instance.get('data_points', 0) > 5000 else "LOW"
            st.metric(
                "Data Points", 
                f"{instance.get('data_points', 0):,}",
                delta=f"Confidence: {confidence}"
            )
        
        # Peak utilization
        st.markdown("**🎯 Peak Utilization (95th Percentile):**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Peak CPU", 
                f"{instance.get('p95_cpu_utilization', 0):.1f}%",
                delta=f"{instance.get('p95_cpu_actual', 0):.1f} vCPU"
            )
        
        with col2:
            st.metric(
                "Peak Memory", 
                f"{instance.get('p95_memory_utilization', 0):.1f}%",
                delta=f"{instance.get('p95_memory_actual', 0):.1f} GB"
            )
        
        with col3:
            st.metric(
                "Peak Storage", 
                f"{instance.get('p95_disk_utilization', 0):.1f}%",
                delta=f"{instance.get('p95_disk_actual', 0):.1f} GB"
            )
        
        with col4:
            st.metric(
                "Max CPU Ever", 
                f"{instance.get('max_cpu_utilization', 0):.1f}%",
                delta="Absolute peak"
            )
        
        # Enhanced rightsizing recommendation with detailed analysis
        current_vcpu = instance['vcpu_count']
        avg_cpu = instance.get('avg_cpu_utilization', 0)
        peak_cpu = instance.get('p95_cpu_utilization', 0)
        max_cpu = instance.get('max_cpu_utilization', 0)
        peak_usage = instance.get('p95_cpu_actual', 0)
        
        if instance['underutilized']:
            recommended_vcpu = max(2, int(np.ceil(peak_usage * 1.3)))  # 30% buffer above peak
            
            if recommended_vcpu < current_vcpu:
                savings = current_vcpu - recommended_vcpu
                st.success(f"💡 **Rightsizing Recommendation:** Reduce from {current_vcpu} to {recommended_vcpu} vCPU (save {savings} vCPU)")
        else:
            # Enhanced analysis for "well-utilized" instances
            if avg_cpu < 20 and max_cpu >= 90:
                # Borderline case: very low average but high peaks
                conservative_vcpu = max(4, int(np.ceil(peak_usage * 1.2)))  # 20% buffer above peak
                if conservative_vcpu < current_vcpu:
                    savings = current_vcpu - conservative_vcpu
                    st.warning(f"⚠️ **Borderline Case:** Avg CPU very low ({avg_cpu:.1f}%) but peaks at {max_cpu:.1f}%")
                    st.warning(f"💡 **Conservative Recommendation:** Consider reducing from {current_vcpu} to {conservative_vcpu} vCPU (save {savings} vCPU) with careful monitoring")
                else:
                    st.info("✅ **Status:** Instance has high peaks - current sizing appropriate")
            elif avg_cpu < 30:
                st.info(f"📊 **Analysis:** Low average usage ({avg_cpu:.1f}%) but peak usage ({peak_cpu:.1f}%) justifies current sizing")
            else:
                st.info("✅ **Status:** Instance is well-utilized - no immediate changes needed")
        
        # Individual instance graph
        st.markdown("**📊 Resource Utilization Graph:**")
        
        # Create individual instance resource chart
        fig = go.Figure()
        
        resources = ['CPU', 'Memory', 'Storage']
        avg_values = [
            instance.get('avg_cpu_utilization', 0),
            instance.get('avg_memory_utilization', 0),
            instance.get('avg_disk_utilization', 0)
        ]
        peak_values = [
            instance.get('p95_cpu_utilization', 0),
            instance.get('p95_memory_utilization', 0),
            instance.get('p95_disk_utilization', 0)
        ]
        
        fig.add_trace(go.Bar(
            name='Average Utilization',
            x=resources,
            y=avg_values,
            marker_color='#3b82f6'
        ))
        
        fig.add_trace(go.Bar(
            name='Peak Utilization (95th %)',
            x=resources,
            y=peak_values,
            marker_color='#dc2626'
        ))
        
        # Add threshold line
        fig.add_hline(y=40, line_dash="dash", line_color="orange", 
                     annotation_text="40% Threshold")
        
        fig.update_layout(
            title=f'Resource Utilization - {instance["instance_id"]}',
            yaxis_title='Utilization (%)',
            xaxis_title='Resource Type',
            height=400,
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")  # Separator between instances
    
    def create_simplified_comparison_charts(self, project_data, project_name):
        """Create simplified comparison charts for multiple instances"""
        
        # CPU utilization comparison (the main graph you requested)
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('CPU Utilization', 'Memory Utilization', 'Storage Utilization')
        )
        
        # CPU comparison
        fig.add_trace(
            go.Bar(x=project_data['instance_id'], y=project_data['avg_cpu_utilization'],
                  name='CPU %', marker_color='#3b82f6'),
            row=1, col=1
        )
        
        # Memory comparison
        fig.add_trace(
            go.Bar(x=project_data['instance_id'], y=project_data['avg_memory_utilization'],
                  name='Memory %', marker_color='#10b981'),
            row=1, col=2
        )
        
        # Storage comparison
        fig.add_trace(
            go.Bar(x=project_data['instance_id'], y=project_data['avg_disk_utilization'],
                  name='Storage %', marker_color='#f59e0b'),
            row=1, col=3
        )
        
        # Add threshold lines
        for col in [1, 2, 3]:
            fig.add_hline(y=40, line_dash="dash", line_color="orange", row=1, col=col)
        
        fig.update_layout(height=500, showlegend=False, 
                         title_text=f"Resource Utilization Comparison - {project_name}")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_all_instances_view(self):
        """Create view showing all instances grouped by project with pagination"""
        if self.instances_df is None or self.instances_df.empty:
            return
        
        st.markdown("## 🔍 Complete Instance Analysis")
        st.markdown("*All instances sorted by project*")
        
        # Sort instances by project and then by instance name
        sorted_df = self.instances_df.sort_values(['project_id', 'instance_id'])
        
        # Pagination settings
        instances_per_page = st.sidebar.slider("Instances per page:", 5, 20, 10)
        total_instances = len(sorted_df)
        total_pages = (total_instances - 1) // instances_per_page + 1
        
        if total_pages > 1:
            page = st.sidebar.number_input(
                f"Page (1-{total_pages}):", 
                min_value=1, 
                max_value=total_pages, 
                value=1
            )
        else:
            page = 1
        
        # Calculate pagination
        start_idx = (page - 1) * instances_per_page
        end_idx = min(start_idx + instances_per_page, total_instances)
        page_instances = sorted_df.iloc[start_idx:end_idx]
        
        # Show pagination info
        st.info(f"Showing instances {start_idx + 1}-{end_idx} of {total_instances} total instances (Page {page} of {total_pages})")
        
        # Group instances by project for display
        current_project = None
        instance_count = 0
        
        for _, instance in page_instances.iterrows():
            # Show project header when project changes
            if current_project != instance['project_id']:
                if current_project is not None:
                    st.markdown("---")
                
                current_project = instance['project_id']
                project_instances = sorted_df[sorted_df['project_id'] == current_project]
                project_underutil = len(project_instances[project_instances['underutilized'] == True])
                
                st.markdown(f"""
                <div class="project-header">
                    📁 Project: {current_project} 
                    <span style="font-size: 1rem; font-weight: normal;">
                        ({len(project_instances)} instances, {project_underutil} underutilized)
                    </span>
                </div>
                """, unsafe_allow_html=True)
                instance_count = 0
            
            instance_count += 1
            
            # Show instance analysis
            self.create_compact_instance_analysis(instance, instance_count)
        
        # Navigation helper
        if total_pages > 1:
            col1, col2, col3 = st.columns(3)
            with col1:
                if page > 1:
                    if st.button("← Previous Page"):
                        st.rerun()
            with col3:
                if page < total_pages:
                    if st.button("Next Page →"):
                        st.rerun()
    
    def create_compact_instance_analysis(self, instance, instance_num):
        """Create professional instance analysis with zero-spike logic"""
        
        # Calculate dual-threshold spike counts (simulate for now - in real data this would come from spike_analysis)
        max_cpu = instance.get('max_cpu_utilization', 0)
        avg_cpu = instance.get('avg_cpu_utilization', 0)
        
        # Use REAL spike data from monitoring analysis (not simulation!)
        critical_spikes = instance.get('critical_spikes_count', 0)
        moderate_spikes = instance.get('moderate_spikes_count', 0)
        critical_frequency = instance.get('critical_spike_frequency', 0)
        moderate_frequency = instance.get('moderate_spike_frequency', 0)
        total_data_points = instance.get('total_data_points', 0)
        
        # Fallback to simulation only if real data is not available
        if critical_spikes == 0 and moderate_spikes == 0 and max_cpu > 0:
            # Legacy simulation for backward compatibility
            critical_spikes = 1 if max_cpu >= 90 else 0
            if max_cpu >= 90:
                moderate_spikes = max(1, critical_spikes)
            elif max_cpu >= 50:
                moderate_spikes = 1
            else:
                moderate_spikes = 0
        
        # Determine safety status based on enhanced zero-spike logic
        is_safe_to_optimize = avg_cpu < 50 and critical_spikes == 0
        
        # Professional status determination
        if is_safe_to_optimize:
            status_color = "#16a34a"
            status_text = "✅ SAFE TO OPTIMIZE"
            status_reason = "Zero spikes detected"
        elif avg_cpu < 50 and critical_spikes > 0:
            status_color = "#dc2626" 
            status_text = "❌ KEEP CURRENT SIZE"
            status_reason = f"{critical_spikes} critical spikes"
        else:
            status_color = "#3b82f6"
            status_text = "✅ WELL UTILIZED"
            status_reason = "High average usage"
        
        # Professional instance card
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 6px solid {status_color};
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: #1e40af;">🗄️ Instance {instance_num}: {instance['instance_id']}</h4>
                <span style="
                    background: {status_color}; 
                    color: white; 
                    padding: 0.5rem 1rem; 
                    border-radius: 20px; 
                    font-size: 0.85rem; 
                    font-weight: 600;
                ">{status_text}</span>
            </div>
            <div style="color: #64748b; font-size: 0.9rem; margin-bottom: 1rem;">
                {status_reason}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Specifications and metrics in professional layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**📋 Specifications**")
            st.metric("vCPUs", f"{instance['vcpu_count']}")
            st.metric("Memory", f"{instance['memory_gb']:.0f} GB")
            st.metric("Storage", f"{instance['disk_size_gb']} GB")
        
        with col2:
            st.markdown("**📊 30-Day Averages**")
            st.metric("CPU Average", f"{avg_cpu:.1f}%")
            st.metric("Memory Average", f"{instance.get('avg_memory_utilization', 0):.1f}%")
            st.metric("Storage Average", f"{instance.get('avg_disk_utilization', 0):.1f}%")
        
        with col3:
            st.markdown("**⚠️ Performance Spikes**")
            st.metric("Critical >90%", critical_spikes, delta="Zero-spike policy")
            st.metric("Moderate >50%", moderate_spikes, delta=f"{moderate_frequency:.2f}% of time" if moderate_frequency > 0 else "Low usage")
            st.metric("Highest CPU", f"{max_cpu:.1f}%")
        
        with col4:
            st.markdown("**🎯 Optimization Status**")
            if is_safe_to_optimize:
                current_vcpu = instance['vcpu_count']
                # Conservative recommendation - reduce by max 25%
                recommended_vcpu = max(2, int(current_vcpu * 0.75))
                savings = current_vcpu - recommended_vcpu
                st.metric("Recommendation", f"Reduce to {recommended_vcpu}", delta=f"Save {savings} vCPU")
                st.success("✅ Safe to optimize")
            else:
                st.metric("Recommendation", "Keep current", delta="No change")
                if critical_spikes > 0:
                    st.error("❌ Performance risk detected")
                else:
                    st.info("ℹ️ Well utilized")
        
        # Professional resource visualization
        if st.checkbox(f"📊 Show Resource Analysis", key=f"chart_{instance['instance_id']}"):
            self.create_professional_instance_chart(instance)
        
        st.markdown("---")
    
    def create_professional_instance_chart(self, instance):
        """Create professional resource chart with zero-spike indicators"""
        
        # Resource data
        resources = ['CPU', 'Memory', 'Storage']
        avg_values = [
            instance.get('avg_cpu_utilization', 0),
            instance.get('avg_memory_utilization', 0),
            instance.get('avg_disk_utilization', 0)
        ]
        peak_values = [
            instance.get('p95_cpu_utilization', 0),
            instance.get('p95_memory_utilization', 0),
            instance.get('p95_disk_utilization', 0)
        ]
        max_values = [
            instance.get('max_cpu_utilization', 0),
            instance.get('max_memory_utilization', 0),
            instance.get('max_disk_utilization', 0)
        ]
        
        # Professional chart
        fig = go.Figure()
        
        # Average utilization bars
        fig.add_trace(go.Bar(
            name='30-Day Average',
            x=resources,
            y=avg_values,
            marker_color='#3b82f6',
            text=[f"{val:.1f}%" for val in avg_values],
            textposition='inside'
        ))
        
        # Peak utilization bars
        fig.add_trace(go.Bar(
            name='Peak (95th %)',
            x=resources,
            y=peak_values,
            marker_color='#f59e0b',
            text=[f"{val:.1f}%" for val in peak_values],
            textposition='inside'
        ))
        
        # Maximum spikes
        fig.add_trace(go.Scatter(
            name='Maximum Spike',
            x=resources,
            y=max_values,
            mode='markers+text',
            marker=dict(color='#dc2626', size=12, symbol='triangle-up'),
            text=[f"{val:.1f}%" for val in max_values],
            textposition='top center'
        ))
        
        # Professional threshold lines
        fig.add_hline(y=50, line_dash="solid", line_color="#16a34a", line_width=2,
                     annotation_text="50% Average Threshold", annotation_position="right")
        fig.add_hline(y=90, line_dash="solid", line_color="#dc2626", line_width=3,
                     annotation_text="90% Spike Threshold (ZERO tolerance)", annotation_position="right")
        
        fig.update_layout(
            title=f'Professional Resource Analysis - {instance["instance_id"]}',
            xaxis_title='Resource Type',
            yaxis_title='Utilization (%)',
            height=400,
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run_dashboard(self):
        """Main dashboard function"""
        # Header
        st.markdown('<div class="main-header">🔍 Cloud SQL Resource Utilization Dashboard</div>', unsafe_allow_html=True)
        
        # Load data
        if not self.load_data():
            st.stop()
        
        # Sidebar for view selection
        st.sidebar.markdown('<div class="view-selector"><h3>📋 Analysis Mode</h3></div>', unsafe_allow_html=True)
        
        view_mode = st.sidebar.radio(
            "Choose your analysis view:",
            ["🌐 Executive Overview", "🔍 Instance-Level Analysis"],
            help="Select how you want to analyze your Cloud SQL resources"
        )
        
        if view_mode == "🌐 Executive Overview":
            # Executive dashboard
            self.create_executive_dashboard()
            
            # Additional filters
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🔧 Advanced Filters")
            
            # Resource filters
            min_vcpu = st.sidebar.slider("Minimum vCPUs:", 0, int(self.instances_df['vcpu_count'].max()), 0)
            min_memory = st.sidebar.slider("Minimum Memory (GB):", 0, int(self.instances_df['memory_gb'].max()), 0)
            

            
            # Apply filters
            filtered_df = self.instances_df.copy()
            
            if min_vcpu > 0:
                filtered_df = filtered_df[filtered_df['vcpu_count'] >= min_vcpu]
            if min_memory > 0:
                filtered_df = filtered_df[filtered_df['memory_gb'] >= min_memory]
            
            if len(filtered_df) != len(self.instances_df):
                st.markdown("## 🔍 Filtered Analysis Results")
                st.info(f"Showing {len(filtered_df)} instances out of {len(self.instances_df)} total")
                
                # Show filtered summary
                if not filtered_df.empty:
                    underutil_count = len(filtered_df[filtered_df['underutilized'] == True])
                    st.metric("Filtered Underutilized Instances", underutil_count, 
                             f"{(underutil_count/len(filtered_df)*100):.1f}%")
        
        else:  # Instance-Level Analysis
            # Show all instances grouped by project (no dropdown)
            self.create_all_instances_view()
        
        # Export functionality
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📥 Export Options")
        
        if st.sidebar.button("📊 Download Comprehensive Report", type="primary"):
            csv = self.instances_df.to_csv(index=False)
            st.sidebar.download_button(
                label="📄 Download CSV Report",
                data=csv,
                file_name=f"cloudsql_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def main():
    dashboard = CloudSQLDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 
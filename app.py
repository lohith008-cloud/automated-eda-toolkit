# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import traceback
import json
import io
import base64
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="End-to-End Data Analysis and Machine Learning Platform ✅",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Cards */
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #718096;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f7fafc;
        border-radius: 8px 8px 0 0;
        padding: 1rem 2rem;
        border: 1px solid #e2e8f0;
        border-bottom: none;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #667eea !important;
        border-color: #e2e8f0 !important;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
    }
    
    /* Data table */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'data_profile' not in st.session_state:
    st.session_state.data_profile = {}
if 'current_page' not in st.session_state:
    st.session_state.current_page = "upload"

# ============================================
# DATA PROFILER
# ============================================
class AIDataProfiler:
    """Advanced data profiler"""
    
    @staticmethod
    def comprehensive_quality_scan(df: pd.DataFrame) -> Dict:
        """Comprehensive quality scan"""
        results = {
            'overall_score': 0,
            'dimensions': {},
            'recommendations': [],
            'issues': []
        }
        
        try:
            # Completeness analysis
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isna().sum().sum()
            missing_percentage = (missing_cells / total_cells * 100) if total_cells > 0 else 0
            
            completeness_score = max(0, 100 - min(missing_percentage * 1.5, 100))
            results['dimensions']['completeness'] = {
                'score': completeness_score,
                'missing_percentage': missing_percentage,
                'missing_cells': int(missing_cells),
                'empty_columns': df.columns[df.isna().all()].tolist()
            }
            
            # Uniqueness analysis
            duplicate_rows = df.duplicated().sum()
            duplicate_percentage = (duplicate_rows / len(df) * 100) if len(df) > 0 else 0
            uniqueness_score = max(0, 100 - min(duplicate_percentage * 2, 100))
            
            results['dimensions']['uniqueness'] = {
                'score': uniqueness_score,
                'duplicate_percentage': duplicate_percentage,
                'duplicate_rows': int(duplicate_rows),
                'constant_columns': [col for col in df.columns if df[col].nunique() == 1]
            }
            
            # Validity analysis
            validity_score = 100
            validity_issues = []
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check for mixed types
                    sample = df[col].dropna().head(50)
                    if len(sample) > 0:
                        types = set()
                        for val in sample:
                            if isinstance(val, (int, np.integer)):
                                types.add('integer')
                            elif isinstance(val, (float, np.floating)):
                                types.add('float')
                            elif isinstance(val, str):
                                types.add('string')
                        
                        if len(types) > 1:
                            validity_score -= 5
                            validity_issues.append(f"Mixed types in column '{col}'")
            
            results['dimensions']['validity'] = {
                'score': max(0, validity_score),
                'issues': validity_issues
            }
            
            # Calculate overall score
            weights = {'completeness': 0.4, 'uniqueness': 0.3, 'validity': 0.3}
            overall = sum(results['dimensions'][dim]['score'] * weights[dim] for dim in weights)
            results['overall_score'] = round(overall, 1)
            
            # Generate recommendations
            if missing_percentage > 10:
                results['recommendations'].append({
                    'type': 'warning',
                    'title': 'High Missing Data',
                    'message': f"{missing_percentage:.1f}% of cells are missing",
                    'action': 'Consider imputation or removal'
                })
            
            if duplicate_percentage > 5:
                results['recommendations'].append({
                    'type': 'warning',
                    'title': 'Duplicate Records',
                    'message': f"{duplicate_rows} duplicate rows found",
                    'action': 'Remove duplicates'
                })
            
            if len(results['dimensions']['uniqueness']['constant_columns']) > 0:
                results['recommendations'].append({
                    'type': 'info',
                    'title': 'Constant Columns',
                    'message': f"{len(results['dimensions']['uniqueness']['constant_columns'])} columns have constant values",
                    'action': 'Consider removing constant columns'
                })
            
        except Exception as e:
            results['issues'].append(f"Profiling error: {str(e)}")
        
        return results
    
    @staticmethod
    def generate_statistical_profile(df: pd.DataFrame) -> Dict:
        """Generate statistical profile"""
        profile = {
            'dataset_summary': {
                'rows': len(df),
                'columns': len(df.columns),
                'total_cells': len(df) * len(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            'data_type_analysis': {},
            'column_statistics': {},
            'correlation_insights': {}
        }
        
        try:
            # Data type analysis
            dtype_counts = df.dtypes.value_counts()
            profile['data_type_analysis'] = {
                'numeric': len(df.select_dtypes(include=[np.number]).columns),
                'categorical': len(df.select_dtypes(include=['object', 'category']).columns),
                'datetime': len(df.select_dtypes(include=['datetime64']).columns),
                'boolean': len(df.select_dtypes(include=['bool']).columns)
            }
            
            # Column statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if len(df[col].dropna()) > 0:
                    profile['column_statistics'][col] = {
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'missing': int(df[col].isna().sum()),
                        'missing_percentage': float(df[col].isna().mean() * 100)
                    }
            
            # Correlation insights
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr().abs()
                strong_correlations = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr = corr_matrix.iloc[i, j]
                        if corr > 0.7:
                            strong_correlations.append({
                                'columns': [corr_matrix.columns[i], corr_matrix.columns[j]],
                                'correlation': float(corr),
                                'type': 'strong_positive'
                            })
                
                profile['correlation_insights'] = {
                    'strong_correlations': strong_correlations[:10],  # Limit to top 10
                    'total_strong': len(strong_correlations)
                }
                
        except Exception as e:
            profile['error'] = str(e)
        
        return profile

# ============================================
# DATA CLEANER
# ============================================
class AIDataCleaner:
    """Intelligent data cleaner"""
    
    @staticmethod
    def clean_dataset(df: pd.DataFrame, operations: List[str] = None) -> pd.DataFrame:
        """Clean dataset with specified operations"""
        if operations is None:
            operations = ['remove_duplicates', 'handle_missing', 'optimize_types']
        
        cleaned_df = df.copy()
        report = {
            'operations_performed': [],
            'metrics': {}
        }
        
        try:
            # Remove duplicates
            if 'remove_duplicates' in operations:
                original_rows = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                removed = original_rows - len(cleaned_df)
                report['operations_performed'].append(f"Removed {removed} duplicate rows")
                report['metrics']['duplicates_removed'] = removed
            
            # Handle missing values
            if 'handle_missing' in operations:
                missing_before = cleaned_df.isna().sum().sum()
                
                for col in cleaned_df.columns:
                    if cleaned_df[col].dtype == 'object':
                        cleaned_df[col] = cleaned_df[col].fillna('Unknown')
                    elif pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        if cleaned_df[col].nunique() > 0:
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                        else:
                            cleaned_df[col] = cleaned_df[col].fillna(0)
                
                missing_after = cleaned_df.isna().sum().sum()
                report['operations_performed'].append(f"Handled {missing_before - missing_after} missing values")
                report['metrics']['missing_handled'] = missing_before - missing_after
            
            # Optimize data types
            if 'optimize_types' in operations:
                conversions = []
                for col in cleaned_df.columns:
                    if cleaned_df[col].dtype == 'object':
                        # Try to convert to datetime
                        try:
                            pd.to_datetime(cleaned_df[col], errors='raise')
                            cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                            conversions.append(f"{col}: object → datetime")
                        except:
                            # Try to convert to numeric
                            try:
                                numeric_series = pd.to_numeric(cleaned_df[col], errors='coerce')
                                if numeric_series.notna().mean() > 0.8:  # 80% success rate
                                    cleaned_df[col] = numeric_series
                                    conversions.append(f"{col}: object → numeric")
                            except:
                                pass
                
                if conversions:
                    report['operations_performed'].append(f"Optimized {len(conversions)} columns")
                    report['metrics']['conversions'] = conversions
            
            # Remove constant columns
            if 'remove_constant' in operations:
                constant_cols = [col for col in cleaned_df.columns if cleaned_df[col].nunique() == 1]
                if constant_cols:
                    cleaned_df = cleaned_df.drop(columns=constant_cols)
                    report['operations_performed'].append(f"Removed {len(constant_cols)} constant columns")
                    report['metrics']['constant_columns_removed'] = len(constant_cols)
            
            report['metrics']['final_shape'] = cleaned_df.shape
            report['metrics']['memory_reduction'] = f"{((df.memory_usage(deep=True).sum() - cleaned_df.memory_usage(deep=True).sum()) / df.memory_usage(deep=True).sum() * 100):.1f}%" if df.memory_usage(deep=True).sum() > 0 else "0%"
            
        except Exception as e:
            report['error'] = str(e)
        
        return cleaned_df, report

# ============================================
# VISUALIZATION ENGINE
# ============================================
class VisualizationEngine:
    """Interactive visualization engine"""
    
    @staticmethod
    def create_dashboard(df: pd.DataFrame, chart_types: List[str] = None):
        """Create interactive dashboard"""
        if chart_types is None:
            chart_types = ['distribution', 'correlation', 'scatter']
        
        dashboard = {}
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Distribution plots
            if 'distribution' in chart_types and numeric_cols:
                for col in numeric_cols[:3]:  # Limit to 3 columns
                    fig = px.histogram(df, x=col, nbins=50,
                                     title=f'Distribution of {col}',
                                     template='plotly_white',
                                     color_discrete_sequence=['#667eea'])
                    dashboard[f'distribution_{col}'] = fig
            
            # Correlation heatmap
            if 'correlation' in chart_types and len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix,
                              text_auto='.2f',
                              color_continuous_scale='RdBu',
                              title='Correlation Heatmap',
                              template='plotly_white',
                              width=800, height=600)
                dashboard['correlation_heatmap'] = fig
            
            # Scatter plots
            if 'scatter' in chart_types and len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                               title=f'{numeric_cols[1]} vs {numeric_cols[0]}',
                               template='plotly_white',
                               color_discrete_sequence=['#764ba2'])
                dashboard['scatter_plot'] = fig
            
            # Categorical distribution
            if 'categorical' in chart_types and categorical_cols:
                for col in categorical_cols[:2]:  # Limit to 2 columns
                    value_counts = df[col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f'Top 10 Values in {col}',
                               template='plotly_white',
                               color_discrete_sequence=['#f093fb'])
                    dashboard[f'categorical_{col}'] = fig
            
            # Box plots
            if 'box' in chart_types and numeric_cols:
                for col in numeric_cols[:2]:
                    fig = px.box(df, y=col,
                               title=f'Box Plot of {col}',
                               template='plotly_white')
                    dashboard[f'box_{col}'] = fig
            
            # Time series (if datetime columns exist)
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if 'timeseries' in chart_types and datetime_cols and numeric_cols:
                for date_col in datetime_cols[:1]:
                    for num_col in numeric_cols[:1]:
                        time_df = df[[date_col, num_col]].dropna()
                        if len(time_df) > 1:
                            fig = px.line(time_df, x=date_col, y=num_col,
                                        title=f'{num_col} over Time',
                                        template='plotly_white')
                            dashboard[f'timeseries_{num_col}'] = fig
            
        except Exception as e:
            dashboard['error'] = str(e)
        
        return dashboard

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="header-title">End-to-End Data Analysis and Machine Learning Platform ✅</h1>
        <p class="header-subtitle">Data Analysis and Predictive Modeling Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        
        # Page selection
        page_options = {
            "📁 Data Upload": "upload",
            "🔍 Data Profiling": "profiling",
            "🧹 Data Cleaning": "cleaning",
            "📈 Visualizations": "visualizations",
            "🤖 ML Analysis": "ml",
            "📋 Export": "export"
        }
        
        selected_page = st.radio(
            "Select Page:",
            list(page_options.keys()),
            key="page_selector"
        )
        st.session_state.current_page = page_options[selected_page]
        
        st.markdown("---")
        
        # Dataset info
        if st.session_state.df is not None:
            st.markdown("### 📊 Dataset Info")
            df = st.session_state.df
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Columns", df.shape[1])
            
            if 'data_profile' in st.session_state and st.session_state.data_profile:
                quality_score = st.session_state.data_profile.get('overall_score', 0)
                st.metric("Quality Score", f"{quality_score}%")
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            if st.button("🔄 Reset Dataset", use_container_width=True):
                if st.session_state.df_original is not None:
                    st.session_state.df = st.session_state.df_original.copy()
                    st.session_state.data_profile = {}
                    st.rerun()
            
            if st.button("🔍 Auto Profile", use_container_width=True):
                st.session_state.current_page = "profiling"
                st.rerun()
    
    # Main content area
    if st.session_state.current_page == "upload":
        render_upload_page()
    elif st.session_state.current_page == "profiling":
        render_profiling_page()
    elif st.session_state.current_page == "cleaning":
        render_cleaning_page()
    elif st.session_state.current_page == "visualizations":
        render_visualizations_page()
    elif st.session_state.current_page == "ml":
        render_ml_page()
    elif st.session_state.current_page == "export":
        render_export_page()

# ============================================
# PAGE FUNCTIONS
# ============================================

def render_upload_page():
    """Render upload page"""
    st.markdown("""
    <div style="background: white; border-radius: 15px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h2 style="color: #2d3748; margin-bottom: 1rem;">📁 Data Upload & Auto-Clean</h2>
        <p style="color: #718096; font-size: 1.1rem;">
            Upload your dataset and experience AI-powered data cleaning and analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #667eea;">🤖 AI-Powered Cleaning</h4>
            <p style="color: #718096;">Automatically detect and fix data quality issues with advanced ML algorithms.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #764ba2;">📊 Advanced Profiling</h4>
            <p style="color: #718096;">Comprehensive data analysis with statistical insights and quality scoring.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #f093fb;">📈 Interactive Visualizations</h4>
            <p style="color: #718096;">Create stunning, interactive charts and dashboards with one click.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Upload section
    st.markdown("### Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Supported formats: CSV, Excel, JSON, Parquet",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            file_extension = uploaded_file.name.split('.')[-1].lower()

            with st.spinner(f"Loading {uploaded_file.name}..."):
                if file_extension == 'csv':
                    uploaded_file.seek(0)
                    total_rows = sum(1 for _ in uploaded_file)

                    uploaded_file.seek(0)
                    df = pd.read_csv(
                        uploaded_file,
                        engine="python",
                        on_bad_lines="skip",
                        encoding_errors="ignore"
                    )

                    skipped_rows = total_rows - len(df)
                    if skipped_rows > 0:
                        st.warning(f"⚠️ {skipped_rows} malformed rows were skipped while loading the CSV.")

                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)

                elif file_extension == 'json':
                    df = pd.read_json(uploaded_file)

                elif file_extension == 'parquet':
                    df = pd.read_parquet(uploaded_file)

                else:
                    st.error("❌ Unsupported file format")
                    return
            
            # Store data
            st.session_state.df_original = df.copy()
            st.session_state.df = df.copy()
            
            # Run profiling
            profiler = AIDataProfiler()
            quality_scan = profiler.comprehensive_quality_scan(df)
            statistical_profile = profiler.generate_statistical_profile(df)
            
            st.session_state.data_profile = {
                'quality_scan': quality_scan,
                'statistical_profile': statistical_profile,
                'overall_score': quality_scan['overall_score']
            }
            
            # Success message
            st.success(f"""
            ✅ **{uploaded_file.name}** loaded successfully!
            
            **Dataset Overview:**
            - **{df.shape[0]:,}** rows × **{df.shape[1]}** columns
            - **{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB** memory usage
            - **{quality_scan['overall_score']}%** initial quality score
            """)
            
            # Quick metrics
            st.markdown("### 📊 Quick Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Rows</div>
                    <div class="metric-value">{df.shape[0]:,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Columns</div>
                    <div class="metric-value">{df.shape[1]}</div>
                </div>with
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Missing Values</div>
                    <div class="metric-value">{df.isna().sum().sum():,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                quality_score = quality_scan['overall_score']
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Quality Score</div>
                    <div class="metric-value">{quality_score}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Data preview
            with st.expander("📋 Data Preview (First 10 rows)", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Auto-clean option
            st.markdown("---")
            st.markdown("### 🤖 AI-Powered Auto-Clean")
            
            if st.button("🚀 Run AI Auto-Clean", type="primary", use_container_width=True):
                cleaner = AIDataCleaner()
                cleaned_df, report = cleaner.clean_dataset(
                    df,
                    operations=['remove_duplicates', 'handle_missing', 'optimize_types', 'remove_constant']
                )
                
                # Update session state
                st.session_state.df = cleaned_df
                
                # Update profiling
                quality_scan = profiler.comprehensive_quality_scan(cleaned_df)
                statistical_profile = profiler.generate_statistical_profile(cleaned_df)
                
                st.session_state.data_profile = {
                    'quality_scan': quality_scan,
                    'statistical_profile': statistical_profile,
                    'overall_score': quality_scan['overall_score']
                }
                
                st.success("✅ AI-Powered Cleaning Complete!")
                st.balloons()
                
                # Show results
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Before Cleaning:**")
                    st.dataframe(df.head(5), use_container_width=True)
                with col2:
                    st.markdown("**After Cleaning:**")
                    st.dataframe(cleaned_df.head(5), use_container_width=True)
                
                # Show cleaning report
                st.markdown("### 📊 Cleaning Results")
                for operation in report.get('operations_performed', []):
                    st.info(f"✅ {operation}")
                
                if 'metrics' in report:
                    metrics = report['metrics']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if 'duplicates_removed' in metrics:
                            st.metric("Duplicates Removed", metrics['duplicates_removed'])
                    with col2:
                        if 'missing_handled' in metrics:
                            st.metric("Missing Values Handled", metrics['missing_handled'])
                    with col3:
                        if 'memory_reduction' in metrics:
                            st.metric("Memory Reduction", metrics['memory_reduction'])
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

def render_profiling_page():
    """Render profiling page"""
    if st.session_state.df is None:
        st.warning("📁 Please upload a dataset first!")
        st.info("Go to the 'Data Upload' page to get started.")
        return
    
    df = st.session_state.df
    
    st.markdown("""
    <div style="background: white; border-radius: 15px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h2 style="color: #2d3748; margin-bottom: 1rem;">🔍 Advanced Data Profiling</h2>
        <p style="color: #718096; font-size: 1.1rem;">
            Comprehensive AI-powered data analysis with quality metrics and statistical insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Run profiling if not already done
    if not st.session_state.data_profile:
        with st.spinner("Running advanced AI profiling..."):
            profiler = AIDataProfiler()
            quality_scan = profiler.comprehensive_quality_scan(df)
            statistical_profile = profiler.generate_statistical_profile(df)
            
            st.session_state.data_profile = {
                'quality_scan': quality_scan,
                'statistical_profile': statistical_profile,
                'overall_score': quality_scan['overall_score']
            }
    
    profile = st.session_state.data_profile
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Quality Analysis", "🔢 Statistics", "📋 Column Details"])
    
    with tab1:
        # Executive summary
        st.markdown("### 📊 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{profile['statistical_profile']['dataset_summary']['rows']:,}")
        with col2:
            st.metric("Total Columns", profile['statistical_profile']['dataset_summary']['columns'])
        with col3:
            st.metric("Memory Usage", f"{profile['statistical_profile']['dataset_summary']['memory_usage_mb']} MB")
        with col4:
            quality_score = profile['overall_score']
            st.metric("Quality Score", f"{quality_score}%")
        
        # Quality score visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=quality_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Quality Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 60], 'color': "#fed7d7"},
                    {'range': [60, 80], 'color': "#feebc8"},
                    {'range': [80, 100], 'color': "#c6f6d5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data type distribution
        st.markdown("### 📊 Data Type Distribution")
        dtype_data = profile['statistical_profile']['data_type_analysis']
        
        fig = px.pie(
            values=[dtype_data['numeric'], dtype_data['categorical'], dtype_data['datetime'], dtype_data['boolean']],
            names=['Numeric', 'Categorical', 'Datetime', 'Boolean'],
            hole=0.4,
            color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Quality analysis
        quality_scan = profile['quality_scan']
        
        st.markdown("### 📈 Data Quality Dimensions")
        
        # Quality cards
        col1, col2, col3 = st.columns(3)
        
        dimensions = [
            ('Completeness', quality_scan['dimensions']['completeness']['score'], '#667eea'),
            ('Uniqueness', quality_scan['dimensions']['uniqueness']['score'], '#764ba2'),
            ('Validity', quality_scan['dimensions']['validity']['score'], '#f093fb')
        ]
        
        for idx, (name, score, color) in enumerate(dimensions):
            with [col1, col2, col3][idx]:
                st.markdown(f"""
                <div style="background: white; border-radius: 10px; padding: 1.5rem; text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.1); border-top: 4px solid {color};">
                    <div style="font-size: 0.9rem; color: #718096; margin-bottom: 0.5rem;">{name}</div>
                    <div style="font-size: 2rem; font-weight: 700; color: {color}; margin-bottom: 0.5rem;">
                        {score:.0f}%
                    </div>
                    <div style="height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden;">
                        <div style="width: {score}%; height: 100%; background: {color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Issues and recommendations
        st.markdown("### 🔍 Issues & Recommendations")
        
        if quality_scan.get('recommendations'):
            for rec in quality_scan['recommendations']:
                if rec['type'] == 'warning':
                    with st.expander(f"⚠️ {rec['title']}", expanded=True):
                        st.write(rec['message'])
                        st.info(f"**Action:** {rec['action']}")
                else:
                    with st.expander(f"ℹ️ {rec['title']}"):
                        st.write(rec['message'])
                        st.info(f"**Action:** {rec['action']}")
        else:
            st.success("✅ No major issues found!")
    
    with tab3:
        # Statistical profile
        stats = profile['statistical_profile']
        
        st.markdown("### 🔢 Descriptive Statistics")
        
        if stats.get('column_statistics'):
            # Select column for detailed stats
            numeric_cols = list(stats['column_statistics'].keys())
            if numeric_cols:
                selected_col = st.selectbox("Select numeric column:", numeric_cols)
                
                if selected_col:
                    col_stats = stats['column_statistics'][selected_col]
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{col_stats['mean']:.2f}")
                    with col2:
                        st.metric("Median", f"{col_stats['median']:.2f}")
                    with col3:
                        st.metric("Std Dev", f"{col_stats['std']:.2f}")
                    with col4:
                        st.metric("Missing", f"{col_stats['missing']:,}")
                    
                    # Create distribution plot
                    fig = px.histogram(df, x=selected_col, nbins=50,
                                     title=f'Distribution of {selected_col}',
                                     template='plotly_white',
                                     color_discrete_sequence=['#667eea'])
                    st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insights
        if stats.get('correlation_insights', {}).get('strong_correlations'):
            st.markdown("### 🔗 Strong Correlations")
            strong_corrs = stats['correlation_insights']['strong_correlations']
            
            for corr in strong_corrs[:5]:  # Show top 5
                st.info(f"**{corr['columns'][0]}** ↔ **{corr['columns'][1]}**: {corr['correlation']:.3f}")
    
    with tab4:
        # Column details
        st.markdown("### 📋 Column Information")
        
        column_info = []
        for col in df.columns:
            col_info = {
                'Column': col,
                'Type': str(df[col].dtype),
                'Missing': df[col].isna().sum(),
                'Missing %': f"{(df[col].isna().sum() / len(df) * 100):.1f}%",
                'Unique': df[col].nunique(),
                'Unique %': f"{(df[col].nunique() / len(df) * 100):.1f}%"
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info['Min'] = f"{df[col].min():.2f}" if not pd.isna(df[col].min()) else "N/A"
                col_info['Max'] = f"{df[col].max():.2f}" if not pd.isna(df[col].max()) else "N/A"
                col_info['Mean'] = f"{df[col].mean():.2f}" if not pd.isna(df[col].mean()) else "N/A"
            
            column_info.append(col_info)
        
        column_df = pd.DataFrame(column_info)
        st.dataframe(column_df, use_container_width=True)

def render_cleaning_page():
    """Render cleaning page"""
    if st.session_state.df is None:
        st.warning("📁 Please upload a dataset first!")
        return
    
    df = st.session_state.df
    
    st.markdown("""
    <div style="background: white; border-radius: 15px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h2 style="color: #2d3748; margin-bottom: 1rem;">🧹 Advanced Data Cleaning</h2>
        <p style="color: #718096; font-size: 1.1rem;">
            Fine-tune your data with advanced cleaning operations and transformations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cleaning tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Missing Values", "📊 Outliers", "📝 Data Types", "🔄 Transformations"])
    
    with tab1:
        st.markdown("### 🔧 Missing Value Treatment")
        
        # Show missing values summary
        missing_summary = df.isna().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) == 0:
            st.success("✅ No missing values found!")
        else:
            # Visualize missing values
            missing_df = pd.DataFrame({
                'Column': missing_cols.index,
                'Missing Count': missing_cols.values,
                'Missing %': (missing_cols.values / len(df) * 100).round(2)
            }).sort_values('Missing %', ascending=False)
            
            st.dataframe(missing_df, use_container_width=True)
            
            # Create bar chart
            fig = px.bar(missing_df.head(10), x='Column', y='Missing %',
                        title='Top 10 Columns with Missing Values',
                        color='Missing %',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
            
            # Treatment options
            st.markdown("### 🛠️ Treatment Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                treatment_method = st.selectbox(
                    "Select treatment method:",
                    ["Select...", "Remove rows with missing", "Remove columns with missing",
                     "Fill with mean", "Fill with median", "Fill with mode", 
                     "Fill with zero", "Forward fill", "Backward fill"]
                )
            
            with col2:
                if treatment_method not in ["Select...", "Remove rows with missing", "Remove columns with missing"]:
                    columns_to_treat = st.multiselect(
                        "Select columns to treat:",
                        missing_cols.index.tolist(),
                        default=missing_cols.index.tolist()[:3]
                    )
            
            if treatment_method != "Select...":
                if st.button("Apply Treatment", type="primary"):
                    cleaned_df = df.copy()
                    
                    try:
                        if treatment_method == "Remove rows with missing":
                            cleaned_df = cleaned_df.dropna()
                            st.success(f"✅ Removed {len(df) - len(cleaned_df)} rows with missing values")
                        
                        elif treatment_method == "Remove columns with missing":
                            cleaned_df = cleaned_df.dropna(axis=1)
                            st.success(f"✅ Removed {len(df.columns) - len(cleaned_df.columns)} columns")
                        
                        elif treatment_method == "Fill with mean":
                            for col in columns_to_treat:
                                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                            st.success(f"✅ Applied mean imputation to {len(columns_to_treat)} columns")
                        
                        elif treatment_method == "Fill with median":
                            for col in columns_to_treat:
                                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                            st.success(f"✅ Applied median imputation to {len(columns_to_treat)} columns")
                        
                        elif treatment_method == "Fill with mode":
                            for col in columns_to_treat:
                                if cleaned_df[col].dtype == 'object':
                                    mode_val = cleaned_df[col].mode()
                                    if not mode_val.empty:
                                        cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
                            st.success(f"✅ Applied mode imputation to {len(columns_to_treat)} columns")
                        
                        elif treatment_method == "Fill with zero":
                            for col in columns_to_treat:
                                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                                    cleaned_df[col] = cleaned_df[col].fillna(0)
                            st.success(f"✅ Filled {len(columns_to_treat)} columns with zero")
                        
                        elif treatment_method == "Forward fill":
                            cleaned_df = cleaned_df.ffill()
                            st.success("✅ Applied forward fill")
                        
                        elif treatment_method == "Backward fill":
                            cleaned_df = cleaned_df.bfill()
                            st.success("✅ Applied backward fill")
                        
                        # Update session state
                        st.session_state.df = cleaned_df
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error applying treatment: {str(e)}")
    
    with tab2:
        st.markdown("### 📊 Outlier Detection & Treatment")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.info("ℹ️ No numeric columns found for outlier analysis")
        else:
            selected_col = st.selectbox("Select column for analysis:", numeric_cols)
            
            if selected_col:
                col_data = df[selected_col].dropna()
                
                if len(col_data) > 0:
                    # Calculate statistics
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Detect outliers
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Values", len(col_data))
                    with col2:
                        st.metric("Outliers", len(outliers))
                    with col3:
                        st.metric("Outlier %", f"{(len(outliers)/len(col_data)*100):.1f}%")
                    with col4:
                        st.metric("IQR", f"{iqr:.2f}")
                    
                    # Create box plot
                    fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Treatment options
                    if len(outliers) > 0:
                        st.markdown("### 🛠️ Outlier Treatment")
                        
                        treatment = st.selectbox(
                            "Select treatment method:",
                            ["Select...", "Clip to bounds", "Remove outliers", "Winsorize", "Log transform"],
                            key="outlier_treatment"
                        )
                        
                        if treatment != "Select...":
                            if st.button("Apply Outlier Treatment", type="primary"):
                                if treatment == "Clip to bounds":
                                    df[selected_col] = df[selected_col].clip(lower_bound, upper_bound)
                                    st.success(f"✅ Clipped outliers in {selected_col}")
                                
                                elif treatment == "Remove outliers":
                                    mask = (df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)
                                    df = df[mask]
                                    st.success(f"✅ Removed {len(outliers)} outliers from {selected_col}")
                                
                                elif treatment == "Winsorize":
                                    lower = col_data.quantile(0.05)
                                    upper = col_data.quantile(0.95)
                                    df[selected_col] = df[selected_col].clip(lower, upper)
                                    st.success(f"✅ Winsorized {selected_col} (5%-95%)")
                                
                                elif treatment == "Log transform":
                                    if df[selected_col].min() > 0:
                                        df[selected_col] = np.log1p(df[selected_col])
                                        st.success(f"✅ Applied log transform to {selected_col}")
                                    else:
                                        st.error("❌ Log transform requires positive values")
                                
                                st.session_state.df = df
                                st.rerun()
    
    with tab3:
        st.markdown("### 📝 Data Type Conversion")
        
        selected_col = st.selectbox("Select column:", df.columns, key="dtype_col")
        
        if selected_col:
            current_dtype = str(df[selected_col].dtype)
            st.write(f"**Current Type:** `{current_dtype}`")
            
            # Show sample values
            with st.expander("Sample Values"):
                st.write(df[selected_col].head(10).tolist())
            
            # Conversion options
            new_dtype = st.selectbox(
                "Convert to:",
                ["Select...", "Integer", "Float", "String", "Datetime", "Boolean", "Category"],
                key="dtype_conversion"
            )
            
            if new_dtype != "Select...":
                if st.button("Convert Data Type", type="primary"):
                    try:
                        if new_dtype == "Integer":
                            df[selected_col] = pd.to_numeric(df[selected_col], errors='coerce').astype('Int64')
                        elif new_dtype == "Float":
                            df[selected_col] = pd.to_numeric(df[selected_col], errors='coerce').astype('float64')
                        elif new_dtype == "String":
                            df[selected_col] = df[selected_col].astype(str)
                        elif new_dtype == "Datetime":
                            df[selected_col] = pd.to_datetime(df[selected_col], errors='coerce')
                        elif new_dtype == "Boolean":
                            df[selected_col] = df[selected_col].astype(bool)
                        elif new_dtype == "Category":
                            df[selected_col] = df[selected_col].astype('category')
                        
                        st.success(f"✅ Converted {selected_col} to {new_dtype}")
                        st.session_state.df = df
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Conversion failed: {str(e)}")
    
    with tab4:
        st.markdown("### 🔄 Advanced Transformations")
        
        transformation = st.selectbox(
            "Select transformation:",
            ["Select...", "Normalize numeric columns", "Standardize numeric columns",
             "One-hot encode categorical", "Label encode categorical", "Extract date features"]
        )
        
        if transformation != "Select...":
            if st.button("Apply Transformation", type="primary"):
                if transformation == "Normalize numeric columns":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        min_val = df[col].min()
                        max_val = df[col].max()
                        if max_val != min_val:
                            df[col] = (df[col] - min_val) / (max_val - min_val)
                    st.success("✅ Normalized all numeric columns")
                
                elif transformation == "Standardize numeric columns":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        if std_val != 0:
                            df[col] = (df[col] - mean_val) / std_val
                    st.success("✅ Standardized all numeric columns")
                
                elif transformation == "One-hot encode categorical":
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    if len(categorical_cols) > 0:
                        df = pd.get_dummies(df, columns=categorical_cols.tolist()[:5])  # Limit to 5
                        st.success("✅ Applied one-hot encoding to categorical columns")
                    else:
                        st.warning("⚠️ No categorical columns found")
                
                st.session_state.df = df
                st.rerun()

def render_visualizations_page():
    """Render visualizations page"""
    if st.session_state.df is None:
        st.warning("📁 Please upload a dataset first!")
        return
    
    df = st.session_state.df
    
    st.markdown("""
    <div style="background: white; border-radius: 15px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h2 style="color: #2d3748; margin-bottom: 1rem;">📈 Advanced Visualizations</h2>
        <p style="color: #718096; font-size: 1.1rem;">
            Create stunning, interactive visualizations to explore and understand your data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create dashboard
    engine = VisualizationEngine()
    dashboard = engine.create_dashboard(
        df,
        chart_types=['distribution', 'correlation', 'scatter', 'categorical', 'box']
    )
    
    # Display charts
    if 'error' in dashboard:
        st.error(f"❌ Error creating visualizations: {dashboard['error']}")
    else:
        # Distribution charts
        if any(key.startswith('distribution_') for key in dashboard.keys()):
            st.markdown("### 📊 Distribution Charts")
            dist_cols = [col for col in dashboard.keys() if col.startswith('distribution_')]
            
            cols = st.columns(min(3, len(dist_cols)))
            for idx, col_key in enumerate(dist_cols[:3]):
                with cols[idx]:
                    st.plotly_chart(dashboard[col_key], use_container_width=True)
        
        # Correlation heatmap
        if 'correlation_heatmap' in dashboard:
            st.markdown("### 🔗 Correlation Heatmap")
            st.plotly_chart(dashboard['correlation_heatmap'], use_container_width=True)
        
        # Scatter plot
        if 'scatter_plot' in dashboard:
            st.markdown("### 📈 Scatter Plot")
            st.plotly_chart(dashboard['scatter_plot'], use_container_width=True)
        
        # Categorical charts
        if any(key.startswith('categorical_') for key in dashboard.keys()):
            st.markdown("### 📋 Categorical Distributions")
            cat_cols = [col for col in dashboard.keys() if col.startswith('categorical_')]
            
            cols = st.columns(min(2, len(cat_cols)))
            for idx, col_key in enumerate(cat_cols[:2]):
                with cols[idx]:
                    st.plotly_chart(dashboard[col_key], use_container_width=True)
        
        # Box plots
        if any(key.startswith('box_') for key in dashboard.keys()):
            st.markdown("### 📦 Box Plots")
            box_cols = [col for col in dashboard.keys() if col.startswith('box_')]
            
            cols = st.columns(min(2, len(box_cols)))
            for idx, col_key in enumerate(box_cols[:2]):
                with cols[idx]:
                    st.plotly_chart(dashboard[col_key], use_container_width=True)
    
    # Custom visualization creator
    st.markdown("---")
    st.markdown("### 🎨 Custom Visualization Creator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type:",
            ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin", "Heatmap"]
        )
    
    with col2:
        x_col = st.selectbox("X-axis:", df.columns.tolist())
    
    if chart_type in ["Scatter", "Line", "Bar"]:
        y_col = st.selectbox("Y-axis:", df.columns.tolist())
    
    if st.button("Create Custom Chart", type="primary"):
        try:
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col,
                               title=f'{y_col} vs {x_col}',
                               template='plotly_white')
            
            elif chart_type == "Line":
                fig = px.line(df, x=x_col, y=y_col,
                            title=f'{y_col} over {x_col}',
                            template='plotly_white')
            
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_col, y=y_col,
                           title=f'{y_col} by {x_col}',
                           template='plotly_white')
            
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_col,
                                 title=f'Distribution of {x_col}',
                                 template='plotly_white')
            
            elif chart_type == "Box":
                fig = px.box(df, y=x_col,
                           title=f'Box Plot of {x_col}',
                           template='plotly_white')
            
            elif chart_type == "Violin":
                fig = px.violin(df, y=x_col,
                              title=f'Violin Plot of {x_col}',
                              template='plotly_white')
            
            elif chart_type == "Heatmap":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix,
                                  text_auto='.2f',
                                  color_continuous_scale='RdBu',
                                  title='Correlation Heatmap',
                                  template='plotly_white')
            
            if 'fig' in locals():
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error creating chart: {str(e)}")

def render_ml_page():
    """Render ML analysis page"""
    if st.session_state.df is None:
        st.warning("📁 Please upload a dataset first!")
        return
    
    df = st.session_state.df
    
    st.markdown("""
    <div style="background: white; border-radius: 15px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h2 style="color: #2d3748; margin-bottom: 1rem;">🤖 Machine Learning Analysis</h2>
        <p style="color: #718096; font-size: 1.1rem;">
            Advanced machine learning analysis, feature importance, and predictive insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ML readiness analysis
    st.markdown("### 📊 ML Readiness Analysis")
    
    # Feature analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Features", len(df.columns))
    
    with col2:
        st.metric("Numeric Features", len(numeric_cols))
    
    with col3:
        st.metric("Categorical Features", len(categorical_cols))
    
    with col4:
        missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        readiness_score = max(0, 100 - min(missing_pct * 2, 100))
        st.metric("ML Readiness", f"{readiness_score:.0f}%")
    
    # Target variable suggestions
    st.markdown("### 🎯 Target Variable Suggestions")
    
    target_candidates = []
    for col in df.columns:
        # Binary classification
        if df[col].nunique() == 2:
            balance = min(df[col].value_counts(normalize=True)) * 100
            target_candidates.append({
                'column': col,
                'type': 'Binary Classification',
                'balance': f"{balance:.1f}%",
                'suitability': 'High' if balance > 20 else 'Medium'
            })
        
        # Multi-class classification
        elif 3 <= df[col].nunique() <= 20:
            balance = df[col].value_counts(normalize=True).min() * 100
            target_candidates.append({
                'column': col,
                'type': 'Multi-class Classification',
                'balance': f"{balance:.1f}%",
                'suitability': 'High' if balance > 5 else 'Medium'
            })
        
        # Regression
        elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 20:
            cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
            target_candidates.append({
                'column': col,
                'type': 'Regression',
                'balance': f"CV: {cv:.3f}",
                'suitability': 'High' if cv > 0.1 else 'Medium'
            })
    
    if target_candidates:
        target_df = pd.DataFrame(target_candidates)
        st.dataframe(target_df, use_container_width=True)
        
        # Feature correlation with selected target
        if len(numeric_cols) >= 2:
            st.markdown("### 🔗 Feature Correlations")
            
            target_col = st.selectbox(
                "Select target variable for correlation analysis:",
                [c['column'] for c in target_candidates if c['column'] in numeric_cols]
            )
            
            if target_col:
                # Calculate correlations
                corr_with_target = df[numeric_cols].corr()[target_col].drop(target_col)
                corr_df = pd.DataFrame({
                    'Feature': corr_with_target.index,
                    'Correlation': corr_with_target.values
                }).sort_values('Correlation', key=abs, ascending=False)
                
                # Display as bar chart
                fig = px.bar(corr_df.head(10), x='Feature', y='Correlation',
                           color='Correlation',
                           color_continuous_scale='RdBu',
                           title=f'Top 10 Features Correlated with {target_col}',
                           template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ No suitable target variables identified. Consider feature engineering.")
    
    # ML suggestions
    st.markdown("### 💡 ML Suggestions")
    
    suggestions = []
    
    if len(numeric_cols) >= 3:
        suggestions.append("**Clustering Analysis**: Use K-means or DBSCAN to find natural groupings in your data")
    
    if any(df[col].nunique() == 2 for col in df.columns):
        suggestions.append("**Binary Classification**: Predict yes/no outcomes using Logistic Regression or Random Forest")
    
    if any(3 <= df[col].nunique() <= 10 for col in df.columns):
        suggestions.append("**Multi-class Classification**: Predict categories with Random Forest or XGBoost")
    
    if len(numeric_cols) >= 2:
        suggestions.append("**Regression Analysis**: Predict continuous values with Linear Regression or Gradient Boosting")
    
    if suggestions:
        for suggestion in suggestions:
            st.info(suggestion)
    else:
        st.info("ℹ️ Consider collecting more data or performing feature engineering for ML tasks")

def render_export_page():
    """Render export page"""
    if st.session_state.df is None:
        st.warning("📁 Please upload a dataset first!")
        return
    
    df = st.session_state.df
    
    st.markdown("""
    <div style="background: white; border-radius: 15px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h2 style="color: #2d3748; margin-bottom: 1rem;">📋 Report & Export</h2>
        <p style="color: #718096; font-size: 1.1rem;">
            Generate comprehensive reports and export your cleaned, analyzed data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Report generation
    st.markdown("### 📊 Generate Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type:",
            ["Comprehensive Report", "Data Quality Report", "Statistical Summary", "ML Readiness Report"]
        )
    
    with col2:
        include_charts = st.checkbox("Include Charts & Visualizations", value=True)
    
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            # Create report content
            if st.session_state.data_profile:
                profile = st.session_state.data_profile
                quality_score = profile.get('overall_score', 0)
                stats = profile.get('statistical_profile', {})
            else:
                quality_score = 0
                stats = {}
            
            report_content = f"""
            # End-to-End Data Analysis and Machine Learning Platform ✅ Report
            **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            **Report Type:** {report_type}
            **Dataset:** {df.shape[0]} rows × {df.shape[1]} columns
            
            ## Executive Summary
            
            - **Overall Quality Score:** {quality_score}%
            - **Missing Values:** {df.isna().sum().sum():,} cells
            - **Duplicate Rows:** {df.duplicated().sum():,}
            - **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
            
            ## Data Overview
            
            **Data Types:**
            - Numeric: {len(df.select_dtypes(include=[np.number]).columns)} columns
            - Categorical: {len(df.select_dtypes(include=['object', 'category']).columns)} columns
            - Datetime: {len(df.select_dtypes(include=['datetime64']).columns)} columns
            
            **Column Statistics:**
            {df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else "No numeric columns"}
            
            ## Recommendations
            
            1. Review data quality metrics
            2. Consider additional cleaning if quality score < 80%
            3. Explore feature engineering opportunities
            4. Export cleaned data for further analysis
            """
            
            st.success("✅ Report generated successfully!")
            
            with st.expander("📄 Report Preview", expanded=True):
                st.markdown(report_content)
    
    # Export options
    st.markdown("### 📤 Export Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Export CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Export CSV",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Cleaned Data', index=False)
            if st.session_state.df_original is not None:
                st.session_state.df_original.to_excel(writer, sheet_name='Original Data', index=False)
        output.seek(0)
        
        st.download_button(
            label="📊 Export Excel",
            data=output,
            file_name="data_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        # Export JSON
        report_data = {
            'dataset_info': {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'columns_list': df.columns.tolist()
            },
            'quality_metrics': st.session_state.data_profile.get('quality_scan', {}),
            'generated_at': datetime.now().isoformat()
        }
        
        report_json = json.dumps(report_data, indent=2, default=str)
        st.download_button(
            label="📋 Export JSON",
            data=report_json,
            file_name="data_report.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col4:
        # Export HTML Report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>End-to-End Data Analysis and Machine Learning Platform ✅ Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 30px; border-radius: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>End-to-End Data Analysis and Machine Learning Platform ✅ Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>Dataset Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Rows</td><td>{df.shape[0]:,}</td></tr>
                <tr><td>Total Columns</td><td>{df.shape[1]}</td></tr>
                <tr><td>Missing Values</td><td>{df.isna().sum().sum():,}</td></tr>
                <tr><td>Duplicate Rows</td><td>{df.duplicated().sum():,}</td></tr>
                <tr><td>Memory Usage</td><td>{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</td></tr>
            </table>
            
            <h2>Data Preview</h2>
            {df.head(20).to_html()}
        </body>
        </html>
        """
        
        st.download_button(
            label="🌐 Export HTML",
            data=html_report,
            file_name="data_report.html",
            mime="text/html",
            use_container_width=True
        )
    
    # Data preview
    st.markdown("### 📋 Data Preview")
    
    preview_rows = st.slider("Number of rows to preview:", 5, 100, 20)
    st.dataframe(df.head(preview_rows), use_container_width=True)

# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":
    try:
        main()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #2c3e50, #1a2530); border-radius: 15px; color: white;">
            <h3 style="margin-bottom: 1rem;">End-to-End Data Analysis and Machine Learning Platform ✅</h3>
            <p style="color: rgba(255,255,255,0.5); font-size: 0.9rem;">
                © 2025 End-to-End Data Analysis and Machine Learning Platform  
                | Developed by Lohith Reddy Gayam
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.code(traceback.format_exc())
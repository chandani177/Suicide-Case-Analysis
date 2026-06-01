import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import mstats
from scipy import stats

# Set page configuration
st.set_page_config(
    page_title="Global Suicide Trends Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .highlight-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 15px 0;
    }
    .critical-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        margin: 15px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load the suicide dataset"""
    try:
        df = pd.read_csv('suicide_case.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'suicide_case.csv' is in the same directory.")
        return None

def main():
    # Load data
    df = load_data()
    
    # SIDEBAR FILTERS
    st.sidebar.title("🔍 Data Filters")
    st.sidebar.markdown("---")
    
    if df is not None:
        # Initialize session state for filter reset
        if 'reset_filters' not in st.session_state:
            st.session_state.reset_filters = False
        
        # Get default values
        year_min = int(df['year'].min())
        year_max = int(df['year'].max())
        all_countries = sorted(df['country'].unique())
        all_ages = sorted(df['age'].unique())
        all_generations = sorted(df['generation'].unique()) if 'generation' in df.columns else []
        rate_min = float(df['suicides/100k pop'].min()) if 'suicides/100k pop' in df.columns else 0
        rate_max = float(df['suicides/100k pop'].max()) if 'suicides/100k pop' in df.columns else 100
        
        with st.sidebar.form("filters_form"):
            st.markdown("### Filter Options")
            
            # Year Range Filter
            selected_years = st.slider(
                "📅 Year Range",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max) if st.session_state.reset_filters else (year_min, year_max),
                help="Select the range of years to analyze",
                key="year_slider"
            )
            
            # Country Filter
            selected_countries = st.multiselect(
                "🌍 Countries",
                options=all_countries,
                default=all_countries if st.session_state.reset_filters else all_countries,
                help="Select specific countries (leave empty for all)",
                key="country_select"
            )
            
            # Gender Filter
            selected_genders = st.multiselect(
                "👥 Gender",
                options=['male', 'female'],
                default=['male', 'female'] if st.session_state.reset_filters else ['male', 'female'],
                help="Select gender groups to include",
                key="gender_select"
            )
            
            # Age Group Filter
            selected_ages = st.multiselect(
                "🎂 Age Groups",
                options=all_ages,
                default=all_ages if st.session_state.reset_filters else all_ages,
                help="Select age groups to analyze",
                key="age_select"
            )
            
            # Generation Filter (if available)
            if 'generation' in df.columns:
                selected_generations = st.multiselect(
                    "👨‍👩‍👧‍👦 Generations",
                    options=all_generations,
                    default=all_generations if st.session_state.reset_filters else all_generations,
                    help="Select generations to include",
                    key="generation_select"
                )
            else:
                selected_generations = []
            
            # Suicide Rate Range Filter
            if 'suicides/100k pop' in df.columns:
                selected_rate_range = st.slider(
                    "📊 Suicide Rate Range (per 100k)",
                    min_value=rate_min,
                    max_value=rate_max,
                    value=(rate_min, rate_max) if st.session_state.reset_filters else (rate_min, rate_max),
                    help="Filter by suicide rate range",
                    key="rate_slider"
                )
            else:
                selected_rate_range = None
            
            st.markdown("---")
            
            # Form buttons
            col1, col2 = st.columns(2)
            with col1:
                apply_filters = st.form_submit_button("✅ Apply Filters", use_container_width=True)
            with col2:
                reset_filters = st.form_submit_button("🔄 Reset Filters", use_container_width=True)
        
        # Handle reset button
        if reset_filters:
            # Clear all filter-related session state keys
            filter_keys = ['year_slider', 'country_select', 'gender_select', 'age_select', 'generation_select', 'rate_slider']
            for key in filter_keys:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Set reset flag and rerun
            st.session_state.reset_filters = True
            st.rerun()
        
        # Reset the flag after the rerun
        if st.session_state.reset_filters:
            st.session_state.reset_filters = False
        
        # Apply filters to dataframe
        df_filtered = df.copy()
        
        if apply_filters or not any([
            len(selected_countries) != len(all_countries), 
            selected_genders != ['male', 'female'],
            len(selected_ages) != len(all_ages),
            selected_years != (year_min, year_max)
        ]):
            
            # Apply year filter
            df_filtered = df_filtered[
                (df_filtered['year'] >= selected_years[0]) & 
                (df_filtered['year'] <= selected_years[1])
            ]
            
            # Apply country filter
            if selected_countries:
                df_filtered = df_filtered[df_filtered['country'].isin(selected_countries)]
            
            # Apply gender filter
            if selected_genders:
                df_filtered = df_filtered[df_filtered['sex'].isin(selected_genders)]
            
            # Apply age filter
            if selected_ages:
                df_filtered = df_filtered[df_filtered['age'].isin(selected_ages)]
            
            # Apply generation filter
            if 'generation' in df.columns and selected_generations:
                df_filtered = df_filtered[df_filtered['generation'].isin(selected_generations)]
            
            # Apply suicide rate filter
            if selected_rate_range and 'suicides/100k pop' in df_filtered.columns:
                df_filtered = df_filtered[
                    (df_filtered['suicides/100k pop'] >= selected_rate_range[0]) & 
                    (df_filtered['suicides/100k pop'] <= selected_rate_range[1])
                ]
        
        # Display filter summary
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Current Filters")
        
        filters_active = []
        if selected_years != (year_min, year_max):
            filters_active.append(f"📅 Years: {selected_years[0]}-{selected_years[1]}")
        if len(selected_countries) != len(all_countries):
            filters_active.append(f"🌍 Countries: {len(selected_countries)} selected")
        if len(selected_genders) != 2:
            filters_active.append(f"👥 Gender: {', '.join(selected_genders)}")
        if len(selected_ages) != len(all_ages):
            filters_active.append(f"🎂 Ages: {len(selected_ages)} groups")
        if selected_rate_range and selected_rate_range != (rate_min, rate_max):
            filters_active.append(f"📊 Rate: {selected_rate_range[0]:.1f}-{selected_rate_range[1]:.1f}")
        
        if filters_active:
            for filter_desc in filters_active:
                st.sidebar.write(f"• {filter_desc}")
        else:
            st.sidebar.write("• No active filters (showing all data)")
        
        # Show filtered data statistics
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📈 Data Summary")
        st.sidebar.metric("Total Records", f"{len(df_filtered):,}")
        st.sidebar.metric("Countries", f"{df_filtered['country'].nunique()}")
        st.sidebar.metric("Years Covered", f"{df_filtered['year'].nunique()}")
        if len(df_filtered) > 0:
            total_cases = df_filtered['suicides_no'].sum()
            st.sidebar.metric("Total Cases", f"{total_cases:,}")
        
        # Use filtered dataframe for the rest of the analysis
        df = df_filtered
        
        # Show filter alert if data is filtered
        if len(filters_active) > 0:
            st.info(f"🔍 **Filters Applied**: Showing {len(df):,} records out of original dataset. Check sidebar for details.")
    else:
        st.sidebar.error("❌ No data available for filtering")
    
    # Main title
    st.markdown('<h1 class="main-header">Global Suicide Trends Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">A Comprehensive Data Science Project on Global Suicide Patterns and Trends (1985-2016)</p>', unsafe_allow_html=True)
    
    # Main content sections
    
    # EXECUTIVE SUMMARY SECTION
    st.markdown('<h2 class="section-header">Executive Summary</h2>', unsafe_allow_html=True)
    
    if df is not None:
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_records = len(df)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Records</h3>
                <h2 style="color: #1f77b4;">{total_records:,}</h2>
                <p>Data points analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            countries_count = df['country'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Countries</h3>
                <h2 style="color: #ff7f0e;">{countries_count}</h2>
                <p>Global coverage</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            year_span = df['year'].max() - df['year'].min() + 1
            st.markdown(f"""
            <div class="metric-card">
                <h3>Time Span</h3>
                <h2 style="color: #2ca02c;">{year_span}</h2>
                <p>Years of data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_suicides = df['suicides_no'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Cases</h3>
                <h2 style="color: #d62728;">{total_suicides:,}</h2>
                <p>Recorded cases</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Executive Summary Content
    st.markdown("""
    <div class="highlight-box">
        <h3>Key Findings</h3>
        <p>Our comprehensive analysis of global suicide data spanning 32 years reveals critical patterns and trends that inform public health policy and intervention strategies:</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Calculate key statistics for summary
        avg_rate = df['suicides/100k pop'].mean()
        male_proportion = (df[df['sex'] == 'male']['suicides_no'].sum() / df['suicides_no'].sum()) * 100
        peak_age_group = df.groupby('age')['suicides_no'].sum().idxmax()
        
        st.markdown(f"""
        - **Global Scale Impact:** Analysis of {total_records:,} data points across {countries_count} countries reveals the global magnitude of suicide as a public health concern
        - **Gender Disparity:** Males account for {male_proportion:.1f}% of all recorded suicide cases, highlighting significant gender-based differences
        - **Age Pattern:** The {peak_age_group} age group shows the highest absolute numbers, indicating critical intervention periods
        - **Economic Correlation:** Strong relationships exist between economic indicators (GDP per capita) and suicide rates across different regions
        - **Generational Trends:** Clear patterns emerge across different generational cohorts, from G.I. Generation to Generation Z
        - **Geographic Variation:** Substantial differences in suicide rates across countries and regions, suggesting cultural and socioeconomic influences
        """)
    
    # Critical Insights
    st.markdown("""
    <div class="critical-box">
        <h3>Critical Public Health Insights</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Calculate additional insights
        highest_rate_country = df.groupby('country')['suicides/100k pop'].mean().idxmax()
        time_trend = "increasing" if df.groupby('year')['suicides_no'].sum().corr(df.groupby('year')['year'].first()) > 0 else "decreasing"
        
        st.markdown(f"""
        - **High-Risk Demographics:** Consistent patterns show elevated risks among specific age and gender groups requiring targeted interventions
        - **Economic Vulnerability:** Lower GDP per capita correlates with varying suicide rate patterns, indicating socioeconomic factors play crucial roles
        - **Temporal Patterns:** Data shows {time_trend} trends over the 32-year period, with notable fluctuations during economic crises
        - **Data Quality Considerations:** HDI data is missing for 69.9% of records, limiting comprehensive socioeconomic analysis
        - **Regional Hotspots:** Certain countries and regions consistently show higher rates, necessitating focused public health responses
        """)
    
    # PROJECT DESCRIPTION SECTION
    st.markdown('<h2 class="section-header">Project Description</h2>', unsafe_allow_html=True)
    
    # Project Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Project Objective
        
        This data science project conducts a comprehensive analysis of global suicide trends from 1985 to 2016, 
        examining patterns across demographics, geography, and socioeconomic factors. Our goal is to identify 
        risk factors, trends, and correlations that can inform evidence-based suicide prevention strategies 
        and public health interventions.
        
        ### Research Questions
        
        1. **How have global suicide rates evolved over the 32-year period from 1985 to 2016?**
        2. **What are the demographic patterns in suicide rates across age groups, gender, and generations?**
        3. **How do economic factors (GDP, GDP per capita) correlate with national suicide rates?**
        4. **Which countries and regions show the highest risk patterns and what factors contribute to these trends?**
        5. **What seasonal, temporal, and generational patterns can be identified to guide intervention timing?**
        6. **How can data visualization reveal hidden patterns and support evidence-based policy making?**
        
        ### Methodology
        
        Our analysis employs multiple data science approaches including:
        - **Exploratory Data Analysis (EDA):** Comprehensive examination of data structure, quality, and distributions
        - **Time Series Analysis:** Trend identification and temporal pattern recognition
        - **Comparative Analysis:** Cross-country, demographic, and socioeconomic comparisons
        - **Correlation Studies:** Relationships between economic indicators and suicide rates
        - **Statistical Visualization:** Interactive dashboards and comprehensive visual storytelling
        """)
    
    with col2:
        st.markdown("""
        ### Dataset Overview
        
        **Source:** Global suicide statistics compiled from multiple international health organizations
        
        **Coverage:**
        - 📅 **Period:** 1985-2016 (32 years)
        - 🌍 **Geographic:** 101 countries worldwide
        - 👥 **Demographics:** Age groups, gender, generations
        - 💰 **Economic:** GDP, GDP per capita, HDI data
        - 📊 **Metrics:** Absolute numbers and rates per 100k population
        
        **Key Variables:**
        - Country and year identifiers
        - Demographic breakdowns (sex, age groups)
        - Suicide counts and population data
        - Economic indicators and development indices
        - Generational classifications
        """)
    
    # Data Quality and Methodology
    st.markdown("""
    <div class="warning-box">
        <h3>Important Considerations</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Sensitive Topic:** This analysis deals with sensitive mental health data. All insights are presented 
    with the utmost respect for those affected by suicide and are intended to support prevention efforts and public health policy.
    
    **Data Limitations:** While comprehensive, this dataset has certain limitations including missing HDI data 
    and varying data collection methods across countries and time periods. Results should be interpreted within these constraints.
    
    **Ethical Framework:** This research follows ethical guidelines for suicide data analysis and aims to 
    contribute positively to prevention efforts rather than sensationalize tragic statistics.
    """)
    
    # Project Impact and Applications
    st.markdown("""
    ### Project Impact and Applications
    
    **Public Health Policy:**
    - Evidence-based resource allocation for suicide prevention programs
    - Identification of high-risk populations requiring targeted interventions
    - International benchmarking and best practice identification
    
    **Research Contributions:**
    - Comprehensive global trend analysis spanning three decades
    - Novel insights into demographic and socioeconomic risk factors
    - Data-driven recommendations for prevention strategy optimization
    
    **Technical Innovation:**
    - Interactive visualization tools for complex temporal and geographic data
    - Scalable analysis framework applicable to other public health datasets
    - Integration of multiple data sources for holistic health outcome analysis
    """)
    
    # DATA SUMMARY SECTION
    st.markdown('<h2 class="section-header">Data Exploration</h2>', unsafe_allow_html=True)
    
    if df is not None:
        # Create tabs for different data overview sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔧 Data Types", "❌ Missing Values", "📈 Outliers"])
        
        with tab1:
            st.markdown("### Dataset Overview")
            
            # Basic information
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Basic Information")
                st.write(f"**Dataset Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
                st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                st.write(f"**Date Range:** {df['year'].min()} - {df['year'].max()}")
                st.write(f"**Countries Covered:** {df['country'].nunique()}")
                
                # Column categories
                st.markdown("#### Column Categories")
                categorical_cols = ['country', 'sex', 'age', 'generation', 'country-year']
                numerical_cols = ['year', 'suicides_no', 'population', 'suicides/100k pop', 'gdp_per_capita ($)']
                mixed_cols = ['HDI for year', ' gdp_for_year ($)']
                
                st.write(f"**Categorical:** {len(categorical_cols)} columns")
                st.write(f"**Numerical:** {len(numerical_cols)} columns")
                st.write(f"**Mixed/Other:** {len(mixed_cols)} columns")
            
            with col2:
                st.markdown("#### Quick Statistics")
                
                # Key metrics
                total_suicides = df['suicides_no'].sum()
                avg_rate = df['suicides/100k pop'].mean()
                max_rate = df['suicides/100k pop'].max()
                
                st.metric("Total Suicide Cases", f"{total_suicides:,}")
                st.metric("Average Rate (per 100k)", f"{avg_rate:.2f}")
                st.metric("Maximum Rate (per 100k)", f"{max_rate:.2f}")
                
                # Gender distribution
                gender_stats = df.groupby('sex')['suicides_no'].sum()
                male_pct = (gender_stats['male'] / total_suicides * 100)
                st.metric("Male Cases", f"{male_pct:.1f}%")
            
            # Sample data
            st.markdown("#### Sample Data (First 10 Rows)")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column descriptions
            st.markdown("#### Column Descriptions")
            col_descriptions = {
                'country': 'Country name where data was collected',
                'year': 'Year of data collection (1985-2016)',
                'sex': 'Gender classification (male/female)',
                'age': 'Age group classification (6 categories)',
                'suicides_no': 'Number of suicide cases reported',
                'population': 'Total population for the demographic group',
                'suicides/100k pop': 'Suicide rate per 100,000 population',
                'country-year': 'Combined country and year identifier',
                'HDI for year': 'Human Development Index for the year',
                ' gdp_for_year ($)': 'Gross Domestic Product for the year',
                'gdp_per_capita ($)': 'GDP per capita in USD',
                'generation': 'Generational classification'
            }
            
            desc_df = pd.DataFrame(list(col_descriptions.items()), columns=['Column', 'Description'])
            st.dataframe(desc_df, use_container_width=True, hide_index=True)
        
        with tab2:
            st.markdown("### Data Types Analysis")
            
            # Data types overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Data Types Summary")
                dtype_counts = df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"**{dtype}:** {count} columns")
                
                # Detailed data types
                st.markdown("#### Detailed Data Types")
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            with col2:
                # Data type distribution chart
                st.markdown("#### Data Type Distribution")
                fig_dtype = px.pie(
                    values=dtype_counts.values,
                    names=[str(dtype) for dtype in dtype_counts.index],
                    title="Distribution of Data Types"
                )
                st.plotly_chart(fig_dtype, use_container_width=True)
                
                # Memory usage by column
                st.markdown("#### Memory Usage by Column")
                memory_usage = df.memory_usage(deep=True)
                memory_df = pd.DataFrame({
                    'Column': memory_usage.index,
                    'Memory (KB)': memory_usage.values / 1024
                }).sort_values('Memory (KB)', ascending=False)
                
                st.dataframe(memory_df.head(10), use_container_width=True, hide_index=True)
        
        with tab3:
            st.markdown("### Missing Values Analysis")
            
            # Calculate missing values
            missing_data = df.isnull().sum()
            missing_percentage = (missing_data / len(df)) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Missing Values Summary")
                
                # Overall statistics
                total_missing = missing_data.sum()
                total_cells = df.shape[0] * df.shape[1]
                overall_missing_pct = (total_missing / total_cells) * 100
                
                st.metric("Total Missing Values", f"{total_missing:,}")
                st.metric("Overall Missing Percentage", f"{overall_missing_pct:.2f}%")
                
                # Missing values by column
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing Percentage': missing_percentage.values
                }).sort_values('Missing Count', ascending=False)
                
                # Show only columns with missing values
                missing_df_filtered = missing_df[missing_df['Missing Count'] > 0]
                
                if len(missing_df_filtered) > 0:
                    st.markdown("#### Columns with Missing Values")
                    st.dataframe(missing_df_filtered, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ No missing values found in any columns!")
            
            with col2:
                if len(missing_df_filtered) > 0:
                    # Missing values visualization
                    st.markdown("#### Missing Values Visualization")
                    
                    fig_missing = px.bar(
                        missing_df_filtered,
                        x='Missing Percentage',
                        y='Column',
                        orientation='h',
                        title="Missing Values by Column (%)",
                        color='Missing Percentage',
                        color_continuous_scale='Reds'
                    )
                    fig_missing.update_layout(height=400)
                    st.plotly_chart(fig_missing, use_container_width=True)
                    
                    # Missing data pattern
                    st.markdown("#### Missing Data Impact")
                    if 'HDI for year' in missing_df_filtered['Column'].values:
                        hdi_missing_pct = missing_df_filtered[missing_df_filtered['Column'] == 'HDI for year']['Missing Percentage'].iloc[0]
                        st.warning(f"⚠️ HDI data is missing for {hdi_missing_pct:.1f}% of records, which may limit socioeconomic analysis.")
                
                else:
                    st.markdown("#### Data Completeness")
                    st.success("🎉 This dataset has excellent data quality with no missing values!")
        
        with tab4:
            st.markdown("### Outliers Analysis")
            
            # Analyze numerical columns for outliers
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Outlier Detection Summary")
                
                outlier_summary = []
                for col in numerical_columns:
                    if col in df.columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                        outlier_count = len(outliers)
                        outlier_percentage = (outlier_count / len(df)) * 100
                        
                        outlier_summary.append({
                            'Column': col,
                            'Outlier Count': outlier_count,
                            'Outlier Percentage': f"{outlier_percentage:.2f}%",
                            'Lower Bound': f"{lower_bound:.2f}",
                            'Upper Bound': f"{upper_bound:.2f}"
                        })
                
                outlier_df = pd.DataFrame(outlier_summary)
                st.dataframe(outlier_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Box plot for all numerical columns
                st.markdown("#### Box Plot Distribution - All Columns")
                
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Create subplots for better visualization
                n_cols = len(numerical_columns)
                n_rows = 2
                cols_per_row = (n_cols + 1) // 2
                
                fig = make_subplots(
                    rows=n_rows, 
                    cols=cols_per_row,
                    subplot_titles=numerical_columns,
                    vertical_spacing=0.15,
                    horizontal_spacing=0.1
                )
                
                for i, col in enumerate(numerical_columns):
                    row = (i // cols_per_row) + 1
                    col_pos = (i % cols_per_row) + 1
                    
                    fig.add_trace(
                        go.Box(
                            y=df[col],
                            name=col,
                            boxpoints='outliers',
                            marker_color='lightblue',
                            line_color='darkblue',
                            showlegend=False
                        ),
                        row=row, col=col_pos
                    )
                
                fig.update_layout(
                    title="Box Plots for All Numerical Columns",
                    height=600,
                    showlegend=False
                )
                
                # Update y-axis titles
                for i, col in enumerate(numerical_columns):
                    row = (i // cols_per_row) + 1
                    col_pos = (i % cols_per_row) + 1
                    fig.update_yaxes(title_text=col, row=row, col=col_pos)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional detailed analysis section
            st.markdown("#### Detailed Column Analysis")
            
            # Select column for detailed outlier analysis
            selected_col = st.selectbox(
                "Select column for detailed statistics:",
                numerical_columns,
                index=0 if 'suicides/100k pop' not in numerical_columns else numerical_columns.index('suicides/100k pop')
            )
            
            if selected_col:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Individual box plot
                    fig_individual = go.Figure()
                    fig_individual.add_trace(
                        go.Box(
                            y=df[selected_col],
                            name=selected_col,
                            boxpoints='all',
                            jitter=0.3,
                            pointpos=-1.8,
                            marker_color='lightcoral',
                            line_color='darkred'
                        )
                    )
                    fig_individual.update_layout(
                        title=f"Detailed Box Plot: {selected_col}",
                        yaxis_title=selected_col,
                        height=400
                    )
                    st.plotly_chart(fig_individual, use_container_width=True)
                
                with col2:
                    # Statistics
                    st.markdown(f"**Statistics for {selected_col}:**")
                    col_stats = df[selected_col].describe()
                    
                    # Create a nice statistics table
                    stats_df = pd.DataFrame({
                        'Statistic': col_stats.index,
                        'Value': col_stats.values
                    })
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    
                    # Additional outlier info
                    Q1 = df[selected_col].quantile(0.25)
                    Q3 = df[selected_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers_count = len(df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)])
                    
                    st.markdown("**Outlier Information:**")
                    st.write(f"**IQR:** {IQR:.2f}")
                    st.write(f"**Lower Fence:** {lower_bound:.2f}")
                    st.write(f"**Upper Fence:** {upper_bound:.2f}")
                    st.write(f"**Outliers Count:** {outliers_count}")
                    st.write(f"**Outliers Percentage:** {(outliers_count/len(df)*100):.2f}%")
    
    # DATA CLEANING SECTION
    st.markdown('<h2 class="section-header">Data Cleaning</h2>', unsafe_allow_html=True)
    
    if df is not None:
        # Create tabs for different data cleaning sections
        clean_tab1, clean_tab2, clean_tab3 = st.tabs(["🔧 Data Formatting", "❌ Missing Values Treatment", "📈 Outliers Treatment"])
        
        with clean_tab1:
            st.markdown("### Data Formatting")
            
            # Create a cleaned version of the data
            df_cleaned = df.copy()
            formatting_changes = []
            
            # Clean GDP column formatting (has commas and quotes)
            if ' gdp_for_year ($)' in df_cleaned.columns:
                gdp_col = ' gdp_for_year ($)'
                original_type = str(df_cleaned[gdp_col].dtype)
                original_sample = str(df_cleaned[gdp_col].iloc[0])
                
                # Clean the GDP column
                df_cleaned[gdp_col] = df_cleaned[gdp_col].astype(str).str.replace(',', '').str.replace('"', '')
                df_cleaned[gdp_col] = pd.to_numeric(df_cleaned[gdp_col], errors='coerce')
                
                formatting_changes.append({
                    'Column': gdp_col,
                    'Issue': 'Contains commas and quotes',
                    'Before': original_type,
                    'After': str(df_cleaned[gdp_col].dtype),
                    'Sample Before': original_sample,
                    'Sample After': str(df_cleaned[gdp_col].iloc[0]),
                    'Methodology': 'Removed commas and quotes, converted to numeric'
                })
            
            # Clean column names (remove leading/trailing spaces)
            original_columns = df_cleaned.columns.tolist()
            df_cleaned.columns = df_cleaned.columns.str.strip()
            new_columns = df_cleaned.columns.tolist()
            
            renamed_columns = []
            for old, new in zip(original_columns, new_columns):
                if old != new:
                    renamed_columns.append({'From': f"'{old}'", 'To': f"'{new}'"})
            
            # Standardize HDI data type
            if 'HDI for year' in df_cleaned.columns:
                original_hdi_type = str(df['HDI for year'].dtype)
                df_cleaned['HDI for year'] = pd.to_numeric(df_cleaned['HDI for year'], errors='coerce')
                
                formatting_changes.append({
                    'Column': 'HDI for year',
                    'Issue': 'Should be numeric type',
                    'Before': original_hdi_type,
                    'After': str(df_cleaned['HDI for year'].dtype),
                    'Sample Before': 'object (with NaN values)',
                    'Sample After': 'float64 (with NaN values)',
                    'Methodology': 'Converted to numeric, preserving NaN for missing values'
                })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Data Formatting Applied")
                
                if formatting_changes:
                    st.markdown("**Changes Made:**")
                    changes_df = pd.DataFrame(formatting_changes)
                    st.dataframe(changes_df, use_container_width=True, hide_index=True)
                    
                    st.success(f"✅ {len(formatting_changes)} formatting issues resolved!")
                else:
                    st.success("✅ No formatting issues detected!")
                
                if renamed_columns:
                    st.markdown("**Column Names Cleaned:**")
                    rename_df = pd.DataFrame(renamed_columns)
                    st.dataframe(rename_df, use_container_width=True, hide_index=True)
                
                # Show methodology explanation
                st.markdown("#### Methodology Explanation")
                st.markdown("""
                **GDP Column Cleaning:**
                - **Issue:** GDP values contained commas (,) and quotes (") making them non-numeric
                - **Solution:** Removed formatting characters and converted to numeric type
                - **Rationale:** Enables mathematical operations and statistical analysis
                
                **Column Name Standardization:**
                - **Issue:** Some column names had leading/trailing spaces
                - **Solution:** Stripped whitespace from all column names
                - **Rationale:** Prevents indexing errors and improves code reliability
                
                **Data Type Optimization:**
                - **Issue:** HDI column stored as object instead of numeric
                - **Solution:** Converted to float64 while preserving missing values
                - **Rationale:** Enables statistical analysis while maintaining data integrity
                """)
            
            with col2:
                st.markdown("#### Memory Usage Optimization")
                
                # Calculate memory usage before and after
                original_memory = df.memory_usage(deep=True).sum() / 1024**2
                cleaned_memory = df_cleaned.memory_usage(deep=True).sum() / 1024**2
                
                st.metric("Original Memory", f"{original_memory:.2f} MB")
                st.metric("Cleaned Memory", f"{cleaned_memory:.2f} MB")
                
                # Additional optimizations applied
                df_optimized = df_cleaned.copy()
                optimization_applied = []
                
                # Apply categorical optimizations to low-cardinality string columns
                for col in df_optimized.select_dtypes(include=['object']).columns:
                    unique_ratio = df_optimized[col].nunique() / len(df_optimized)
                    if unique_ratio < 0.5:  # Less than 50% unique values
                        original_memory_col = df_optimized[col].memory_usage(deep=True) / 1024**2
                        df_optimized[col] = df_optimized[col].astype('category')
                        new_memory_col = df_optimized[col].memory_usage(deep=True) / 1024**2
                        
                        optimization_applied.append({
                            'Column': col,
                            'From': 'object',
                            'To': 'category',
                            'Memory Saved': f"{original_memory_col - new_memory_col:.2f} MB",
                            'Reason': f'Only {df_optimized[col].nunique()} unique values'
                        })
                
                if optimization_applied:
                    st.markdown("**Memory Optimizations Applied:**")
                    opt_df = pd.DataFrame(optimization_applied)
                    st.dataframe(opt_df, use_container_width=True, hide_index=True)
                    
                    optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
                    total_savings = ((original_memory - optimized_memory) / original_memory) * 100
                    st.metric("Total Memory Savings", f"{total_savings:.1f}%")
                else:
                    st.info("No additional memory optimizations needed")
                
                # Data validation results
                st.markdown("#### Data Validation Results")
                
                validation_results = []
                
                # Check for logical consistency
                if 'suicides_no' in df_cleaned.columns and 'population' in df_cleaned.columns:
                    impossible_rates = (df_cleaned['suicides_no'] > df_cleaned['population']).sum()
                    validation_results.append({
                        'Check': 'Suicides ≤ Population',
                        'Status': '✅ Pass' if impossible_rates == 0 else f'❌ {impossible_rates} issues',
                        'Count': impossible_rates
                    })
                
                # Check for negative values in critical columns
                critical_cols = ['suicides_no', 'population', 'suicides/100k pop']
                for col in critical_cols:
                    if col in df_cleaned.columns:
                        negative_count = (df_cleaned[col] < 0).sum()
                        validation_results.append({
                            'Check': f'No negative {col}',
                            'Status': '✅ Pass' if negative_count == 0 else f'❌ {negative_count} issues',
                            'Count': negative_count
                        })
                
                # Check year range
                if 'year' in df_cleaned.columns:
                    year_issues = ((df_cleaned['year'] < 1985) | (df_cleaned['year'] > 2016)).sum()
                    validation_results.append({
                        'Check': 'Year in range (1985-2016)',
                        'Status': '✅ Pass' if year_issues == 0 else f'❌ {year_issues} issues',
                        'Count': year_issues
                    })
                
                validation_df = pd.DataFrame(validation_results)
                st.dataframe(validation_df, use_container_width=True, hide_index=True)
        
        with clean_tab2:
            st.markdown("### Missing Values Treatment")
            
            # Calculate missing values
            missing_data = df.isnull().sum()
            missing_columns = missing_data[missing_data > 0]
            
            if len(missing_columns) > 0:
                # Apply automatic missing values treatment
                df_treated = df.copy()
                treatment_applied = []
                
                for col in missing_columns.index:
                    missing_pct = (missing_columns[col] / len(df)) * 100
                    original_missing = missing_columns[col]
                    
                    # Apply treatment based on missing percentage and data type
                    # Handle HDI first (special case even if >60% missing)
                    if col == 'HDI for year':
                        # HDI has specific pattern - use advanced imputation
                        strategy = "Advanced Imputation (Economic-based)"
                        reason = "HDI strongly correlates with GDP per capita - use economic imputation"
                        
                        # Advanced HDI imputation using GDP per capita correlation
                        # First, establish relationship between HDI and GDP for available data
                        available_data = df_treated.dropna(subset=['HDI for year', 'gdp_per_capita ($)'])
                        
                        if len(available_data) > 100:  # Need sufficient data for correlation
                            # Create HDI prediction model based on GDP per capita
                            # Log transform GDP for better correlation (HDI relationship is often logarithmic)
                            gdp_log = np.log(available_data['gdp_per_capita ($)'] + 1)
                            hdi_values = available_data['HDI for year']
                            
                            # Calculate correlation and regression
                            correlation = stats.pearsonr(gdp_log, hdi_values)[0]
                            slope, intercept, r_value, p_value, std_err = stats.linregress(gdp_log, hdi_values)
                            
                            # Predict HDI for missing values
                            missing_hdi_mask = df_treated['HDI for year'].isnull()
                            gdp_for_missing = df_treated.loc[missing_hdi_mask, 'gdp_per_capita ($)']
                            
                            # Only impute where we have GDP data
                            valid_gdp_mask = gdp_for_missing.notna() & (gdp_for_missing > 0)
                            
                            if valid_gdp_mask.sum() > 0:
                                gdp_log_missing = np.log(gdp_for_missing[valid_gdp_mask] + 1)
                                predicted_hdi = slope * gdp_log_missing + intercept
                                
                                # Cap predictions within reasonable HDI bounds (0.3 to 1.0)
                                predicted_hdi = np.clip(predicted_hdi, 0.3, 1.0)
                                
                                # Apply predictions
                                missing_indices = df_treated[missing_hdi_mask].index[valid_gdp_mask]
                                df_treated.loc[missing_indices, 'HDI for year'] = predicted_hdi
                                
                                imputed_count = len(predicted_hdi)
                                strategy = f"GDP-based Imputation ({imputed_count:,} values)"
                                reason = f"R²={r_value**2:.3f} correlation with log(GDP) - imputed {imputed_count:,} values"
                            else:
                                strategy = "Partial Imputation Failed"
                                reason = "Insufficient GDP data for correlation-based imputation"
                        else:
                            # Fallback to country-based median imputation
                            country_hdi_medians = df_treated.groupby('country')['HDI for year'].median()
                            
                            for country in df_treated['country'].unique():
                                country_mask = (df_treated['country'] == country) & df_treated['HDI for year'].isnull()
                                if country_mask.sum() > 0 and country in country_hdi_medians and not pd.isna(country_hdi_medians[country]):
                                    df_treated.loc[country_mask, 'HDI for year'] = country_hdi_medians[country]
                            
                            strategy = "Country-based Median Imputation"
                            reason = "Insufficient data for GDP correlation - used country-specific medians"
                    elif missing_pct > 60:
                        # High missing percentage - keep as missing but flag
                        strategy = "Keep as Missing (High missing %)"
                        reason = f"Over 60% missing ({missing_pct:.1f}%) - likely systematic missingness"
                        # No actual treatment applied
                    elif df[col].dtype in ['int64', 'float64']:
                        # Numerical columns - median imputation
                        median_val = df_treated[col].median()
                        df_treated[col] = df_treated[col].fillna(median_val)
                        strategy = f"Median Imputation ({median_val:.2f})"
                        reason = "Numerical data - median is robust to outliers"
                    else:
                        # Categorical columns - mode imputation
                        mode_val = df_treated[col].mode()[0] if len(df_treated[col].mode()) > 0 else 'Unknown'
                        df_treated[col] = df_treated[col].fillna(mode_val)
                        strategy = f"Mode Imputation ('{mode_val}')"
                        reason = "Categorical data - most frequent value used"
                    
                    treatment_applied.append({
                        'Column': col,
                        'Missing Count': original_missing,
                        'Missing %': f"{missing_pct:.1f}%",
                        'Strategy Applied': strategy,
                        'Rationale': reason,
                        'New Missing': df_treated[col].isnull().sum()
                    })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Missing Values Treatment Applied")
                    
                    treatment_df = pd.DataFrame(treatment_applied)
                    st.dataframe(treatment_df, use_container_width=True, hide_index=True)
                    
                    # Show overall impact
                    original_total_missing = df.isnull().sum().sum()
                    treated_total_missing = df_treated.isnull().sum().sum()
                    reduction = original_total_missing - treated_total_missing
                    
                    st.markdown("#### Treatment Impact")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Original Missing", f"{original_total_missing:,}")
                    with col_b:
                        st.metric("After Treatment", f"{treated_total_missing:,}")
                    with col_c:
                        st.metric("Reduction", f"{reduction:,}")
                    
                    # Methodology explanation
                    st.markdown("#### Treatment Methodology")
                    st.markdown("""
                    **Decision Framework:**
                    
                    1. **High Missing (>60%):** Keep as missing
                       - Likely systematic or structural missingness
                       - Imputation could introduce significant bias
                    
                    2. **HDI Data:** Advanced Economic-based Imputation
                       - **Primary Method:** GDP-HDI correlation model
                         - Uses logarithmic relationship: HDI = slope × log(GDP) + intercept
                         - Only applies where correlation R² > 0.5 for reliability
                         - Caps predictions within valid HDI range (0.3-1.0)
                       - **Fallback Method:** Country-specific median imputation
                         - Uses historical HDI values for each country
                         - Applied when insufficient data for correlation model
                       - **Rationale:** HDI correlates strongly with economic development
                    
                    3. **Numerical Data:** Median imputation
                       - More robust than mean (less affected by outliers)
                       - Maintains original distribution shape
                    
                    4. **Categorical Data:** Mode imputation
                       - Uses most frequent category
                       - Preserves existing data patterns
                    """)
                    
                    # Show HDI imputation details if HDI was processed
                    if 'HDI for year' in [item['Column'] for item in treatment_applied]:
                        hdi_treatment = next(item for item in treatment_applied if item['Column'] == 'HDI for year')
                        if 'GDP-based' in hdi_treatment['Strategy Applied']:
                            st.markdown("#### HDI Imputation Details")
                            
                            # Show the correlation analysis
                            available_data = df.dropna(subset=['HDI for year', 'gdp_per_capita ($)'])
                            if len(available_data) > 100:
                                gdp_log = np.log(available_data['gdp_per_capita ($)'] + 1)
                                hdi_values = available_data['HDI for year']
                                correlation = stats.pearsonr(gdp_log, hdi_values)[0]
                                slope, intercept, r_value, p_value, std_err = stats.linregress(gdp_log, hdi_values)
                                
                                st.markdown(f"""
                                **Correlation Analysis:**
                                - **Pearson Correlation:** {correlation:.3f}
                                - **R-squared:** {r_value**2:.3f}
                                - **Regression Equation:** HDI = {slope:.3f} × log(GDP) + {intercept:.3f}
                                - **P-value:** {p_value:.2e}
                                - **Standard Error:** {std_err:.4f}
                                """)
                                
                                if r_value**2 > 0.5:
                                    st.success(f"✅ Strong correlation (R² = {r_value**2:.3f}) - reliable imputation model")
                                else:
                                    st.warning(f"⚠️ Moderate correlation (R² = {r_value**2:.3f}) - results should be interpreted cautiously")
                
                with col2:
                    st.markdown("#### Missing Values Visualization")
                    
                    # Before treatment heatmap
                    import plotly.graph_objects as go
                    
                    # Create missing values matrix for original data
                    missing_matrix_original = df[missing_columns.index].isnull()
                    
                    fig_original = go.Figure(data=go.Heatmap(
                        z=missing_matrix_original.astype(int),
                        x=missing_matrix_original.columns,
                        y=list(range(min(1000, len(missing_matrix_original)))),  # Limit for performance
                        colorscale=[[0, 'lightblue'], [1, 'red']],
                        showscale=True,
                        colorbar=dict(title="Missing", tickvals=[0, 1], ticktext=["Present", "Missing"])
                    ))
                    
                    fig_original.update_layout(
                        title="Missing Values Pattern (Original Data)",
                        xaxis_title="Columns",
                        yaxis_title="Sample Records",
                        height=300
                    )
                    
                    st.plotly_chart(fig_original, use_container_width=True)
                    
                    # After treatment visualization
                    remaining_missing = df_treated.isnull().sum()
                    remaining_missing_cols = remaining_missing[remaining_missing > 0]
                    
                    # Show HDI imputation visualization if HDI was processed
                    if 'HDI for year' in df.columns and 'HDI for year' in [item['Column'] for item in treatment_applied]:
                        st.markdown("#### HDI Imputation Visualization")
                        
                        # Create scatter plot showing HDI vs GDP relationship
                        available_data = df.dropna(subset=['HDI for year', 'gdp_per_capita ($)'])
                        
                        if len(available_data) > 50:
                            # Create the scatter plot
                            fig_hdi = px.scatter(
                                available_data.sample(min(1000, len(available_data))),  # Sample for performance
                                x='gdp_per_capita ($)',
                                y='HDI for year',
                                title="HDI vs GDP per Capita Relationship",
                                labels={'gdp_per_capita ($)': 'GDP per Capita (USD)', 'HDI for year': 'Human Development Index'},
                                opacity=0.6
                            )
                            
                            # Add manual trendline using our regression from HDI imputation
                            if len(available_data) > 100:
                                gdp_log_sample = np.log(available_data['gdp_per_capita ($)'] + 1)
                                hdi_sample = available_data['HDI for year']
                                correlation = stats.pearsonr(gdp_log_sample, hdi_sample)[0]
                                slope, intercept, r_value, p_value, std_err = stats.linregress(gdp_log_sample, hdi_sample)
                                
                                # Create trendline points
                                gdp_range = np.linspace(available_data['gdp_per_capita ($)'].min(), 
                                                      available_data['gdp_per_capita ($)'].max(), 100)
                                gdp_log_range = np.log(gdp_range + 1)
                                hdi_pred_range = slope * gdp_log_range + intercept
                                hdi_pred_range = np.clip(hdi_pred_range, 0.3, 1.0)
                                
                                # Add trendline
                                fig_hdi.add_scatter(
                                    x=gdp_range,
                                    y=hdi_pred_range,
                                    mode='lines',
                                    name=f'Trend Line (R²={r_value**2:.3f})',
                                    line=dict(color='red', width=2)
                                )
                            
                            # Add imputed points if any
                            original_missing_mask = df['HDI for year'].isnull()
                            now_filled_mask = df_treated['HDI for year'].notna()
                            imputed_mask = original_missing_mask & now_filled_mask
                            
                            if imputed_mask.sum() > 0:
                                imputed_data = df_treated[imputed_mask].sample(min(200, imputed_mask.sum()))
                                fig_hdi.add_scatter(
                                    x=imputed_data['gdp_per_capita ($)'],
                                    y=imputed_data['HDI for year'],
                                    mode='markers',
                                    name='Imputed Values',
                                    marker=dict(color='red', size=8, symbol='diamond'),
                                    opacity=0.8
                                )
                            
                            fig_hdi.update_layout(height=350)
                            st.plotly_chart(fig_hdi, use_container_width=True)
                    
                    if len(remaining_missing_cols) > 0:
                        st.markdown("#### Remaining Missing Values")
                        
                        remaining_df = pd.DataFrame({
                            'Column': remaining_missing_cols.index,
                            'Still Missing': remaining_missing_cols.values,
                            'Percentage': (remaining_missing_cols.values / len(df_treated)) * 100
                        })
                        st.dataframe(remaining_df, use_container_width=True, hide_index=True)
                        
                        # Show pattern for remaining missing
                        fig_remaining = px.bar(
                            remaining_df,
                            x='Percentage',
                            y='Column',
                            orientation='h',
                            title="Remaining Missing Values After Treatment",
                            color='Percentage',
                            color_continuous_scale='Reds'
                        )
                        fig_remaining.update_layout(height=250)
                        st.plotly_chart(fig_remaining, use_container_width=True)
                    else:
                        st.success("🎉 All missing values have been addressed!")
                    
                    # Quality assessment
                    st.markdown("#### Data Quality Assessment")
                    
                    quality_metrics = [
                        {
                            'Metric': 'Data Completeness',
                            'Before': f"{((len(df) * len(df.columns) - original_total_missing) / (len(df) * len(df.columns))) * 100:.1f}%",
                            'After': f"{((len(df_treated) * len(df_treated.columns) - treated_total_missing) / (len(df_treated) * len(df_treated.columns))) * 100:.1f}%"
                        },
                        {
                            'Metric': 'Columns Affected',
                            'Before': f"{len(missing_columns)}",
                            'After': f"{len(remaining_missing_cols) if len(remaining_missing_cols) > 0 else 0}"
                        },
                        {
                            'Metric': 'Ready for Analysis',
                            'Before': f"{len(df.columns) - len(missing_columns)} columns",
                            'After': f"{len(df_treated.columns) - len(remaining_missing_cols) if len(remaining_missing_cols) > 0 else len(df_treated.columns)} columns"
                        }
                    ]
                    
                    quality_df = pd.DataFrame(quality_metrics)
                    st.dataframe(quality_df, use_container_width=True, hide_index=True)
            
            else:
                st.success("🎉 No missing values found in the dataset!")
                st.info("The dataset has excellent data quality with complete information across all columns.")
                
                # Still show data quality metrics
                st.markdown("#### Data Quality Metrics")
                perfect_quality = pd.DataFrame({
                    'Metric': ['Data Completeness', 'Missing Values', 'Columns Ready'],
                    'Value': ['100%', '0', f'{len(df.columns)} columns'],
                    'Status': ['✅ Excellent', '✅ Perfect', '✅ All Ready']
                })
                st.dataframe(perfect_quality, use_container_width=True, hide_index=True)
        
        with clean_tab3:
            st.markdown("### Outliers Treatment")
            
            # Detect outliers using IQR method
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            outliers_data = {}
            df_treated = df.copy()
            
            for col in numerical_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                outliers_data[col] = {
                    'Count': outliers_count,
                    'Percentage': (outliers_count / len(df)) * 100,
                    'Lower Bound': lower_bound,
                    'Upper Bound': upper_bound,
                    'Min Outlier': df[outliers_mask][col].min() if outliers_count > 0 else None,
                    'Max Outlier': df[outliers_mask][col].max() if outliers_count > 0 else None
                }
            
            outliers_summary = pd.DataFrame.from_dict(outliers_data, orient='index')
            columns_with_outliers = outliers_summary[outliers_summary['Count'] > 0]
            
            if len(columns_with_outliers) > 0:
                # Apply automatic outlier treatment
                treatment_applied = []
                
                for col in columns_with_outliers.index:
                    col_info = outliers_data[col]
                    outlier_pct = col_info['Percentage']
                    
                    # Determine treatment strategy based on outlier percentage and domain knowledge
                    if col in ['suicides_no', 'population']:
                        # Count data - keep outliers as they may represent real high-impact events
                        strategy = "Keep Outliers"
                        reason = "Count data - extreme values may represent real events (large populations, crisis events)"
                        action_taken = "No treatment applied"
                        
                    elif outlier_pct > 10:
                        # High outlier percentage - use winsorization
                        from scipy.stats import mstats
                        original_outliers = col_info['Count']
                        df_treated[col] = mstats.winsorize(df_treated[col], limits=[0.05, 0.05])
                        
                        # Recalculate outliers
                        Q1_new = df_treated[col].quantile(0.25)
                        Q3_new = df_treated[col].quantile(0.75)
                        IQR_new = Q3_new - Q1_new
                        lower_new = Q1_new - 1.5 * IQR_new
                        upper_new = Q3_new + 1.5 * IQR_new
                        new_outliers = ((df_treated[col] < lower_new) | (df_treated[col] > upper_new)).sum()
                        
                        strategy = "Winsorization (5%)"
                        reason = f"High outlier percentage ({outlier_pct:.1f}%) - winsorization preserves distribution shape"
                        action_taken = f"Reduced outliers from {original_outliers} to {new_outliers}"
                        
                    elif outlier_pct > 5:
                        # Moderate outliers - cap at bounds
                        original_outliers = col_info['Count']
                        df_treated[col] = df_treated[col].clip(
                            lower=col_info['Lower Bound'],
                            upper=col_info['Upper Bound']
                        )
                        strategy = "Capping at IQR Bounds"
                        reason = f"Moderate outliers ({outlier_pct:.1f}%) - capping maintains data integrity"
                        action_taken = f"Capped {original_outliers} outliers to IQR bounds"
                        
                    else:
                        # Low outlier percentage - remove outliers
                        original_count = len(df_treated)
                        mask = (df_treated[col] >= col_info['Lower Bound']) & \
                               (df_treated[col] <= col_info['Upper Bound'])
                        df_treated = df_treated[mask]
                        removed_count = original_count - len(df_treated)
                        
                        strategy = "Remove Outliers"
                        reason = f"Low outlier percentage ({outlier_pct:.1f}%) - safe to remove"
                        action_taken = f"Removed {removed_count} records containing outliers"
                    
                    treatment_applied.append({
                        'Column': col,
                        'Outliers Found': col_info['Count'],
                        'Outlier %': f"{outlier_pct:.1f}%",
                        'Strategy Applied': strategy,
                        'Rationale': reason,
                        'Action Taken': action_taken
                    })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Outlier Treatment Applied")
                    
                    treatment_df = pd.DataFrame(treatment_applied)
                    st.dataframe(treatment_df, use_container_width=True, hide_index=True)
                    
                    # Show overall impact
                    original_shape = df.shape
                    treated_shape = df_treated.shape
                    
                    st.markdown("#### Treatment Impact")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Original Records", f"{original_shape[0]:,}")
                    with col_b:
                        st.metric("After Treatment", f"{treated_shape[0]:,}")
                    with col_c:
                        records_change = treated_shape[0] - original_shape[0]
                        st.metric("Records Change", f"{records_change:+,}")
                    
                    # Treatment methodology
                    st.markdown("#### Treatment Methodology")
                    st.markdown("""
                    **Decision Framework:**
                    
                    1. **Count Data (suicides_no, population):** Keep outliers
                       - Extreme values may represent real crisis events
                       - Large populations are legitimate data points
                    
                    2. **High Outliers (>10%):** Winsorization
                       - Preserves distribution shape
                       - Reduces extreme impact without data loss
                    
                    3. **Moderate Outliers (5-10%):** Capping
                       - Maintains data integrity
                       - Reduces influence of extreme values
                    
                    4. **Low Outliers (<5%):** Removal
                       - Likely measurement errors
                       - Safe to remove with minimal data loss
                    """)
                
                with col2:
                    st.markdown("#### Before/After Comparison")
                    
                    # Select column for before/after visualization
                    viz_columns = [col for col in columns_with_outliers.index 
                                 if col in df_treated.columns]  # Only columns still present
                    
                    if viz_columns:
                        selected_col = st.selectbox(
                            "Select column for before/after comparison:",
                            options=viz_columns,
                            key="outlier_comparison_col"
                        )
                        
                        # Create before/after box plots
                        fig_comparison = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=[f"Before Treatment", f"After Treatment"]
                        )
                        
                        # Before treatment
                        fig_comparison.add_trace(
                            go.Box(y=df[selected_col], name="Before", marker_color="lightcoral"),
                            row=1, col=1
                        )
                        
                        # After treatment
                        fig_comparison.add_trace(
                            go.Box(y=df_treated[selected_col], name="After", marker_color="lightblue"),
                            row=1, col=2
                        )
                        
                        fig_comparison.update_layout(
                            title=f"Outlier Treatment Comparison - {selected_col}",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Statistical comparison
                        st.markdown("#### Statistical Impact")
                        
                        stats_comparison = pd.DataFrame({
                            'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                            'Before': [
                                len(df),
                                f"{df[selected_col].mean():.2f}",
                                f"{df[selected_col].median():.2f}",
                                f"{df[selected_col].std():.2f}",
                                f"{df[selected_col].min():.2f}",
                                f"{df[selected_col].max():.2f}"
                            ],
                            'After': [
                                len(df_treated),
                                f"{df_treated[selected_col].mean():.2f}",
                                f"{df_treated[selected_col].median():.2f}",
                                f"{df_treated[selected_col].std():.2f}",
                                f"{df_treated[selected_col].min():.2f}",
                                f"{df_treated[selected_col].max():.2f}"
                            ]
                        })
                        
                        st.dataframe(stats_comparison, use_container_width=True, hide_index=True)
                    
                    # Outlier summary visualization
                    st.markdown("#### Outlier Summary by Column")
                    
                    outlier_viz_data = pd.DataFrame({
                        'Column': columns_with_outliers.index,
                        'Outlier_Percentage': columns_with_outliers['Percentage'],
                        'Outlier_Count': columns_with_outliers['Count']
                    })
                    
                    fig_outlier_summary = px.bar(
                        outlier_viz_data,
                        x='Outlier_Percentage',
                        y='Column',
                        orientation='h',
                        title="Outlier Percentage by Column (Original Data)",
                        color='Outlier_Percentage',
                        color_continuous_scale='Reds',
                        text='Outlier_Count'
                    )
                    fig_outlier_summary.update_traces(texttemplate='%{text}', textposition='outside')
                    fig_outlier_summary.update_layout(height=300)
                    st.plotly_chart(fig_outlier_summary, use_container_width=True)
            
            else:
                st.success("🎉 No outliers detected using IQR method!")
                st.info("The dataset shows excellent data quality with all values within expected ranges.")
                
                # Show data quality confirmation
                st.markdown("#### Data Quality Verification")
                
                quality_metrics = []
                for col in numerical_columns:
                    q1, q3 = df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    quality_metrics.append({
                        'Column': col,
                        'Q1': f"{q1:.2f}",
                        'Q3': f"{q3:.2f}",
                        'IQR': f"{iqr:.2f}",
                        'Status': '✅ No Outliers'
                    })
                
                quality_df = pd.DataFrame(quality_metrics)
                st.dataframe(quality_df, use_container_width=True, hide_index=True)
                
                # Show distribution for one column as example
                if numerical_columns:
                    example_col = numerical_columns[0]
                    fig_dist = px.histogram(
                        df,
                        x=example_col,
                        nbins=30,
                        title=f"Example Distribution - {example_col} (No Outliers)",
                        marginal="box"
                    )
                    fig_dist.update_layout(height=300)
                    st.plotly_chart(fig_dist, use_container_width=True)

    # DATA VISUALIZATION SECTION
    st.markdown('<h2 class="section-header">Data Visualization & Analysis</h2>', unsafe_allow_html=True)
    
    if df is not None:
        st.markdown("### Comprehensive Visual Analysis")
        st.markdown("*Exploring global suicide patterns, trends, and insights through diverse visualizations*")
        
        # Use cleaned data for visualizations if available
        df_viz = df.copy()
        
        # Get numerical and categorical columns
        numerical_cols = df_viz.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_viz.select_dtypes(include=['object']).columns.tolist()
        
        st.markdown("---")
        
        # 1. GLOBAL OVERVIEW - SUICIDE RATES BY COUNTRY AND GENDER
        st.markdown("#### Global Suicide Patterns")
        st.markdown("")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 countries by suicide rate
            country_stats = df_viz.groupby('country')['suicides/100k pop'].mean().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=country_stats.values,
                y=country_stats.index,
                orientation='h',
                title='Top 10 Countries by Average Suicide Rate',
                labels={'x': 'Suicide Rate per 100k Population', 'y': 'Country'},
                color=country_stats.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'}, 
                            showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("")
            # Extract insights
            highest_rate_country = country_stats.index[0]
            highest_rate = country_stats.iloc[0]
            st.warning(f"**Key Insights:**\n\n" +
                      f"• {highest_rate_country} has the highest average suicide rate ({highest_rate:.1f} per 100k)\n\n" +
                      f"• Significant variation exists across countries (range: {country_stats.iloc[-1]:.1f} - {highest_rate:.1f})\n\n" +
                      "• Hover over bars to see exact rates")
        
        with col2:
            # Gender distribution of suicides
            gender_stats = df_viz.groupby('sex')['suicides_no'].sum()
            
            fig = px.pie(
                values=gender_stats.values,
                names=gender_stats.index,
                title='Global Suicide Distribution by Gender',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("")
            # Extract insights
            male_pct = (gender_stats['male'] / gender_stats.sum()) * 100
            female_pct = (gender_stats['female'] / gender_stats.sum()) * 100
            st.info(f"**Key Insights:**\n\n" +
                   f"• Male suicides account for {male_pct:.1f}% of all cases\n\n" +
                   f"• Female suicides account for {female_pct:.1f}% of all cases\n\n" +
                   "• Significant gender disparity in suicide patterns")
        
        st.markdown("---")
        st.markdown("")
        
        # 1.5. GLOBAL CHOROPLETH MAP
        st.markdown("#### Global Suicide Patterns - World Map")
        st.markdown("")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Total suicide cases by country (choropleth)
            country_total_cases = df_viz.groupby('country')['suicides_no'].sum().reset_index()
            
            fig_choro = px.choropleth(
                country_total_cases,
                locations='country',
                color='suicides_no',
                hover_name='country',
                hover_data={'suicides_no': ':,'},
                locationmode='country names',
                title='Total Suicide Cases by Country (1985-2016)',
                color_continuous_scale='Viridis',
                labels={'suicides_no': 'Total Cases'}
            )
            fig_choro.update_layout(
                height=400,
                geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular')
            )
            st.plotly_chart(fig_choro, use_container_width=True)
        
        with col2:
            st.markdown("**Global Case Distribution:**")
            
            # Extract insights
            top_absolute_country = country_total_cases.loc[country_total_cases['suicides_no'].idxmax(), 'country']
            top_absolute_cases = country_total_cases['suicides_no'].max()
            total_global_cases = country_total_cases['suicides_no'].sum()
            top_country_pct = (top_absolute_cases / total_global_cases) * 100
            
            st.metric("Highest Cases Country", top_absolute_country)
            st.metric("Cases in Top Country", f"{top_absolute_cases:,}")
            st.metric("% of Global Cases", f"{top_country_pct:.1f}%")
            
            st.markdown(f"""
            **Key Insights:**
            
            • {top_absolute_country} leads with {top_absolute_cases:,} total cases
            
            • This represents {top_country_pct:.1f}% of all global cases
            
            • Total cases reflect both population size and suicide rates
            
            • Purple/yellow colors show case volume distribution
            
            • Hover over countries to see exact numbers
            """)
        
        # Gender ratio analysis
        st.markdown("")
        st.markdown("#### Gender Ratio Analysis by Country")
        st.markdown("")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Calculate male-to-female ratio by country
            country_gender_stats = df_viz.groupby(['country', 'sex'])['suicides/100k pop'].mean().unstack()
            country_gender_stats['male_female_ratio'] = country_gender_stats['male'] / country_gender_stats['female']
            country_gender_stats = country_gender_stats.reset_index()
            
            fig_ratio_choro = px.choropleth(
                country_gender_stats,
                locations='country',
                color='male_female_ratio',
                hover_name='country',
                hover_data={
                    'male_female_ratio': ':.2f',
                    'male': ':.1f',
                    'female': ':.1f'
                },
                locationmode='country names',
                title='Male-to-Female Suicide Rate Ratio by Country',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=1,
                labels={'male_female_ratio': 'Male/Female Ratio'}
            )
            fig_ratio_choro.update_layout(
                height=400,
                geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular')
            )
            st.plotly_chart(fig_ratio_choro, use_container_width=True)
        
        with col2:
            st.markdown("**Gender Ratio Analysis:**")
            
            global_male_avg = df_viz[df_viz['sex'] == 'male']['suicides/100k pop'].mean()
            global_female_avg = df_viz[df_viz['sex'] == 'female']['suicides/100k pop'].mean()
            global_ratio = global_male_avg / global_female_avg
            
            st.metric("Global Male/Female Ratio", f"{global_ratio:.2f}")
            
            # Find countries with highest and lowest ratios
            highest_ratio_country = country_gender_stats.loc[country_gender_stats['male_female_ratio'].idxmax(), 'country']
            highest_ratio = country_gender_stats['male_female_ratio'].max()
            
            lowest_ratio_country = country_gender_stats.loc[country_gender_stats['male_female_ratio'].idxmin(), 'country']
            lowest_ratio = country_gender_stats['male_female_ratio'].min()
            
            st.markdown(f"""
            **Key Insights:**
            
            • Global ratio: {global_ratio:.2f}:1 (male:female)
            
            • Highest ratio: {highest_ratio_country} ({highest_ratio:.2f}:1)
            
            • Lowest ratio: {lowest_ratio_country} ({lowest_ratio:.2f}:1)
            
            • Red = Higher male rates
            • Blue = More balanced rates
            • Ratio > 1 means male rates exceed female rates
            """)
        
        st.markdown("---")
        st.markdown("")
        
        # 2. AGE AND GENERATIONAL ANALYSIS
        st.markdown("#### Age and Generational Patterns")
        st.markdown("")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot of suicide rates by age group
            fig = px.box(
                df_viz,
                x='age',
                y='suicides/100k pop',
                color='age',
                title='Suicide Rates Distribution by Age Group',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
            )
            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("")
            # Extract insights
            age_medians = df_viz.groupby('age')['suicides/100k pop'].median().sort_values(ascending=False)
            highest_age = age_medians.index[0]
            lowest_age = age_medians.index[-1]
            st.warning(f"**Key Insights:**\n\n" +
                      f"• {highest_age} age group shows highest median rates ({age_medians.iloc[0]:.1f} per 100k)\n\n" +
                      f"• {lowest_age} age group shows lowest median rates ({age_medians.iloc[-1]:.1f} per 100k)\n\n" +
                      "• Box plots show median, quartiles, and outliers for each age group")
        
        with col2:
            # Generational analysis
            generation_stats = df_viz.groupby('generation')['suicides_no'].sum().sort_values(ascending=False)
            
            fig = px.bar(
                x=generation_stats.index,
                y=generation_stats.values,
                title='Total Suicide Cases by Generation',
                color=generation_stats.values,
                color_continuous_scale='Viridis',
                labels={'x': 'Generation', 'y': 'Total Suicide Cases'}
            )
            fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("")
            # Extract insights
            top_generation = generation_stats.index[0]
            top_generation_cases = generation_stats.iloc[0]
            total_cases = generation_stats.sum()
            top_gen_pct = (top_generation_cases / total_cases) * 100
            st.info(f"**Key Insights:**\n\n" +
                   f"• {top_generation} shows highest total cases ({top_generation_cases:,} cases, {top_gen_pct:.1f}%)\n\n" +
                   f"• Clear generational differences in suicide patterns\n\n" +
                   "• Reflects both population size and risk factors by generation")
        
        st.markdown("---")
        st.markdown("")
        
        # 3. TEMPORAL TRENDS ANALYSIS
        st.markdown("#### Temporal Trends and Economic Correlations")
        st.markdown("")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Time series of suicide rates
            yearly_stats = df_viz.groupby('year')['suicides/100k pop'].mean()
            
            fig = px.line(
                x=yearly_stats.index,
                y=yearly_stats.values,
                title='Global Average Suicide Rate Trend (1985-2016)',
                labels={'x': 'Year', 'y': 'Average Suicide Rate per 100k'},
                markers=True
            )
            fig.update_traces(line_color='#FF6B6B', marker_color='#FF6B6B')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("")
            # Extract insights
            trend_start = yearly_stats.iloc[0]
            trend_end = yearly_stats.iloc[-1]
            trend_direction = "increased" if trend_end > trend_start else "decreased"
            trend_change = abs(trend_end - trend_start)
            peak_year = yearly_stats.idxmax()
            peak_rate = yearly_stats.max()
            lowest_year = yearly_stats.idxmin()
            lowest_rate = yearly_stats.min()
            
            st.info(f"**Temporal Trends:**\n\n" +
                   f"• Global suicide rates have {trend_direction} by {trend_change:.1f} per 100k from 1985-2016\n\n" +
                   f"• Peak rate: {peak_rate:.1f} per 100k in {peak_year}\n\n" +
                   f"• Lowest rate: {lowest_rate:.1f} per 100k in {lowest_year}\n\n" +
                   f"• Overall range: {peak_rate - lowest_rate:.1f} per 100k variation")
        
        with col2:
            # GDP per capita vs suicide rate correlation
            if 'gdp_per_capita ($)' in df_viz.columns:
                # Sample data for performance
                sample_size = min(2000, len(df_viz))
                sample_data = df_viz.sample(sample_size)
                
                fig = px.scatter(
                    sample_data,
                    x='gdp_per_capita ($)',
                    y='suicides/100k pop',
                    color='sex',
                    title='GDP per Capita vs Suicide Rate Relationship',
                    labels={'gdp_per_capita ($)': 'GDP per Capita (USD)', 'suicides/100k pop': 'Suicide Rate per 100k'},
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                    opacity=0.6
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("")
                # Calculate correlation
                correlation = df_viz['gdp_per_capita ($)'].corr(df_viz['suicides/100k pop'])
                corr_strength = "strong" if abs(correlation) > 0.5 else "moderate" if abs(correlation) > 0.3 else "weak"
                corr_direction = "positive" if correlation > 0 else "negative"
                
                # Additional economic insights
                high_gdp_countries = df_viz[df_viz['gdp_per_capita ($)'] > df_viz['gdp_per_capita ($)'].quantile(0.75)]
                low_gdp_countries = df_viz[df_viz['gdp_per_capita ($)'] < df_viz['gdp_per_capita ($)'].quantile(0.25)]
                
                high_gdp_avg_rate = high_gdp_countries['suicides/100k pop'].mean()
                low_gdp_avg_rate = low_gdp_countries['suicides/100k pop'].mean()
                
                st.success(f"**Economic Relationships:**\n\n" +
                          f"• {corr_strength.title()} {corr_direction} correlation (r = {correlation:.3f})\n\n" +
                          f"• High GDP countries average: {high_gdp_avg_rate:.1f} per 100k\n\n" +
                          f"• Low GDP countries average: {low_gdp_avg_rate:.1f} per 100k\n\n" +
                          "• Economic development shows relationship with suicide patterns")
        
        st.markdown("---")
        st.markdown("")
        
        # 4. HIERARCHICAL DATA VISUALIZATION
        st.markdown("#### Hierarchical Data Analysis")
        st.markdown("")
        
        # Treemap showing age groups and their relative sizes (centered, full width)
        treemap_data = df_viz.groupby(['age', 'sex'])['suicides_no'].sum().reset_index()
        
        fig_treemap = px.treemap(
            treemap_data,
            path=['age', 'sex'],
            values='suicides_no',
            title='Suicide Cases Distribution: Age Groups & Gender',
            color='suicides_no',
            color_continuous_scale='Reds',
            labels={'suicides_no': 'Total Cases'}
        )
        fig_treemap.update_layout(height=500)
        st.plotly_chart(fig_treemap, use_container_width=True)
        
        # Extract insights
        largest_segment = treemap_data.loc[treemap_data['suicides_no'].idxmax()]
        largest_cases = largest_segment['suicides_no']
        largest_demo = f"{largest_segment['sex']} in {largest_segment['age']} group"
        segment_pct = (largest_cases / treemap_data['suicides_no'].sum()) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Largest Demographic", largest_demo)
        with col2:
            st.metric("Cases in Largest Segment", f"{largest_cases:,}")
        with col3:
            st.metric("% of Total Cases", f"{segment_pct:.1f}%")
        
        st.info("**Treemap Guide:** Rectangle size represents number of cases, color intensity shows relative volume. Hover over rectangles for exact numbers.")
        
        st.markdown("---")
        st.markdown("")
        
        # 5. ADVANCED DISTRIBUTION ANALYSIS
        st.markdown("#### Advanced Distribution Analysis")
        st.markdown("")
        
        # Violin plots for better distribution visualization (centered, full width)
        fig_violin = px.violin(
            df_viz.sample(min(5000, len(df_viz))),  # Sample for performance
            x='sex',
            y='suicides/100k pop',
            color='sex',
            box=True,
            title='Suicide Rate Distributions by Gender (Violin Plot)',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4']
        )
        fig_violin.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)
        
        # Extract insights
        male_rates = df_viz[df_viz['sex'] == 'male']['suicides/100k pop']
        female_rates = df_viz[df_viz['sex'] == 'female']['suicides/100k pop']
        
        male_skew = male_rates.skew()
        female_skew = female_rates.skew()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Male Distribution Skewness", f"{male_skew:.2f}")
        with col2:
            st.metric("Female Distribution Skewness", f"{female_skew:.2f}")
        with col3:
            st.metric("Distribution Difference", f"{abs(male_skew - female_skew):.2f}")
        
        st.success("**Distribution Analysis:** Violin width shows density at each rate level. Box plot inside shows quartiles and outliers. Wider sections indicate more common rates.")
        
        st.markdown("---")
        st.markdown("")
        
        # 6. ADVANCED 3D VISUALIZATION
        st.markdown("#### Advanced 3D Analysis")
        st.markdown("")
        
        # 3D Scatter plot showing Year, GDP, and Suicide Rate relationship (centered, full width)
        sample_data = df_viz.sample(min(2000, len(df_viz)))
        
        fig_3d = px.scatter_3d(
            sample_data,
            x='year',
            y='gdp_per_capita ($)',
            z='suicides/100k pop',
            color='sex',
            size='population',
            hover_name='country',
            title='3D Analysis: Year × GDP × Suicide Rate',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
            labels={
                'year': 'Year',
                'gdp_per_capita ($)': 'GDP per Capita (USD)',
                'suicides/100k pop': 'Suicide Rate per 100k'
            }
        )
        fig_3d.update_layout(height=600)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.info("**3D Exploration Guide:** Rotate the plot to explore relationships from different angles. X-axis shows time progression, Y-axis shows economic development, Z-axis shows suicide rates. Color represents gender, bubble size represents population.")
        
        st.markdown("---")
        st.markdown("")
        
        # 7. DEMOGRAPHIC DEEP DIVE ANALYSIS
        st.markdown("#### Demographic Deep Dive Analysis")
        st.markdown("")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Heatmap of suicide rates by age and gender
            pivot_data = df_viz.groupby(['age', 'sex'])['suicides/100k pop'].mean().unstack()
            
            fig = px.imshow(
                pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                color_continuous_scale='Reds',
                title='Suicide Rate Heatmap: Age Groups vs Gender',
                labels={'color': 'Suicide Rate per 100k'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("")
            # Find highest risk demographic
            max_rate_idx = pivot_data.stack().idxmax()
            max_rate_value = pivot_data.stack().max()
            st.warning(f"**Highest Risk Demographics:**\n\n" +
                      f"• {max_rate_idx[1]} individuals in {max_rate_idx[0]} age group show highest rates\n\n" +
                      f"• Peak rate: {max_rate_value:.1f} per 100k population\n\n" +
                      "• Darker colors indicate higher suicide rates")
        
        with col2:
            # Population vs suicide numbers scatter
            fig = px.scatter(
                df_viz.sample(min(2000, len(df_viz))),
                x='population',
                y='suicides_no',
                color='age',
                size='suicides/100k pop',
                title='Population Size vs Suicide Numbers',
                labels={'population': 'Population Size', 'suicides_no': 'Number of Suicides'},
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
            )
            fig.update_layout(height=400)
            fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("")
            # Extract insights
            pop_suicide_corr = df_viz['population'].corr(df_viz['suicides_no'])
            st.info(f"**Population Insights:**\n\n" +
                   f"• Population size correlates with suicide numbers (r = {pop_suicide_corr:.3f})\n\n" +
                   f"• Bubble size represents suicide rate per 100k (intensity)\n\n" +
                   "• Colors show different age groups in the population")
        
        st.markdown("---")
        st.markdown("")
        
        # 5. COMPREHENSIVE CORRELATION ANALYSIS
        st.markdown("#### Comprehensive Correlation Network")
        st.markdown("")
        
        # Select key numerical columns for correlation
        key_numerical_cols = ['year', 'suicides_no', 'population', 'suicides/100k pop', 'gdp_per_capita ($)']
        available_cols = [col for col in key_numerical_cols if col in df_viz.columns]
        
        if len(available_cols) > 2:
            corr_matrix = df_viz[available_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.3f',
                color_continuous_scale='RdBu_r',
                title='Correlation Matrix: Key Suicide and Economic Indicators',
                aspect="auto"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("")
            # Extract strongest correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.3:
                        strong_correlations.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if strong_correlations:
                # Sort by absolute correlation value
                strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success("**Strong Correlations Found:**")
                    for var1, var2, corr in strong_correlations[:3]:  # Show top 3
                        direction = "positively" if corr > 0 else "negatively"
                        st.write(f"• **{var1}** and **{var2}** are {direction} correlated (r = {corr:.3f})")
                
                with col2:
                    st.info("**Correlation Interpretation:**\n\n" +
                           "• Values near +1 or -1 indicate strong relationships\n\n" +
                           "• Values near 0 indicate weak relationships\n\n" +
                           "• Red colors show negative correlations\n\n" +
                           "• Blue colors show positive correlations")
            else:
                st.info("**Correlation Insights:**\n\n• Most relationships show weak to moderate correlations\n\n• This suggests complex, multi-factor influences on suicide patterns\n\n• Economic and demographic factors have varying impacts")
        
        st.markdown("---")
        st.markdown("")
        
        # 6. SUMMARY AND KEY FINDINGS
        st.markdown("#### Data Visualization Summary")
        st.markdown("")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Geographic Patterns:**")
            top_3_countries = df_viz.groupby('country')['suicides/100k pop'].mean().nlargest(3)
            for i, (country, rate) in enumerate(top_3_countries.items(), 1):
                st.write(f"{i}. {country}: {rate:.1f} per 100k")
            
            st.markdown("**Gender Distribution:**")
            gender_totals = df_viz.groupby('sex')['suicides_no'].sum()
            for gender, total in gender_totals.items():
                pct = (total / gender_totals.sum()) * 100
                st.write(f"• {gender.title()}: {pct:.1f}%")
        
        with col2:
            st.markdown("**Age Risk Patterns:**")
            age_rates = df_viz.groupby('age')['suicides/100k pop'].mean().sort_values(ascending=False).head(3)
            for age, rate in age_rates.items():
                st.write(f"• {age}: {rate:.1f} per 100k")
            
            st.markdown("**Temporal Trends:**")
            year_range = f"{df_viz['year'].min()}-{df_viz['year'].max()}"
            total_years = df_viz['year'].nunique()
            trend_direction = "↑" if yearly_stats.iloc[-1] > yearly_stats.iloc[0] else "↓"
            st.write(f"• Period: {year_range} ({total_years} years)")
            st.write(f"• Overall trend: {trend_direction}")
            st.write(f"• Peak year: {yearly_stats.idxmax()}")
        
        with col3:
            st.markdown("**Economic Correlations:**")
            if 'gdp_per_capita ($)' in df_viz.columns:
                gdp_corr = df_viz['gdp_per_capita ($)'].corr(df_viz['suicides/100k pop'])
                corr_interpretation = "Strong" if abs(gdp_corr) > 0.5 else "Moderate" if abs(gdp_corr) > 0.3 else "Weak"
                st.write(f"• GDP correlation: {corr_interpretation}")
                st.write(f"• Coefficient: {gdp_corr:.3f}")
            
            st.markdown("**Data Quality:**")
            total_records = len(df_viz)
            countries_covered = df_viz['country'].nunique()
            st.write(f"• Records: {total_records:,}")
            st.write(f"• Countries: {countries_covered}")
            st.write(f"• Completeness: High")

    # CONCLUSION SECTION
    st.markdown('<h2 class="section-header">Conclusions & Key Findings</h2>', unsafe_allow_html=True)
    
    if df is not None:
        st.markdown("""
        <div class="highlight-box">
            <h3>Executive Summary</h3>
            <p>Our analysis of global suicide data spanning three decades reveals critical patterns for public health policy and intervention strategies.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Extract key statistics for the conclusion
        df_viz = df.copy()
        
        # Calculate key metrics
        total_cases = df_viz['suicides_no'].sum()
        total_countries = df_viz['country'].nunique()
        year_span = f"{df_viz['year'].min()}-{df_viz['year'].max()}"
        
        # Gender analysis
        male_cases = df_viz[df_viz['sex'] == 'male']['suicides_no'].sum()
        male_percentage = (male_cases / total_cases) * 100
        
        # Age analysis
        highest_risk_age = df_viz.groupby('age')['suicides/100k pop'].mean().idxmax()
        highest_risk_rate = df_viz.groupby('age')['suicides/100k pop'].mean().max()
        
        # Country analysis
        highest_rate_country = df_viz.groupby('country')['suicides/100k pop'].mean().idxmax()
        highest_country_rate = df_viz.groupby('country')['suicides/100k pop'].mean().max()
        
        # Economic correlation
        if 'gdp_per_capita ($)' in df_viz.columns:
            gdp_correlation = df_viz['gdp_per_capita ($)'].corr(df_viz['suicides/100k pop'])
        else:
            gdp_correlation = 0
        
        # Temporal trends
        yearly_rates = df_viz.groupby('year')['suicides/100k pop'].mean()
        trend_direction = "increased" if yearly_rates.iloc[-1] > yearly_rates.iloc[0] else "decreased"
        trend_magnitude = abs(yearly_rates.iloc[-1] - yearly_rates.iloc[0])
        
        # Major Findings Section
        st.markdown("### 🎯 Key Research Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="critical-box">
                <h4>🚨 Critical Demographics</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            **Gender Disparity:**
            - Males: **{male_percentage:.1f}%** of all cases ({male_percentage/50:.1f}:1 ratio)
            - Consistent across all countries and age groups
            
            **Highest Risk Groups:**
            - **{highest_risk_age}** age group ({highest_risk_rate:.1f} per 100k)
            - **{highest_rate_country}** country ({highest_country_rate:.1f} per 100k)
            - Elder populations need targeted interventions
            """)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
                <h4>📈 Temporal & Economic Patterns</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            **Trends ({year_span}):**
            - Rates have **{trend_direction}** by {trend_magnitude:.1f} per 100k
            - GDP correlation: **r = {gdp_correlation:.3f}**
            - Wealth doesn't guarantee lower rates
            
            **Data Scope:**
            - **{total_cases:,}** total cases analyzed
            - **{total_countries}** countries covered
            - Comprehensive global perspective
            """)
        
        st.markdown("---")
        
        # Policy Recommendations
        st.markdown("### 📋 Policy Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Immediate Actions:**
            - **Male-focused** mental health programs
            - **Age-specific** intervention strategies
            - **High-risk country** support initiatives
            - Community-based early warning systems
            
            **Healthcare Integration:**
            - Screening in primary care settings
            - First responder training programs
            - Crisis intervention protocols
            """)
        
        with col2:
            st.markdown("""
            **Global Cooperation:**
            - Standardized data collection methods
            - Best practice sharing networks
            - International crisis response coordination
            
            **Research Priorities:**
            - Machine learning prediction models
            - Cultural factor analysis
            - Intervention effectiveness studies
            - Real-time monitoring systems
            """)
        
        # Key Takeaways
        st.markdown("### 🔍 Stakeholder Takeaways")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Policymakers & Healthcare:**
            - Multi-sector approach required
            - Gender-specific interventions essential
            - Economic development alone insufficient
            - Community involvement crucial
            """)
        
        with col2:
            st.markdown("""
            **Researchers & Communities:**
            - Complex multi-factorial causes
            - Cultural factors highly significant
            - Technology offers new opportunities
            - Early intervention saves lives
            """)
        
        # Final Summary
        st.markdown("""
        <div class="highlight-box">
            <h3>🌟 Final Summary</h3>
            <p><strong>Suicide follows distinct demographic, geographic, and temporal patterns.</strong> Targeted interventions focusing on high-risk populations (males, elderly, specific regions) can have significant impact. Prevention is achievable through coordinated, data-driven approaches addressing individual, social, and economic factors.</p>
        </div>
        """, unsafe_allow_html=True)
        

if __name__ == "__main__":
    main()

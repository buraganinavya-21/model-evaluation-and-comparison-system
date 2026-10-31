"""
===================================================================
MODEL EVALUATION AND COMPARISON SYSTEM
===================================================================
B.Tech Capstone Project - Streamlit Web Application
Author: Student Capstone Project
Technologies: Python, Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
===================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# Import modular components
from src.preprocessing import get_dataset_overview, validate_target, preprocess_and_split
from src.models import get_available_models, get_model_insights
from src.evaluation import train_and_evaluate_all
from src.visualization import (
    plot_target_distribution,
    plot_metrics_comparison,
    plot_accuracy_ranking,
    plot_confusion_matrix_heatmap
)

# Page configuration
st.set_page_config(
    page_title="Model Evaluation and Comparison System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, professional Light Theme academic dashboard
st.markdown("""
<style>
    /* Main container background & text color */
    .stApp {
        background-color: #FFFFFF;
        color: #0F172A;
    }
    
    /* Header typography */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    
    /* Metric Cards - High Contrast Light Theme */
    div[data-testid="stMetric"] {
        background-color: #EFF6FF !important;
        padding: 0.9rem 1.1rem !important;
        border-radius: 8px !important;
        border: 1px solid #BFDBFE !important;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.08) !important;
    }
    div[data-testid="stMetricLabel"] label, div[data-testid="stMetricLabel"] p {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }
    div[data-testid="stMetricValue"] div, div[data-testid="stMetricValue"] p {
        color: #1E3A8A !important;
        font-weight: 700 !important;
        font-size: 1.55rem !important;
    }
    
    /* Card Boxes */
    .card-box {
        background-color: #F8FAFC;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
        color: #0F172A;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #1E3A8A !important;
    }
    
    /* Tabs styling */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        color: #475569 !important;
        font-size: 0.98rem !important;
    }
    button[aria-selected="true"] {
        color: #1E3A8A !important;
        border-bottom-color: #2563EB !important;
    }
    
    /* Buttons */
    .stButton>button {
        font-weight: 600;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


def load_dataset(data_source, uploaded_file):
    """Loads dataframe based on sidebar selection."""
    if data_source == "Sample Dataset (Heart Disease)":
        sample_path = os.path.join(os.path.dirname(__file__), "sample_dataset.csv")
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            return df, "sample_dataset.csv"
        else:
            st.error("Sample dataset file not found locally!")
            return None, None
    elif data_source == "Upload CSV File" and uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df, uploaded_file.name
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None, None
    return None, None


def main():
    # Application Title Banner
    st.markdown('<div class="main-header">MODEL EVALUATION AND COMPARISON SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An Interactive Machine Learning Classification Benchmark & Model Analysis Platform</div>', unsafe_allow_html=True)
    st.divider()

    # -----------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # -----------------------------------------------------------------------------
    st.sidebar.header("📁 1. Dataset & Configuration")
    
    data_source = st.sidebar.radio(
        "Select Dataset Source:",
        ["Sample Dataset (Heart Disease)", "Upload CSV File"]
    )
    
    uploaded_file = None
    if data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

    # Load Dataset
    df, dataset_name = load_dataset(data_source, uploaded_file)
    
    if df is None:
        st.info("👈 Please select or upload a `.csv` dataset from the sidebar to begin.")
        
        # Display beginner prompt & architecture overview
        st.markdown("### Welcome to the Model Evaluation & Comparison System")
        st.write("""
        This platform simplifies machine learning model benchmarking for classification problems.
        
        #### How to use:
        1. **Select or Upload Dataset**: Choose the sample dataset or upload your own CSV file in the sidebar.
        2. **Select Target Column**: Pick the output label column you wish to predict.
        3. **Select Algorithms**: Choose one or multiple classification algorithms.
        4. **Train & Evaluate**: Click the **Train & Evaluate Models** button to view accuracy, precision, recall, F1-score, and visual charts.
        """)
        
        # Display preview of sample info
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Project Information**")
        st.sidebar.caption("B.Tech Capstone Project | Machine Learning System")
        return

    # Sidebar Dataset Configuration
    st.sidebar.subheader("🎯 2. Target Column Selection")
    all_columns = df.columns.tolist()
    
    # Default selection heuristic (prefer 'target_disease' or last column)
    default_idx = len(all_columns) - 1
    if "target_disease" in all_columns:
        default_idx = all_columns.index("target_disease")
    elif "target" in all_columns:
        default_idx = all_columns.index("target")
        
    target_col = st.sidebar.selectbox("Select Target (y) Column:", all_columns, index=default_idx)
    
    st.sidebar.subheader("⚙️ 3. Pipeline Settings")
    test_size = st.sidebar.slider("Test Set Size Ratio:", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    remove_duplicates = st.sidebar.checkbox("Remove Duplicate Rows", value=True)
    
    st.sidebar.subheader("🤖 4. Model Selection")
    available_models_dict = get_available_models(random_state=42)
    selected_model_names = st.sidebar.multiselect(
        "Select Classification Algorithms:",
        options=list(available_models_dict.keys()),
        default=list(available_models_dict.keys())
    )
    
    train_button = st.sidebar.button("🚀 Train & Evaluate Models", use_container_width=True, type="primary")

    # -----------------------------------------------------------------------------
    # MAIN DASHBOARD TABS
    # -----------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dataset Overview",
        "⚙️ Preprocessing Details",
        "🚀 Model Comparison",
        "🔍 Confusion Matrix",
        "💡 Algorithm Insights"
    ])

    # Dataset Overview Module
    overview = get_dataset_overview(df)
    target_val = validate_target(df, target_col)

    with tab1:
        st.subheader("Module 1: Dataset Overview & Inspection")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dataset Name", dataset_name if dataset_name else "Uploaded CSV")
        col2.metric("Total Rows", f"{overview['num_rows']:,}")
        col3.metric("Total Columns", f"{overview['num_cols']}")
        col4.metric("Duplicate Rows", f"{overview['duplicate_rows']}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data Preview & Dtypes
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("#### First 5 Rows Preview")
            st.dataframe(df.head(5), use_container_width=True)
        with c2:
            st.markdown("#### Feature Summary & Data Types")
            st.dataframe(overview['dtype_summary'], height=220, use_container_width=True)
            
        st.divider()
        
        # Target Validation & Distribution
        st.markdown(f"#### Target Column Analysis: `{target_col}`")
        if not target_val['is_valid']:
            st.error(target_val['message'])
        else:
            if target_val['status'] == 'warning':
                st.warning(target_val['message'])
            else:
                st.success(f"✓ Valid Classification Target: Detected **{target_val['unique_count']}** unique class labels.")
                
            fig_dist = plot_target_distribution(target_val['class_distribution'], target_col)
            st.pyplot(fig_dist)

    # Preprocessing Module Tab
    with tab2:
        st.subheader("Module 2: Data Preprocessing Pipeline")
        st.write("""
        To prevent **data leakage**, numerical scaling and categorical encoding are fitted **only** on the training dataset (`X_train`)
        and subsequently applied to transform the testing dataset (`X_test`).
        """)
        
        st.markdown(r"""
        **Pipeline Steps Executed:**
        1. **Duplicate Removal**: Removes duplicate records if enabled.
        2. **Imputation**: Missing numerical values filled using `median`; categorical missing values filled with `most_frequent`.
        3. **Categorical Encoding**: Converted text/categorical features into numeric dummy vectors using `OneHotEncoder(handle_unknown='ignore')`.
        4. **Numerical Scaling**: Standardized numerical features using `StandardScaler` ($\mu=0, \sigma=1$).
        5. **Target Encoding**: Encoded target classes into zero-indexed integers using `LabelEncoder`.
        """)
        
        if target_val['is_valid']:
            prep_info = preprocess_and_split(
                df, target_col, test_size=test_size, remove_duplicates=remove_duplicates, random_state=42
            )
            
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Training Samples", f"{prep_info['train_samples']} rows")
            p2.metric("Testing Samples", f"{prep_info['test_samples']} rows")
            p3.metric("Processed Features", f"{prep_info['num_features_processed']} features")
            p4.metric("Test Split Ratio", f"{int(test_size*100)}%")
            
            st.markdown("#### Detected Input Features")
            st.write(f"**Numerical Features ({len(prep_info['num_cols'])}):** `{', '.join(prep_info['num_cols']) if prep_info['num_cols'] else 'None'}`")
            st.write(f"**Categorical Features ({len(prep_info['cat_cols'])}):** `{', '.join(prep_info['cat_cols']) if prep_info['cat_cols'] else 'None'}`")
        else:
            st.error("Cannot run preprocessing because the selected target column is invalid.")

    # Model Evaluation & Comparison Tab
    # Use session state to retain trained model outputs across tab clicks
    if train_button or ('eval_results' in st.session_state and st.session_state.get('last_target') == target_col):
        if train_button:
            if not target_val['is_valid']:
                st.error("Cannot train models: Invalid target column.")
            elif not selected_model_names:
                st.warning("Please select at least one algorithm from the sidebar.")
            else:
                with st.spinner("Training selected Machine Learning models..."):
                    prep = preprocess_and_split(
                        df, target_col, test_size=test_size, remove_duplicates=remove_duplicates, random_state=42
                    )
                    
                    results_dict, summary_df = train_and_evaluate_all(
                        selected_names=selected_model_names,
                        available_models=available_models_dict,
                        X_train=prep['X_train_proc'],
                        y_train=prep['y_train'],
                        X_test=prep['X_test_proc'],
                        y_test=prep['y_test'],
                        class_names=prep['class_names']
                    )
                    
                    st.session_state['eval_results'] = results_dict
                    st.session_state['summary_df'] = summary_df
                    st.session_state['class_names'] = prep['class_names']
                    st.session_state['last_target'] = target_col
                    
    with tab3:
        st.subheader("Module 3 & 4: Model Evaluation & Comparison Results")
        
        if 'summary_df' in st.session_state and not st.session_state['summary_df'].empty:
            summary_df = st.session_state['summary_df']
            
            st.markdown("#### 1. Evaluation Summary Table")
            
            # High-contrast Light Theme Styler
            styled_summary = (
                summary_df.style
                .format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1-Score': '{:.4f}',
                    'ROC-AUC': '{:.4f}'
                })
                .highlight_max(axis=0, color='#DBEAFE', subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
            )
            
            st.dataframe(styled_summary, use_container_width=True)
            
            st.caption("Note: Metrics are calculated on the test split. Precision, Recall, and F1-Score use weighted averaging for multi-class targets. Highest score in each column is highlighted in soft blue.")
            st.divider()
            
            st.markdown("#### 2. Visual Model Comparison")
            g1, g2 = st.columns([1, 1])
            with g1:
                fig_grp = plot_metrics_comparison(summary_df)
                st.pyplot(fig_grp)
            with g2:
                fig_rank = plot_accuracy_ranking(summary_df)
                st.pyplot(fig_rank)
        else:
            st.info("👈 Select models in the sidebar and click **'Train & Evaluate Models'** to generate evaluation results.")

    # Confusion Matrix Tab
    with tab4:
        st.subheader("Module 5: Detailed Model Analysis & Confusion Matrix")
        
        if 'eval_results' in st.session_state and st.session_state['eval_results']:
            results_dict = st.session_state['eval_results']
            class_names = st.session_state['class_names']
            
            trained_models = [m for m in results_dict.keys() if 'error' not in results_dict[m]]
            
            if trained_models:
                selected_model_cm = st.selectbox("Select Model to Inspect Confusion Matrix:", trained_models)
                
                res = results_dict[selected_model_cm]
                
                col_cm1, col_cm2 = st.columns([1, 1])
                with col_cm1:
                    st.markdown(f"#### Confusion Matrix: `{selected_model_cm}`")
                    fig_cm = plot_confusion_matrix_heatmap(res['confusion_matrix'], class_names, selected_model_cm)
                    st.pyplot(fig_cm)
                with col_cm2:
                    st.markdown(f"#### Classification Report: `{selected_model_cm}`")
                    st.dataframe(res['classification_report_df'].style.format('{:.4f}'), use_container_width=True)
            else:
                st.warning("No models were successfully trained.")
        else:
            st.info("Please run model training first from the sidebar button.")

    # Algorithm Insights Tab
    with tab5:
        st.subheader("Module 6: Algorithm Insights & Theoretical Reference")
        st.write("Explore fundamental concepts, hyperparameters, advantages, and limitations of each classification algorithm.")
        
        insights_info = get_model_insights()
        for algo_name, info in insights_info.items():
            with st.expander(f"📌 {info['name']} ({info['type']})"):
                st.markdown(f"**Working Principle:** {info['principle']}")
                st.markdown(f"**Key Hyperparameters:** `{info['hyperparameters']}`")
                st.markdown(f"**Advantages:** {info['advantages']}")
                st.markdown(f"**Limitations:** {info['limitations']}")

    # Footer
    st.divider()
    st.caption("Model Evaluation and Comparison System | College B.Tech Capstone Project | Built with Python, Scikit-learn & Streamlit")


if __name__ == "__main__":
    main()

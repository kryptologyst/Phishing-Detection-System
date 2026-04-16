"""Streamlit demo application for phishing detection."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import time

from src.phishing_detector import PhishingDetector
from src.utils import load_config


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Phishing Detection System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🛡️ Phishing Detection System")
    st.markdown("**Research and Educational Tool for Phishing Detection**")
    
    # Disclaimer
    with st.expander("⚠️ Important Disclaimer", expanded=True):
        st.warning("""
        **This is a research and educational demonstration tool only.**
        
        - NOT intended for production security operations
        - May produce inaccurate results
        - Should not be used as a primary security control
        - Always consult with security professionals for production deployments
        """)
    
    # Initialize session state
    if 'detector' not in st.session_state:
        with st.spinner("Initializing phishing detection system..."):
            try:
                st.session_state.detector = PhishingDetector()
                st.session_state.models_trained = False
                st.session_state.results = None
            except Exception as e:
                st.error(f"Error initializing system: {e}")
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        st.subheader("Model Selection")
        model_options = ["random_forest", "xgboost", "neural_network"]
        selected_model = st.selectbox("Choose Model", model_options, index=0)
        
        # Dataset size
        st.subheader("Dataset Configuration")
        n_samples = st.slider("Number of Samples", 1000, 50000, 10000, 1000)
        
        # Training button
        st.subheader("Training")
        if st.button("🚀 Train All Models", type="primary"):
            train_models(n_samples)
        
        # Status
        st.subheader("Status")
        if st.session_state.models_trained:
            st.success("✅ Models Trained")
            st.info(f"Models: {', '.join(st.session_state.detector.trained_models.keys())}")
        else:
            st.warning("⚠️ Models Not Trained")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single URL Analysis", "📊 Batch Analysis", "📈 Model Performance", "ℹ️ About"])
    
    with tab1:
        single_url_analysis(selected_model)
    
    with tab2:
        batch_analysis(selected_model)
    
    with tab3:
        model_performance()
    
    with tab4:
        about_section()


def train_models(n_samples: int):
    """Train all models with progress tracking."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Generating dataset...")
        progress_bar.progress(0.1)
        
        # Generate dataset
        dataset = st.session_state.detector.generate_dataset(n_samples)
        
        status_text.text("Creating train-test split...")
        progress_bar.progress(0.2)
        
        # Create splits
        X, y = dataset['X'], dataset['y']
        X_train, X_test, y_train, y_test = st.session_state.detector.data_processor.create_train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        status_text.text("Loading models...")
        progress_bar.progress(0.3)
        
        # Load models
        st.session_state.detector.load_default_models()
        
        status_text.text("Training models...")
        progress_bar.progress(0.4)
        
        # Train models
        training_metrics = st.session_state.detector.train_all_models(X_train, y_train, X_test, y_test)
        
        status_text.text("Evaluating models...")
        progress_bar.progress(0.8)
        
        # Evaluate models
        evaluation_metrics = st.session_state.detector.evaluate_all_models(X_test, y_test)
        
        status_text.text("Creating leaderboard...")
        progress_bar.progress(0.9)
        
        # Create leaderboard
        leaderboard = st.session_state.detector.create_leaderboard(evaluation_metrics)
        
        # Store results
        st.session_state.results = {
            'dataset_info': dataset['metadata'],
            'training_metrics': training_metrics,
            'evaluation_metrics': evaluation_metrics,
            'leaderboard': leaderboard,
            'feature_names': dataset['feature_names']
        }
        
        st.session_state.models_trained = True
        
        progress_bar.progress(1.0)
        status_text.text("✅ Training completed successfully!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.success("All models trained successfully!")
        
    except Exception as e:
        st.error(f"Error during training: {e}")
        progress_bar.empty()
        status_text.empty()


def single_url_analysis(model_name: str):
    """Single URL analysis interface."""
    st.header("Single URL Analysis")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first using the sidebar.")
        return
    
    # URL input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url = st.text_input(
            "Enter URL to analyze:",
            placeholder="https://example.com",
            help="Enter a URL to check for phishing risk"
        )
    
    with col2:
        analyze_button = st.button("🔍 Analyze", type="primary")
    
    # Content input (optional)
    content = st.text_area(
        "Optional: Web page content",
        placeholder="Paste web page content here for enhanced analysis...",
        height=100,
        help="Optional: Provide web page content for more accurate analysis"
    )
    
    if analyze_button and url:
        if url.strip():
            try:
                with st.spinner("Analyzing URL..."):
                    result = st.session_state.detector.predict_phishing_risk(
                        url, model_name, content if content.strip() else None
                    )
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_score = result['phishing_probability']
                    if risk_score > 0.7:
                        st.error(f"🚨 **HIGH RISK**: {risk_score:.1%}")
                    elif risk_score > 0.4:
                        st.warning(f"⚠️ **MEDIUM RISK**: {risk_score:.1%}")
                    else:
                        st.success(f"✅ **LOW RISK**: {risk_score:.1%}")
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                
                with col3:
                    st.metric("Features Analyzed", result['features_extracted'])
                
                # Detailed results
                st.subheader("Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Prediction:**")
                    st.write(f"- Phishing: {result['phishing_probability']:.1%}")
                    st.write(f"- Legitimate: {result['legitimate_probability']:.1%}")
                    st.write(f"- Model: {result['model_used']}")
                
                with col2:
                    # Feature importance (if available)
                    importance = st.session_state.detector.get_feature_importance(model_name)
                    if importance:
                        st.write("**Top Features:**")
                        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                        for feature, score in top_features:
                            st.write(f"- {feature}: {score:.3f}")
                
                # Risk explanation
                st.subheader("Risk Assessment")
                if result['phishing_probability'] > 0.7:
                    st.error("This URL shows characteristics commonly associated with phishing attempts.")
                elif result['phishing_probability'] > 0.4:
                    st.warning("This URL shows some suspicious characteristics. Exercise caution.")
                else:
                    st.success("This URL appears to be legitimate based on the analysis.")
                
            except Exception as e:
                st.error(f"Error analyzing URL: {e}")
        else:
            st.warning("Please enter a valid URL.")


def batch_analysis(model_name: str):
    """Batch URL analysis interface."""
    st.header("Batch URL Analysis")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first using the sidebar.")
        return
    
    # Batch input options
    input_method = st.radio(
        "Choose input method:",
        ["Manual Entry", "File Upload", "Sample URLs"]
    )
    
    urls = []
    
    if input_method == "Manual Entry":
        url_text = st.text_area(
            "Enter URLs (one per line):",
            placeholder="https://example1.com\nhttps://example2.com",
            height=200
        )
        if url_text:
            urls = [url.strip() for url in url_text.split('\n') if url.strip()]
    
    elif input_method == "File Upload":
        uploaded_file = st.file_uploader("Upload CSV file with URLs", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'url' in df.columns:
                    urls = df['url'].tolist()
                else:
                    st.error("CSV file must contain a 'url' column")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    elif input_method == "Sample URLs":
        sample_urls = [
            "https://google.com",
            "https://facebook.com",
            "https://secure-bank-login.tk",
            "https://paypal-security.ml",
            "https://amazon.com"
        ]
        urls = sample_urls
        st.info("Using sample URLs for demonstration")
    
    if urls and st.button("🔍 Analyze Batch", type="primary"):
        if len(urls) > 100:
            st.warning("Large batch detected. This may take some time...")
        
        try:
            with st.spinner(f"Analyzing {len(urls)} URLs..."):
                results = st.session_state.detector.batch_predict(urls, model_name)
            
            # Create results DataFrame
            df_results = pd.DataFrame(results)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total URLs", len(urls))
            
            with col2:
                high_risk = len(df_results[df_results['phishing_probability'] > 0.7])
                st.metric("High Risk", high_risk)
            
            with col3:
                medium_risk = len(df_results[(df_results['phishing_probability'] > 0.4) & 
                                           (df_results['phishing_probability'] <= 0.7)])
                st.metric("Medium Risk", medium_risk)
            
            with col4:
                low_risk = len(df_results[df_results['phishing_probability'] <= 0.4])
                st.metric("Low Risk", low_risk)
            
            # Results table
            st.subheader("Analysis Results")
            
            # Add risk category
            df_results['risk_category'] = df_results['phishing_probability'].apply(
                lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low'
            )
            
            # Display table
            st.dataframe(
                df_results[['url', 'phishing_probability', 'risk_category', 'confidence']],
                use_container_width=True
            )
            
            # Download results
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="phishing_analysis_results.csv",
                mime="text/csv"
            )
            
            # Visualization
            st.subheader("Risk Distribution")
            
            fig = px.histogram(
                df_results, 
                x='phishing_probability',
                nbins=20,
                title="Distribution of Phishing Risk Scores",
                labels={'phishing_probability': 'Phishing Probability', 'count': 'Number of URLs'}
            )
            fig.add_vline(x=0.4, line_dash="dash", line_color="orange", 
                         annotation_text="Medium Risk Threshold")
            fig.add_vline(x=0.7, line_dash="dash", line_color="red", 
                         annotation_text="High Risk Threshold")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error during batch analysis: {e}")


def model_performance():
    """Model performance visualization."""
    st.header("Model Performance")
    
    if not st.session_state.models_trained or not st.session_state.results:
        st.warning("Please train models first to view performance metrics.")
        return
    
    results = st.session_state.results
    
    # Leaderboard
    st.subheader("Model Leaderboard")
    
    if 'leaderboard' in results:
        # Create leaderboard table
        leaderboard_data = []
        for metric, rankings in results['leaderboard'].items():
            for rank_info in rankings:
                leaderboard_data.append({
                    'Metric': metric,
                    'Model': rank_info['model'],
                    'Score': rank_info['score'],
                    'Rank': rank_info['rank']
                })
        
        if leaderboard_data:
            df_leaderboard = pd.DataFrame(leaderboard_data)
            
            # Pivot table for better visualization
            pivot_table = df_leaderboard.pivot_table(
                index='Model', 
                columns='Metric', 
                values='Score', 
                aggfunc='first'
            ).fillna(0)
            
            st.dataframe(pivot_table, use_container_width=True)
    
    # Performance metrics
    st.subheader("Detailed Performance Metrics")
    
    if 'evaluation_metrics' in results:
        metrics = results['evaluation_metrics']
        
        # Create metrics comparison
        metrics_data = []
        for model_name, model_metrics in metrics.items():
            if 'error' not in model_metrics:
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': model_metrics.get('accuracy', 0),
                    'Precision': model_metrics.get('precision', 0),
                    'Recall': model_metrics.get('recall', 0),
                    'F1-Score': model_metrics.get('f1_score', 0),
                    'AUCPR': model_metrics.get('aucpr', 0),
                    'ROC AUC': model_metrics.get('roc_auc', 0)
                })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            
            # Display metrics table
            st.dataframe(df_metrics, use_container_width=True)
            
            # Performance comparison chart
            st.subheader("Performance Comparison")
            
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUCPR', 'ROC AUC']
            
            fig = go.Figure()
            
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=df_metrics['Model'],
                    y=df_metrics[metric],
                    text=df_metrics[metric].round(3),
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance")
    
    if st.session_state.detector.trained_models:
        model_for_importance = st.selectbox(
            "Select model for feature importance:",
            list(st.session_state.detector.trained_models.keys())
        )
        
        importance = st.session_state.detector.get_feature_importance(model_for_importance)
        
        if importance:
            # Sort features by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            # Create DataFrame
            df_importance = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])
            
            # Display top features
            st.dataframe(df_importance.head(10), use_container_width=True)
            
            # Feature importance chart
            fig = px.bar(
                df_importance.head(15),
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top 15 Feature Importance - {model_for_importance}",
                labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)


def about_section():
    """About section with system information."""
    st.header("About the Phishing Detection System")
    
    st.markdown("""
    This phishing detection system is designed for **research and educational purposes only**. 
    It demonstrates various machine learning approaches to detecting phishing attempts based on 
    URL characteristics and web content analysis.
    """)
    
    st.subheader("Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **URL Analysis:**
        - URL length and structure
        - Domain characteristics
        - Suspicious patterns
        - Entropy analysis
        
        **Content Analysis:**
        - Text length and complexity
        - Suspicious keywords
        - Form analysis
        - Link patterns
        """)
    
    with col2:
        st.markdown("""
        **Models Supported:**
        - Random Forest
        - XGBoost
        - Neural Networks
        
        **Evaluation Metrics:**
        - AUCPR (Area Under Precision-Recall Curve)
        - Precision@K
        - F1-Score
        - Alert volume analysis
        """)
    
    st.subheader("Technical Details")
    
    if st.session_state.results and 'dataset_info' in st.session_state.results:
        dataset_info = st.session_state.results['dataset_info']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dataset Size", f"{dataset_info['n_samples']:,}")
        
        with col2:
            st.metric("Phishing Ratio", f"{dataset_info['phishing_ratio']:.1%}")
        
        with col3:
            st.metric("Features", len(st.session_state.results.get('feature_names', [])))
    
    st.subheader("Privacy and Safety")
    
    st.info("""
    **Privacy Protection:**
    - PII detection and anonymization
    - No persistent storage of analyzed URLs
    - Built-in data protection measures
    
    **Safety Measures:**
    - Input validation and sanitization
    - Rate limiting for batch analysis
    - Error handling and logging
    """)
    
    st.subheader("Limitations")
    
    st.warning("""
    **Important Limitations:**
    - This is a research prototype, not a production system
    - May produce false positives and false negatives
    - Limited to URL and basic content analysis
    - Sophisticated phishing techniques may bypass detection
    - Always consult security professionals for production use
    """)


if __name__ == "__main__":
    main()

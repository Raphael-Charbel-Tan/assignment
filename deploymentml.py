import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import io

# Set page configuration
st.set_page_config(
    page_title="Clustering Model Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .card {
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        background-color: #f9f9f9;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .highlight {
        font-weight: bold;
        color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# Functions for loading models
@st.cache_resource
def load_models():
    """Load the saved models from pickle files"""
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('pca.pkl', 'rb') as f:
            pca = pickle.load(f)
        with open('gmm_model.pkl', 'rb') as f:
            gmm = pickle.load(f)
        return scaler, pca, gmm
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Functions for data processing and visualization
def plot_pca_components(pca_model):
    """Visualize PCA components importance"""
    explained_variance = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, 
           color='royalblue', label='Individual explained variance')
    ax.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid',
           color='red', label='Cumulative explained variance')
    ax.axhline(y=0.95, color='green', linestyle='--', label='95% Explained Variance')
    
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Component Importance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_pca_2d(data, labels, title="2D PCA Visualization of Clusters"):
    """Create a 2D scatter plot of the first two PCA components with cluster colors"""
    fig = px.scatter(
        x=data[:, 0],
        y=data[:, 1],
        color=labels,
        labels={'x': 'Principal Component 1', 'y': 'Principal Component 2', 'color': 'Cluster'},
        title=title,
        color_continuous_scale=px.colors.qualitative.Bold,
        template="plotly_white"
    )
    fig.update_layout(height=600)
    return fig

def plot_pca_3d(data, labels, title="3D PCA Visualization of Clusters"):
    """Create a 3D scatter plot of the first three PCA components with cluster colors"""
    fig = px.scatter_3d(
        x=data[:, 0], 
        y=data[:, 1], 
        z=data[:, 2],
        color=labels,
        labels={'x': 'PC 1', 'y': 'PC 2', 'z': 'PC 3', 'color': 'Cluster'},
        title=title,
        opacity=0.7,
        template="plotly_white"
    )
    fig.update_layout(height=700)
    return fig

def plot_feature_importance(pca_model, feature_names):
    """Visualize feature importance based on PCA loadings"""
    # Get component loadings
    loadings = pca_model.components_
    n_components = loadings.shape[0]
    n_features = len(feature_names)
    
    # Create a heatmap of feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(loadings, cmap='viridis')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_components))
    ax.set_xticklabels(feature_names, rotation=90)
    ax.set_yticklabels([f'PC {i+1}' for i in range(n_components)])
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Feature Loading Strength')
    
    # Add title and labels
    ax.set_title('PCA Feature Importance Heatmap')
    
    # Annotate cells with the values
    for i in range(n_components):
        for j in range(n_features):
            text = ax.text(j, i, f"{loadings[i, j]:.2f}",
                          ha="center", va="center", color="white" if abs(loadings[i, j]) > 0.5 else "black")
    
    plt.tight_layout()
    return fig

def plot_silhouette_scores(X_pca, cluster_range=range(2, 11)):
    """Calculate and plot silhouette scores for different numbers of clusters"""
    silhouette_scores = []
    
    for n_clusters in cluster_range:
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(X_pca)
        
        # Silhouette score is only valid for n_clusters > 1
        score = silhouette_score(X_pca, cluster_labels)
        silhouette_scores.append(score)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(cluster_range),
        y=silhouette_scores,
        mode='lines+markers',
        name='Silhouette Score',
        marker=dict(size=10, color='royalblue'),
        line=dict(width=2, color='royalblue')
    ))
    
    fig.update_layout(
        title="Silhouette Score for Different Numbers of Clusters",
        xaxis_title="Number of Clusters",
        yaxis_title="Silhouette Score",
        height=500,
        template="plotly_white"
    )
    
    # Add a line at the maximum silhouette score
    max_idx = np.argmax(silhouette_scores)
    max_clusters = list(cluster_range)[max_idx]
    max_score = silhouette_scores[max_idx]
    
    fig.add_shape(type="line",
        x0=max_clusters, y0=0, x1=max_clusters, y1=max_score,
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=max_clusters,
        y=max_score,
        text=f"Best: {max_clusters} clusters (score: {max_score:.3f})",
        showarrow=True,
        arrowhead=1
    )
    
    return fig

def predict_cluster(input_data, scaler, pca, gmm):
    """Predict cluster for new input data"""
    # Reshape if needed
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    
    # Scale and transform with PCA
    scaled_data = scaler.transform(input_data)
    pca_data = pca.transform(scaled_data)
    
    # Predict cluster
    cluster = gmm.predict(pca_data)
    probabilities = gmm.predict_proba(pca_data)
    
    return cluster, probabilities, pca_data

# Main function with sidebar navigation
def main():
    # Load models
    scaler, pca, gmm = load_models()
    if not (scaler and pca and gmm):
        st.error("Failed to load required models. Please check that model files exist.")
        return
    
    # Display header
    st.markdown("<h1 class='main-header'>Interactive Clustering Model Dashboard</h1>", unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = ["Home", "Data Upload & Exploration", "PCA Visualization", "Clustering Analysis", 
             "Model Performance", "Feature Importance", "Make Predictions"]
    selected_page = st.sidebar.radio("Go to", pages)
    
    # Sidebar model info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    st.sidebar.info(f"""
    - PCA Components: {pca.n_components_}
    - GMM Clusters: {gmm.n_components}
    - GMM Covariance Type: {gmm.covariance_type}
    """)
    
    # Display the selected page
    if selected_page == "Home":
        show_home_page()
    elif selected_page == "Data Upload & Exploration":
        show_data_exploration(scaler)
    elif selected_page == "PCA Visualization":
        show_pca_visualization(pca)
    elif selected_page == "Clustering Analysis":
        show_clustering_analysis(scaler, pca, gmm)
    elif selected_page == "Model Performance":
        show_model_performance(pca, gmm)
    elif selected_page == "Feature Importance":
        show_feature_importance(pca)
    elif selected_page == "Make Predictions":
        show_predictions(scaler, pca, gmm)

def show_home_page():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>📊 Welcome to the Clustering Model Dashboard</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    This interactive dashboard allows you to explore and interact with a clustering model based on 
    Principal Component Analysis (PCA) and Gaussian Mixture Models (GMM).
    
    ### Available Features:
    
    - **Data Upload & Exploration**: Upload and analyze your dataset
    - **PCA Visualization**: Explore dimensionality reduction with interactive 2D and 3D visualizations
    - **Clustering Analysis**: Visualize how your data is clustered using GMM
    - **Model Performance**: Analyze silhouette scores and other performance metrics
    - **Feature Importance**: Understand which features contribute most to the clustering
    - **Make Predictions**: Submit new data points and see which cluster they belong to
    
    Use the navigation panel on the left to explore these features.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

def show_data_exploration(scaler):
    st.markdown("<h2 class='sub-header'>Data Upload & Exploration</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    Upload your dataset to explore its properties and prepare it for clustering analysis.
    The dataset should be in CSV format with features in columns.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load the dataset
            df = pd.read_csv(uploaded_file)
            
            # Display dataset info
            st.markdown("<h3 class='sub-header'>Dataset Overview</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
            with col2:
                st.write(f"**Missing Values:** {df.isna().sum().sum()}")
                st.write(f"**Duplicate Rows:** {df.duplicated().sum()}")
            
            # Show data sample
            st.markdown("<h3 class='sub-header'>Data Sample</h3>", unsafe_allow_html=True)
            st.dataframe(df.head(10))
            
            # Data preprocessing options
            st.markdown("<h3 class='sub-header'>Data Preprocessing</h3>", unsafe_allow_html=True)
            
            # Choose columns for analysis
            st.markdown("**Select columns for analysis:**")
            selected_columns = st.multiselect("Choose features", df.columns.tolist(), df.columns.tolist())
            
            if selected_columns:
                df_selected = df[selected_columns]
                
                # Handle missing values
                if df_selected.isna().sum().sum() > 0:
                    st.warning("Dataset contains missing values. Choose a handling method:")
                    missing_method = st.selectbox(
                        "Missing value handling method",
                        ["Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with 0"]
                    )
                    
                    if missing_method == "Drop rows with missing values":
                        df_selected = df_selected.dropna()
                        st.info(f"Dropped {df.shape[0] - df_selected.shape[0]} rows with missing values.")
                    elif missing_method == "Fill with mean":
                        df_selected = df_selected.fillna(df_selected.mean())
                    elif missing_method == "Fill with median":
                        df_selected = df_selected.fillna(df_selected.median())
                    elif missing_method == "Fill with 0":
                        df_selected = df_selected.fillna(0)
                
                # Display descriptive statistics
                st.markdown("<h3 class='sub-header'>Descriptive Statistics</h3>", unsafe_allow_html=True)
                st.dataframe(df_selected.describe())
                
                # Data visualization
                st.markdown("<h3 class='sub-header'>Data Visualization</h3>", unsafe_allow_html=True)
                
                chart_type = st.selectbox(
                    "Select visualization type",
                    ["Correlation Heatmap", "Feature Distributions", "Pairwise Relationships"]
                )
                
                if chart_type == "Correlation Heatmap":
                    fig, ax = plt.subplots(figsize=(10, 8))
                    correlation = df_selected.corr()
                    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
                    plt.title("Feature Correlation Heatmap")
                    st.pyplot(fig)
                    
                elif chart_type == "Feature Distributions":
                    # Let user select features for histograms
                    hist_features = st.multiselect(
                        "Select features to plot distributions (max 4 recommended)",
                        selected_columns,
                        selected_columns[:min(4, len(selected_columns))]
                    )
                    
                    if hist_features:
                        fig = px.histogram(
                            df_selected, 
                            x=hist_features[0] if len(hist_features) > 0 else None,
                            color_discrete_sequence=['royalblue'],
                            marginal="box"
                        )
                        fig.update_layout(title=f"Distribution of {hist_features[0]}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if len(hist_features) > 1:
                            cols = st.columns(min(3, len(hist_features) - 1))
                            for i, col in enumerate(cols):
                                if i + 1 < len(hist_features):
                                    feature = hist_features[i + 1]
                                    fig = px.histogram(
                                        df_selected, 
                                        x=feature,
                                        color_discrete_sequence=['green'],
                                        marginal="box"
                                    )
                                    fig.update_layout(title=f"Distribution of {feature}")
                                    col.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Pairwise Relationships":
                    # Let user select features for pairplot
                    pair_features = st.multiselect(
                        "Select features for pairwise plot (2-4 recommended)",
                        selected_columns,
                        selected_columns[:min(3, len(selected_columns))]
                    )
                    
                    if len(pair_features) >= 2:
                        fig = px.scatter_matrix(
                            df_selected,
                            dimensions=pair_features,
                            color_discrete_sequence=['royalblue']
                        )
                        fig.update_layout(title="Pairwise Feature Relationships")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select at least 2 features for pairwise plot.")
                
                # Feature scaling demonstration
                st.markdown("<h3 class='sub-header'>Feature Scaling Demonstration</h3>", unsafe_allow_html=True)
                
                if st.checkbox("Show effect of feature scaling"):
                    # Select a sample of data for visualization
                    sample_size = min(1000, df_selected.shape[0])
                    df_sample = df_selected.sample(sample_size, random_state=42)
                    
                    # Apply scaling using the loaded scaler
                    numeric_cols = df_sample.select_dtypes(include=np.number).columns.tolist()
                    if len(numeric_cols) > 0:
                        # Create a new scaler for demonstration
                        demo_scaler = StandardScaler()
                        df_scaled = pd.DataFrame(
                            demo_scaler.fit_transform(df_sample[numeric_cols]),
                            columns=numeric_cols
                        )
                        
                        # Visualize before and after scaling
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Before Scaling")
                            fig = px.box(df_sample[numeric_cols].melt(), 
                                         x="variable", y="value", 
                                         color="variable",
                                         title="Original Feature Distributions")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with col2:
                            st.subheader("After Scaling")
                            fig = px.box(df_scaled.melt(), 
                                         x="variable", y="value", 
                                         color="variable",
                                         title="Scaled Feature Distributions")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No numeric columns available for scaling demonstration.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_pca_visualization(pca_model):
    st.markdown("<h2 class='sub-header'>PCA Visualization</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms 
    high-dimensional data into a lower-dimensional representation while preserving as much variance as possible.
    
    This page visualizes the PCA components and allows you to explore your data in reduced dimensions.
    """)
    
    # Display PCA components information
    st.markdown("<h3 class='sub-header'>PCA Components Information</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Number of Components:** {pca_model.n_components_}")
        total_variance = sum(pca_model.explained_variance_ratio_)
        st.write(f"**Total Explained Variance:** {total_variance:.2%}")
    with col2:
        st.write(f"**Original Data Dimensions:** {pca_model.n_features_in_}")
        st.write(f"**Data Reduction:** {pca_model.n_features_in_ - pca_model.n_components_} dimensions removed")
    
    # Show PCA components importance
    st.markdown("<h3 class='sub-header'>PCA Components Importance</h3>", unsafe_allow_html=True)
    
    # Plot PCA components importance
    fig = plot_pca_components(pca_model)
    st.pyplot(fig)
    
    # Upload data for PCA visualization
    st.markdown("<h3 class='sub-header'>Visualize Your Data with PCA</h3>", unsafe_allow_html=True)
    st.markdown("""
    Upload your dataset to visualize it in the PCA space. The data should be preprocessed 
    and contain the same features that were used to train the PCA model.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file for PCA visualization", type="csv", key="pca_vis_upload")
    
    if uploaded_file is not None:
        try:
            # Load the dataset
            df = pd.read_csv(uploaded_file)
            
            # Display a sample of the data
            st.subheader("Data Sample")
            st.dataframe(df.head())
            
            # Select features for PCA visualization
            st.subheader("Select Features for PCA Visualization")
            
            # Get only numeric columns for PCA
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            selected_features = st.multiselect(
                "Choose numeric features for PCA visualization",
                numeric_cols,
                default=numeric_cols
            )
            
            if selected_features:
                # Prepare data for PCA
                X = df[selected_features].values
                
                # Let the user select a target variable for coloring the points
                color_options = ["None"] + df.columns.tolist()
                color_variable = st.selectbox("Choose a variable for coloring the points (optional)", color_options)
                
                # If a color variable is selected, prepare the labels
                if color_variable != "None":
                    labels = df[color_variable].values
                    color_name = color_variable
                else:
                    # No color variable selected, use a constant color
                    labels = np.zeros(X.shape[0])
                    color_name = None
                
                # Create a new scaler for demonstration
                demo_scaler = StandardScaler()
                X_scaled = demo_scaler.fit_transform(X)
                
                # Apply PCA transformation
                demo_pca = PCA(n_components=min(3, len(selected_features)))
                X_pca = demo_pca.fit_transform(X_scaled)
                
                # Visualization options
                vis_type = st.radio("Select visualization type", ["2D Visualization", "3D Visualization"])
                
                if vis_type == "2D Visualization":
                    # Create 2D PCA plot
                    fig = px.scatter(
                        x=X_pca[:, 0],
                        y=X_pca[:, 1],
                        color=labels if color_name else None,
                        labels={"x": "Principal Component 1", "y": "Principal Component 2", "color": color_name},
                        title="2D PCA Visualization",
                        opacity=0.7
                    )
                    
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:  # 3D Visualization
                    if X_pca.shape[1] >= 3:
                        # Create 3D PCA plot
                        fig = px.scatter_3d(
                            x=X_pca[:, 0],
                            y=X_pca[:, 1],
                            z=X_pca[:, 2],
                            color=labels if color_name else None,
                            labels={"x": "PC 1", "y": "PC 2", "z": "PC 3", "color": color_name},
                            title="3D PCA Visualization",
                            opacity=0.7
                        )
                        
                        fig.update_layout(height=700)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough dimensions for 3D visualization. Please select more features.")
                
                # Display PCA explained variance
                st.subheader("Explained Variance by Components")
                explained_variance = demo_pca.explained_variance_ratio_
                
                fig = px.bar(
                    x=[f"PC{i+1}" for i in range(len(explained_variance))],
                    y=explained_variance,
                    labels={"x": "Principal Components", "y": "Explained Variance Ratio"},
                    title="PCA Component Importance"
                )
                
                fig.add_trace(go.Scatter(
                    x=[f"PC{i+1}" for i in range(len(explained_variance))],
                    y=np.cumsum(explained_variance),
                    mode="lines+markers",
                    name="Cumulative Explained Variance",
                    line=dict(color="red")
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing the file for PCA visualization: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_clustering_analysis(scaler, pca, gmm):
    st.markdown("<h2 class='sub-header'>Clustering Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    This page allows you to explore the clustering results from the Gaussian Mixture Model (GMM).
    You can visualize clusters in PCA space and analyze cluster properties.
    
    Upload your dataset to see how it is clustered using the pre-trained GMM model.
    """)
    
    # Upload data for clustering visualization
    uploaded_file = st.file_uploader("Choose a CSV file for clustering analysis", type="csv", key="cluster_vis_upload")
    
    if uploaded_file is not None:
        try:
            # Load the dataset
            df = pd.read_csv(uploaded_file)
            
            # Display a sample of the data
            st.subheader("Data Sample")
            st.dataframe(df.head())
            
            # Select numeric features for clustering
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            # Let user select columns
            selected_features = st.multiselect(
                "Choose numeric features for clustering analysis",
                numeric_cols,
                default=numeric_cols
            )
            
            if selected_features:
                # Prepare data for clustering
                X = df[selected_features].values
                
                # Scale the data
                X_scaled = scaler.transform(X)
                
                # Apply PCA transformation
                X_pca = pca.transform(X_scaled)
                
                # Get cluster assignments
                cluster_labels = gmm.predict(X_pca)
                cluster_probs = gmm.predict_proba(X_pca)
                
                # Display number of samples in each cluster
                st.subheader("Cluster Distribution")
                
                cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                fig = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    labels={"x": "Cluster", "y": "Number of Samples"},
                    title="Distribution of Samples across Clusters",
                    color=cluster_counts.index,
                    text=cluster_counts.values
                )
                
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display cluster assignments
                df_clusters = df.copy()
                df_clusters['Cluster'] = cluster_labels
                
                # Add probability columns
                for i in range(gmm.n_components):
                    df_clusters[f'Prob_Cluster_{i}'] = cluster_probs[:, i]
                
                st.subheader("Data with Cluster Assignments")
                st.dataframe(df_clusters.head(10))
                
                # Visualization options
                vis_tab1, vis_tab2 = st.tabs(["2D Visualization", "3D Visualization"])
                
                with vis_tab1:
                    # Create 2D PCA plot with cluster colors
                    fig = plot_pca_2d(X_pca, cluster_labels, "2D PCA Visualization of Clusters")
                    st.plotly_chart(fig, use_container_width=True)
                
                with vis_tab2:
                    if X_pca.shape[1] >= 3:
                        # Create 3D PCA plot with cluster colors
                        fig = plot_pca_3d(X_pca, cluster_labels, "3D PCA Visualization of Clusters")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough dimensions for 3D visualization.")
                
                # Cluster statistics
                st.subheader("Cluster Statistics")
                
                # Let user select a feature to analyze across clusters
                feature_for_analysis = st.selectbox(
                    "Choose a feature to analyze across clusters",
                    selected_features
                )
                
                if feature_for_analysis:
                    # Create a
# Create a box plot to compare feature distributions across clusters
                    fig = px.box(
                        df_clusters, 
                        x='Cluster', 
                        y=feature_for_analysis,
                        color='Cluster',
                        title=f"Distribution of {feature_for_analysis} across Clusters",
                        points="all"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and display feature means for each cluster
                    cluster_means = df_clusters.groupby('Cluster')[selected_features].mean()
                    
                    st.subheader("Feature Means by Cluster")
                    st.dataframe(cluster_means)
                    
                    # Heatmap of feature means by cluster
                    fig = px.imshow(
                        cluster_means.values,
                        x=cluster_means.columns,
                        y=cluster_means.index,
                        labels=dict(x="Feature", y="Cluster", color="Mean Value"),
                        title="Heatmap of Feature Means by Cluster",
                        color_continuous_scale='viridis'
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Option to download clustered data
                    st.subheader("Download Clustered Data")
                    
                    csv_buffer = io.StringIO()
                    df_clusters.to_csv(csv_buffer, index=False)
                    csv_str = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="Download data with cluster assignments",
                        data=csv_str,
                        file_name="clustered_data.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing the file for clustering analysis: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_model_performance(pca, gmm):
    st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    This page provides metrics and visualizations to evaluate the performance of the clustering model.
    Upload your data to calculate performance metrics like silhouette score and BIC.
    """)
    
    # Upload data for performance evaluation
    uploaded_file = st.file_uploader("Choose a CSV file for model evaluation", type="csv", key="model_eval_upload")
    
    if uploaded_file is not None:
        try:
            # Load the dataset
            df = pd.read_csv(uploaded_file)
            
            # Select numeric features for evaluation
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            # Let user select columns
            selected_features = st.multiselect(
                "Choose numeric features for model evaluation",
                numeric_cols,
                default=numeric_cols
            )
            
            if selected_features:
                # Prepare data
                X = df[selected_features].values
                
                # Scale the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply PCA transformation
                X_pca = pca.transform(X_scaled)
                
                # Performance metrics section
                st.markdown("<h3 class='sub-header'>Clustering Performance Metrics</h3>", unsafe_allow_html=True)
                
                # Calculate performance metrics
                with st.spinner("Calculating performance metrics..."):
                    # Silhouette Score Analysis
                    fig = plot_silhouette_scores(X_pca)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Current model evaluation
                    st.subheader("Current Model Evaluation")
                    
                    # Get predictions from current model
                    cluster_labels = gmm.predict(X_pca)
                    
                    # Calculate silhouette score for current model
                    silhouette_avg = silhouette_score(X_pca, cluster_labels)
                
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Silhouette Score", f"{silhouette_avg:.4f}")
                        st.markdown("""
                        **Silhouette Score** ranges from -1 to 1:
                        - Values close to 1 indicate well-separated clusters
                        - Values close to 0 indicate overlapping clusters
                        - Negative values indicate incorrectly assigned samples
                        """)
                    
                    
                    # Cluster quality visualization
                    st.subheader("Cluster Quality Visualization")
                    
                    # Calculate average probability of samples belonging to their assigned cluster
                    cluster_probs = gmm.predict_proba(X_pca)
                    max_probs = np.max(cluster_probs, axis=1)
                    
                    # Create a histogram of assignment probabilities
                    fig = px.histogram(
                        x=max_probs,
                        nbins=50,
                        labels={"x": "Probability of Belonging to Assigned Cluster"},
                        title="Distribution of Cluster Assignment Probabilities",
                        color_discrete_sequence=['royalblue']
                    )
                    
                    fig.update_layout(height=500)
                    fig.add_vline(x=0.8, line_dash="dash", line_color="red")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display percentage of well-assigned samples
                    well_assigned = np.mean(max_probs >= 0.8)
                    st.metric(
                        "Well-Assigned Samples (p ≥ 0.8)", 
                        f"{well_assigned:.2%}",
                        help="Percentage of samples with at least 80% probability of belonging to their assigned cluster"
                    )
        
        except Exception as e:
            st.error(f"Error processing the file for model evaluation: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_feature_importance(pca):
    st.markdown("<h2 class='sub-header'>Feature Importance Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    This page helps you understand which features contribute most to the clustering results.
    Feature importance is derived from the PCA component loadings and other metrics.
    """)
    
    # Upload data for feature importance analysis
    uploaded_file = st.file_uploader("Choose a CSV file for feature importance analysis", type="csv", key="feature_imp_upload")
    
    if uploaded_file is not None:
        try:
            # Load the dataset
            df = pd.read_csv(uploaded_file)
            
            # Select features for analysis
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            # Let user select columns
            selected_features = st.multiselect(
                "Choose numeric features for importance analysis",
                numeric_cols,
                default=numeric_cols
            )
            
            if selected_features:
                # Feature importance from PCA loadings
                st.markdown("<h3 class='sub-header'>PCA Component Loadings</h3>", unsafe_allow_html=True)
                
                # Plot feature importance based on PCA loadings
                fig = plot_feature_importance(pca, selected_features)
                st.pyplot(fig)
                
                # Show top contributing features for each component
                st.subheader("Top Contributing Features per Component")
                
                # Get loadings
                loadings = pca.components_
                
                # Display top features for each component
                n_top_features = min(5, len(selected_features))
                
                for i in range(min(3, pca.n_components_)):
                    # Get the component loadings
                    component_loadings = pd.Series(
                        np.abs(loadings[i]),
                        index=selected_features
                    )
                    
                    # Sort by absolute value
                    top_features = component_loadings.sort_values(ascending=False).head(n_top_features)
                    
                    # Display as bar chart
                    fig = px.bar(
                        x=top_features.index,
                        y=top_features.values,
                        labels={"x": "Feature", "y": "Absolute Loading"},
                        title=f"Top {n_top_features} Features for Principal Component {i+1}",
                        color=top_features.values,
                        color_continuous_scale="viridis"
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature correlation analysis
                st.markdown("<h3 class='sub-header'>Feature Correlation Analysis</h3>", unsafe_allow_html=True)
                
                # Compute correlation matrix
                correlation = df[selected_features].corr()
                
                # Plot correlation heatmap
                fig = px.imshow(
                    correlation.values,
                    x=correlation.columns,
                    y=correlation.columns,
                    labels=dict(x="Feature", y="Feature", color="Correlation"),
                    title="Feature Correlation Heatmap",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature variance analysis
                st.markdown("<h3 class='sub-header'>Feature Variance Analysis</h3>", unsafe_allow_html=True)
                
                # Calculate and display feature variances
                variances = df[selected_features].var().sort_values(ascending=False)
                
                fig = px.bar(
                    x=variances.index,
                    y=variances.values,
                    labels={"x": "Feature", "y": "Variance"},
                    title="Feature Variance (Higher variance features may have more impact)",
                    color=variances.values,
                    color_continuous_scale="viridis"
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing the file for feature importance analysis: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_predictions(scaler, pca, gmm):
    st.markdown("<h2 class='sub-header'>Make Predictions</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    This page allows you to input new data points and predict which cluster they belong to.
    You can input values manually or upload a CSV file with new data points.
    """)
    
    # Create tabs for different input methods
    input_tab1, input_tab2 = st.tabs(["Manual Input", "File Upload"])
    
    # Define the known feature names
    feature_names = [
        "CO(GT)", "PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", 
        "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", 
        "T", "RH", "AH"
    ]
    
    with input_tab1:
        st.subheader("Enter Feature Values")
        
        # Create input fields for each feature
        n_features = len(feature_names)
        
        # Create input fields for each known feature
        feature_values = []
        
        st.markdown("**Enter values for each feature:**")
        
        # Create multiple columns for more compact display
        cols_per_row = 2
        for i in range(0, n_features, cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < n_features:
                    with cols[j]:
                        feature_value = st.number_input(
                            f"{feature_names[idx]}", 
                            value=0.0, 
                            key=f"feat_val_{idx}"
                        )
                        feature_values.append(feature_value)
        
        # Button to predict
        if st.button("Predict Cluster", key="manual_predict"):
            # Create input array
            input_data = np.array(feature_values).reshape(1, -1)
            
            # Make prediction
            cluster, probabilities, pca_data = predict_cluster(input_data, scaler, pca, gmm)
            
            # Display results
            st.subheader("Prediction Results")
            
            # Show assigned cluster
            st.markdown(f"**Assigned Cluster:** {cluster[0]}")
            
            # Show cluster probabilities
            st.subheader("Cluster Probabilities")
            
            # Create probability dataframe
            prob_df = pd.DataFrame({
                'Cluster': range(probabilities.shape[1]),
                'Probability': probabilities[0]
            })
            
            # Sort by probability
            prob_df = prob_df.sort_values('Probability', ascending=False)
            
            # Plot probabilities
            fig = px.bar(
                prob_df,
                x='Cluster',
                y='Probability',
                title="Cluster Assignment Probabilities",
                color='Probability',
                color_continuous_scale="viridis"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualize position in PCA space
            st.subheader("Position in PCA Space")
            
            # Check if we have existing data to compare against
            st.markdown("*Note: To see how this point compares to existing clusters, upload a dataset in the 'Clustering Analysis' page.*")
    
    with input_tab2:
        st.subheader("Upload New Data Points")
        
        uploaded_file = st.file_uploader("Choose a CSV file with new data points", type="csv", key="prediction_upload")
        
        if uploaded_file is not None:
            try:
                # Load the dataset
                df = pd.read_csv(uploaded_file)
                
                # Display a sample
                st.subheader("Data Sample")
                st.dataframe(df.head())
                
                # Select features for prediction
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                
                # Let user select columns
                selected_features = st.multiselect(
                    "Choose numeric features for prediction",
                    numeric_cols,
                    default=[col for col in feature_names if col in numeric_cols]  # Default to known features if present
                )
                
                if selected_features:
                    # Button to predict for all rows
                    if st.button("Predict Clusters for All Rows", key="file_predict"):
                        # Prepare data
                        X = df[selected_features].values
                        
                        # Scale and transform data
                        X_scaled = scaler.transform(X)
                        X_pca = pca.transform(X_scaled)
                        
                        # Predict clusters
                        clusters = gmm.predict(X_pca)
                        probabilities = gmm.predict_proba(X_pca)
                        
                        # Add predictions to dataframe
                        df_results = df.copy()
                        df_results['Predicted_Cluster'] = clusters
                        
                        # Add probability columns
                        for i in range(gmm.n_components):
                            df_results[f'Prob_Cluster_{i}'] = probabilities[:, i]
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(df_results)
                        
                        # Show cluster distribution
                        st.subheader("Cluster Distribution")
                        
                        cluster_counts = pd.Series(clusters).value_counts().sort_index()
                        fig = px.bar(
                            x=cluster_counts.index,
                            y=cluster_counts.values,
                            labels={"x": "Cluster", "y": "Number of Samples"},
                            title="Distribution of Predicted Clusters",
                            color=cluster_counts.index,
                            text=cluster_counts.values
                        )
                        
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Visualize in PCA space
                        st.subheader("Visualization in PCA Space")
                        
                        # Create PCA visualization
                        fig = px.scatter(
                            x=X_pca[:, 0],
                            y=X_pca[:, 1],
                            color=clusters,
                            labels={"x": "Principal Component 1", "y": "Principal Component 2", "color": "Cluster"},
                            title="PCA Visualization of Predicted Clusters",
                            color_continuous_scale=px.colors.qualitative.Bold
                        )
                        
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Option to download results
                        st.subheader("Download Prediction Results")
                        
                        csv_buffer = io.StringIO()
                        df_results.to_csv(csv_buffer, index=False)
                        csv_str = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="Download data with predictions",
                            data=csv_str,
                            file_name="prediction_results.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing the file for predictions: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()

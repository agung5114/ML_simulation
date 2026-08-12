import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine, fetch_california_housing, load_titanic
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score

st.set_page_config(page_title="ML End-to-End Simulator", layout="wide", page_icon="🤖")

# Create a directory to store models if it doesn't exist
MODEL_DIR = "saved_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

@st.cache_data
def load_clf_data(dataset_name):
    if dataset_name == "Titanic":
        data = load_titanic()
    else:
        data = load_breast_cancer()
    return pd.DataFrame(data.data, columns=data.feature_names), data.target

@st.cache_data
def load_reg_data(dataset_name):
    if dataset_name == "Diabetes":
        data = load_diabetes()
    else:
        data = fetch_california_housing()
    return pd.DataFrame(data.data, columns=data.feature_names), data.target

@st.cache_data
def load_clu_data(dataset_name):
    if dataset_name == "Iris":
        data = load_iris()
    else:
        data = load_wine()
    return pd.DataFrame(data.data, columns=data.feature_names), data.target

def plot_decision_boundary(df_X, y, model, top_2_names, title="Decision Boundary"):
    """Creates a plotly figure with a contour plot for the decision boundary on the top 2 features."""
    idx0 = df_X.columns.get_loc(top_2_names[0])
    idx1 = df_X.columns.get_loc(top_2_names[1])
    
    x_min, x_max = df_X.iloc[:, idx0].min() - 1, df_X.iloc[:, idx0].max() + 1
    y_min, y_max = df_X.iloc[:, idx1].min() - 1, df_X.iloc[:, idx1].max() + 1
    h = (x_max - x_min) / 100.0 # grid step size
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict over grid by filling other features with their means
    grid = np.tile(df_X.mean().values, (len(xx.ravel()), 1))
    grid[:, idx0] = xx.ravel()
    grid[:, idx1] = yy.ravel()
    
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    # Add contour for background
    fig.add_trace(go.Contour(x=np.arange(x_min, x_max, h), y=np.arange(y_min, y_max, h), z=Z, 
                             showscale=False, colorscale='RdBu', opacity=0.3))
    # Add scatter for data points
    fig.add_trace(go.Scatter(x=df_X.iloc[:, idx0], y=df_X.iloc[:, idx1], mode='markers',
                             marker=dict(color=y, colorscale='RdBu', line=dict(width=1, color='black')),
                             showlegend=False))
    
    fig.update_layout(title=title, xaxis_title=top_2_names[0], yaxis_title=top_2_names[1], height=500)
    return fig

st.title("🤖 ML End-to-End Pipeline Simulator")
st.markdown("Learn how to Process Data, Train, Save, Load, and Predict using Machine Learning models.")

# Sidebar navigation
st.sidebar.title("Navigation")
task = st.sidebar.radio("Select Machine Learning Task:", 
                        ["Classification", 
                         "Regression", 
                         "Clustering"])

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates the full lifecycle: \n\n1. Train a model on synthetic data. \n2. Save the 'brain' to disk. \n3. Load the model back. \n4. Predict new user inputs.")

if task == "Classification":
    st.header("1. Classification: Predicting Categories")
    st.write("Explore data, train, and compare models to predict categorical labels.")
    
    # Dataset Selection
    dataset_name = st.selectbox("Select Dataset:", ["Breast Cancer", "Wine Dataset"])
    # Clear session state if dataset changes
    if st.session_state.get('clf_dataset') != dataset_name:
        st.session_state['clf_dataset'] = dataset_name
        if 'clf_results' in st.session_state:
            del st.session_state['clf_results']
            
    df_X, y = load_clf_data(dataset_name)
    
    st.subheader("Data Overview & Descriptive Statistics")
    st.dataframe(df_X.head(), use_container_width=True)
    st.dataframe(df_X.describe(), use_container_width=True)
    
    # Train and split on ALL features
    X_train, X_test, y_train, y_test = train_test_split(df_X.values, y, test_size=0.2, random_state=42)
    
    st.subheader("Train & Compare Models")
    if st.button("Train Models", type="primary"):
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
            "SVM (Linear)": SVC(kernel='linear', probability=True, random_state=42)
        }
        results = []
        for name, clf in models.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # Hitung Metrics Evaluasi
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1})
            
            # Save model ke disk secara terpisah
            joblib.dump(clf, os.path.join(MODEL_DIR, f"clf_{name.replace(' ', '_')}.joblib"))
        
        st.session_state['clf_results'] = pd.DataFrame(results)
        st.success("Models trained on ALL features, evaluated, and saved successfully!")
        
    if 'clf_results' in st.session_state:
        # Menyorot skor terbaik (hijau) di dalam tabel metrik
        st.dataframe(st.session_state['clf_results'].style.highlight_max(subset=['Accuracy', 'F1-Score'], color='lightgreen'), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Feature Importance Rank")
        
        # Dropdown for selecting ranking algorithm based on TRAINED models
        rank_algo_clf = st.selectbox("Select Trained Model to Rank Features:", ["Random Forest", "Logistic Regression"])
        
        if rank_algo_clf == "Random Forest":
            explainer = joblib.load(os.path.join(MODEL_DIR, f"clf_Random_Forest.joblib"))
            importances = pd.Series(explainer.feature_importances_, index=df_X.columns)
            algo_desc = "Random Forest uses decision splits (Gini impurity) to calculate importance."
        else:
            explainer = joblib.load(os.path.join(MODEL_DIR, f"clf_Logistic_Regression.joblib"))
            importances = pd.Series(np.abs(explainer.coef_[0]), index=df_X.columns)
            algo_desc = "Logistic Regression uses the absolute value of its mathematical coefficients."
            
        top_5_features = importances.nlargest(5)
        
        fig_imp = px.bar(top_5_features, orientation='v', 
                         title=f"Top 5 Features ({rank_algo_clf})",
                         labels={'value': 'Importance Score', 'index': 'Feature Name'},
                         color=top_5_features.values, color_continuous_scale='viridis')
        fig_imp.update_layout(showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        top_2_names = top_5_features.index[:2].tolist()
        st.info(f"**Interpretation:** {algo_desc} We will use **{top_2_names[0]}** and **{top_2_names[1]}** for our 2D simulation below. (Other features will automatically be held at their average value).")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Simulate Prediction")
            selected_model_name = st.selectbox("Select Model for Prediction:", ["Random Forest", "Logistic Regression", "SVM (Linear)"])
            f1_input = st.number_input(f"Input {top_2_names[0]}:", value=float(round(df_X[top_2_names[0]].mean(), 2)))
            f2_input = st.number_input(f"Input {top_2_names[1]}:", value=float(round(df_X[top_2_names[1]].mean(), 2)))
            predict_btn = st.button("Predict")
            
        with col2:
            st.subheader("Prediction Result")
            model_path = os.path.join(MODEL_DIR, f"clf_{selected_model_name.replace(' ', '_')}.joblib")
            
            if predict_btn:
                if os.path.exists(model_path):
                    loaded_clf = joblib.load(model_path)
                    
                    # Build full array with means for all features (Added .copy() to make it writable)
                    new_data = df_X.mean().values.copy().reshape(1, -1)
                    idx0 = df_X.columns.get_loc(top_2_names[0])
                    idx1 = df_X.columns.get_loc(top_2_names[1])
                    new_data[0, idx0] = f1_input
                    new_data[0, idx1] = f2_input
                    
                    prediction = loaded_clf.predict(new_data)[0]
                    prob = loaded_clf.predict_proba(new_data)[0] if hasattr(loaded_clf, 'predict_proba') else None
                    
                    st.success(f"**Model Used:** {selected_model_name}")
                    st.write(f"**Prediction:** Class {prediction}")
                    if prob is not None:
                        st.write(f"**Confidence:** {prob[prediction]*100:.2f}%")
                else:
                    st.error("Model file not found. Please click 'Train Models' first!")
        
        if os.path.exists(model_path):
            st.markdown("---")
            st.subheader(f"Visualizing Decision Boundary ({selected_model_name})")
            loaded_clf = joblib.load(model_path)
            fig = plot_decision_boundary(df_X, y, loaded_clf, top_2_names, title=f"Decision Boundary - {selected_model_name}")
            st.plotly_chart(fig, use_container_width=True)

elif task == "Regression":
    st.header("2. Regression: Predicting Continuous Numbers")
    st.write("Explore data and train models to predict continuous targets.")
    
    # Dataset Selection
    dataset_name = st.selectbox("Select Dataset:", ["Diabetes", "California Housing"])
    if st.session_state.get('reg_dataset') != dataset_name:
        st.session_state['reg_dataset'] = dataset_name
        if 'reg_results' in st.session_state:
            del st.session_state['reg_results']
            
    df_X, y = load_reg_data(dataset_name)
    
    st.subheader("Data Overview & Descriptive Statistics")
    st.dataframe(df_X.head(), use_container_width=True)
    st.dataframe(df_X.describe(), use_container_width=True)
    
    # Train and split on ALL features
    X_train, X_test, y_train, y_test = train_test_split(df_X.values, y, test_size=0.2, random_state=42)
    
    st.subheader("Train & Compare Models")
    if st.button("Train Models", type="primary"):
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=10, random_state=42)
        }
        results = []
        for name, reg in models.items():
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({"Model": name, "MAE": mae, "MSE": mse, "R-Squared": r2})
            joblib.dump(reg, os.path.join(MODEL_DIR, f"reg_{name.replace(' ', '_')}.joblib"))
        
        st.session_state['reg_results'] = pd.DataFrame(results)
        st.success("Models trained on ALL features, evaluated, and saved successfully!")
        
    if 'reg_results' in st.session_state:
        # Menyorot Error terkecil (minimum) dan R2 tertinggi (maksimum)
        st.dataframe(st.session_state['reg_results'].style.highlight_min(subset=['MAE', 'MSE'], color='lightgreen').highlight_max(subset=['R-Squared'], color='lightgreen'), use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Importance Rank")
        
        rank_algo_reg = st.selectbox("Select Trained Model to Rank Features:", ["Random Forest Regressor", "Linear Regression"])

        if rank_algo_reg == "Random Forest Regressor":
            explainer = joblib.load(os.path.join(MODEL_DIR, f"reg_Random_Forest_Regressor.joblib"))
            importances = pd.Series(explainer.feature_importances_, index=df_X.columns)
        else:
            explainer = joblib.load(os.path.join(MODEL_DIR, f"reg_Linear_Regression.joblib"))
            importances = pd.Series(np.abs(explainer.coef_), index=df_X.columns)

        top_5_features = importances.nlargest(5)
        
        fig_imp = px.bar(top_5_features, orientation='v', 
                         title=f"Top 5 Features ({rank_algo_reg})",
                         labels={'value': 'Importance Score', 'index': 'Feature Name'},
                         color=top_5_features.values, color_continuous_scale='plasma')
        fig_imp.update_layout(showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        top_1_name = top_5_features.index[0]
        st.info(f"**Interpretation:** The model relies heavily on **'{top_1_name}'**. For our 1D visual simulation below, we will adjust this feature while holding all other indicators at their average.")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Simulate Prediction")
            selected_model_name = st.selectbox("Select Model for Prediction:", ["Linear Regression", "Ridge Regression", "Random Forest Regressor"])
            x_input = st.number_input(f"Input {top_1_name}:", value=float(round(df_X[top_1_name].mean(), 4)))
            predict_btn = st.button("Predict")
            
        with col2:
            st.subheader("Prediction Result")
            model_path = os.path.join(MODEL_DIR, f"reg_{selected_model_name.replace(' ', '_')}.joblib")
            
            if predict_btn:
                if os.path.exists(model_path):
                    loaded_reg = joblib.load(model_path)
                    
                    # Added .copy() to make array writable
                    new_data = df_X.mean().values.copy().reshape(1, -1)
                    idx0 = df_X.columns.get_loc(top_1_name)
                    new_data[0, idx0] = x_input
                    
                    prediction = loaded_reg.predict(new_data)[0]
                    
                    st.success(f"**Model Used:** {selected_model_name}")
                    st.write(f"**Prediction:** Target Y = {prediction:.2f}")
                else:
                    st.error("Model file not found. Please click 'Train Models' first!")

        if os.path.exists(model_path):
            st.markdown("---")
            st.subheader(f"Visualizing Best Fit Line ({selected_model_name})")
            
            loaded_reg = joblib.load(model_path)
            fig = go.Figure()
            # Data points aktual
            fig.add_trace(go.Scatter(x=df_X[top_1_name], y=y, mode='markers', name='Actual Data', marker=dict(color='blue', opacity=0.5)))
            
            # Garis prediksi berdasarkan model yang dipilih
            line_X_values = np.linspace(df_X[top_1_name].min(), df_X[top_1_name].max(), 100)
            grid = np.tile(df_X.mean().values, (100, 1))
            idx0 = df_X.columns.get_loc(top_1_name)
            grid[:, idx0] = line_X_values
            
            line_y = loaded_reg.predict(grid)
            fig.add_trace(go.Scatter(x=line_X_values, y=line_y, mode='lines', name=f'{selected_model_name} Prediction', line=dict(color='red', width=3)))
                
            fig.update_layout(xaxis_title=f"{top_1_name} Feature", yaxis_title="Target", height=500)
            st.plotly_chart(fig, use_container_width=True)

elif task == "Clustering":
    st.header("3. Clustering: Discovering Hidden Groups")
    st.write("Explore data and train clustering models.")
    
    # Dataset Selection
    dataset_name = st.selectbox("Select Dataset:", ["Iris", "Wine Dataset"])
    if st.session_state.get('clu_dataset') != dataset_name:
        st.session_state['clu_dataset'] = dataset_name
        if 'clu_results' in st.session_state:
            del st.session_state['clu_results']
            
    df_X, _ = load_clu_data(dataset_name)
    
    st.subheader("Data Overview & Descriptive Statistics")
    st.dataframe(df_X.head(), use_container_width=True)
    st.dataframe(df_X.describe(), use_container_width=True)
    
    st.subheader("Train & Compare Models")
    k_value = st.slider("Select Number of Clusters (K)", min_value=2, max_value=6, value=3)
    
    if st.button("Train Models", type="primary"):
        models = {
            "K-Means": KMeans(n_clusters=k_value, random_state=42, n_init=10),
            "Gaussian Mixture": GaussianMixture(n_components=k_value, random_state=42)
        }
        results = []
        for name, clu in models.items():
            labels = clu.fit_predict(df_X.values)
            
            sil_score = silhouette_score(df_X.values, labels)
            
            results.append({"Model": name, "Silhouette Score": sil_score})
            joblib.dump(clu, os.path.join(MODEL_DIR, f"clu_{name.replace(' ', '_')}.joblib"))
        
        st.session_state['clu_results'] = pd.DataFrame(results)
        st.success(f"Models trained on all features with K={k_value} clusters!")
        
    if 'clu_results' in st.session_state:
        st.dataframe(st.session_state['clu_results'].style.highlight_max(subset=['Silhouette Score'], color='lightgreen'), use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Importance / Selection")
        
        rank_algo_clu = st.selectbox("Select Method to Rank Features:", ["Data Variance", "PCA (Principal Component 1 Loadings)"])

        if rank_algo_clu == "Data Variance":
            importances = df_X.var()
            title_desc = "Feature Ranking by Variance"
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1).fit(df_X.values)
            importances = pd.Series(np.abs(pca.components_[0]), index=df_X.columns)
            title_desc = "Feature Ranking by PCA Loadings"

        top_features = importances.sort_values(ascending=False)
        
        fig_imp = px.bar(top_features, orientation='v', 
                         title=title_desc,
                         labels={'value': 'Score', 'index': 'Feature Name'},
                         color=top_features.values, color_continuous_scale='teal')
        fig_imp.update_layout(showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        top_2_names = top_features.index[:2].tolist()
        st.info(f"**Interpretation:** We will use **{top_2_names[0]}** and **{top_2_names[1]}** for our 2D simulation below.")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Simulate Prediction")
            selected_model_name = st.selectbox("Select Model for Prediction:", ["K-Means", "Gaussian Mixture"])
            f1_input = st.number_input(f"Input {top_2_names[0]}:", value=float(round(df_X[top_2_names[0]].mean(), 2)))
            f2_input = st.number_input(f"Input {top_2_names[1]}:", value=float(round(df_X[top_2_names[1]].mean(), 2)))
            predict_btn = st.button("Predict Cluster")
            
        with col2:
            st.subheader("Prediction Result")
            model_path = os.path.join(MODEL_DIR, f"clu_{selected_model_name.replace(' ', '_')}.joblib")
            
            if predict_btn:
                if os.path.exists(model_path):
                    loaded_clu = joblib.load(model_path)
                    
                    # Added .copy() to make array writable
                    new_data = df_X.mean().values.copy().reshape(1, -1)
                    idx0 = df_X.columns.get_loc(top_2_names[0])
                    idx1 = df_X.columns.get_loc(top_2_names[1])
                    new_data[0, idx0] = f1_input
                    new_data[0, idx1] = f2_input
                    
                    predicted_cluster = loaded_clu.predict(new_data)[0]
                    
                    st.success(f"**Model Used:** {selected_model_name}")
                    st.write(f"**Prediction:** Assigned to Cluster {predicted_cluster}")
                else:
                    st.error("Model file not found. Please click 'Train Models' first!")

        if os.path.exists(model_path):
            st.markdown("---")
            st.subheader(f"Visualizing Clusters ({selected_model_name})")
            
            loaded_clu = joblib.load(model_path)
            labels = loaded_clu.predict(df_X.values)
            
            fig = go.Figure()
            
            idx0 = df_X.columns.get_loc(top_2_names[0])
            idx1 = df_X.columns.get_loc(top_2_names[1])
            
            # Add clustered points
            n_clusters = loaded_clu.n_components if hasattr(loaded_clu, 'n_components') else loaded_clu.n_clusters
            for i in range(n_clusters):
                cluster_points = df_X.values[labels == i]
                fig.add_trace(go.Scatter(x=cluster_points[:, idx0], y=cluster_points[:, idx1], mode='markers', 
                                         name=f'Cluster {i}', marker=dict(size=8, opacity=0.7)))
            
            # Add centroids
            if hasattr(loaded_clu, 'cluster_centers_'):
                centroids = loaded_clu.cluster_centers_
                fig.add_trace(go.Scatter(x=centroids[:, idx0], y=centroids[:, idx1], mode='markers', name='Centroids',
                                         marker=dict(color='black', symbol='x', size=15, line=dict(width=2))))
            elif hasattr(loaded_clu, 'means_'):
                centroids = loaded_clu.means_
                fig.add_trace(go.Scatter(x=centroids[:, idx0], y=centroids[:, idx1], mode='markers', name='Means (Centroids)',
                                         marker=dict(color='black', symbol='x', size=15, line=dict(width=2))))
            
            fig.update_layout(xaxis_title=top_2_names[0], yaxis_title=top_2_names[1], height=500)
            st.plotly_chart(fig, use_container_width=True)

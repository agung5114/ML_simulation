import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
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

def get_classification_data():
    """Loads Breast Cancer dataset (using 2 features for 2D visualization)."""
    if 'clf_X' not in st.session_state:
        data = load_breast_cancer()
        # Use only Mean Radius (0) and Mean Texture (1) for 2D plot
        st.session_state['clf_X'] = data.data[:, :2]
        st.session_state['clf_y'] = data.target
    return st.session_state['clf_X'], st.session_state['clf_y']

def get_regression_data():
    """Loads Diabetes dataset (using BMI feature for 1D visualization)."""
    if 'reg_X' not in st.session_state:
        data = load_diabetes()
        # Use only BMI feature (index 2)
        st.session_state['reg_X'] = data.data[:, 2:3]
        st.session_state['reg_y'] = data.target
    return st.session_state['reg_X'], st.session_state['reg_y']

def get_clustering_data():
    """Loads Iris dataset (using 2 features for 2D visualization)."""
    if 'clu_X' not in st.session_state:
        data = load_iris()
        # Use Sepal Length (0) and Sepal Width (1)
        st.session_state['clu_X'] = data.data[:, :2]
    return st.session_state['clu_X']

def plot_decision_boundary(X, y, model, title="Classification Decision Boundary"):
    """Creates a plotly figure with a contour plot for the decision boundary."""
    # Define grid bounds
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.1 # grid step size
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict over grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    # Add contour for background
    fig.add_trace(go.Contour(x=np.arange(x_min, x_max, h), y=np.arange(y_min, y_max, h), z=Z, 
                             showscale=False, colorscale='RdBu', opacity=0.3))
    # Add scatter for data points
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                             marker=dict(color=y, colorscale='RdBu', line=dict(width=1, color='black')),
                             showlegend=False))
    
    fig.update_layout(title=title, xaxis_title="Feature 1", yaxis_title="Feature 2", height=500)
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
    st.write("Train and compare different models to predict Malignant (0) vs Benign (1) using Mean Radius & Mean Texture.")
    
    X, y = get_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.subheader("Train & Compare Models")
    if st.button("Train Models", type="primary"):
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42),
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
        st.success("Models trained, evaluated, and saved successfully!")
        
    if 'clf_results' in st.session_state:
        # Menyorot skor terbaik (hijau) di dalam tabel metrik
        st.dataframe(st.session_state['clf_results'].style.highlight_max(subset=['Accuracy', 'F1-Score'], color='lightgreen'), use_container_width=True)
        
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Simulate Prediction")
        # Pilihan dropdown untuk memilih algoritma yang ingin disimulasikan
        selected_model_name = st.selectbox("Select Model for Prediction:", ["Random Forest", "Logistic Regression", "SVM (Linear)"])
        f1_input = st.number_input("Input Mean Radius:", value=15.0)
        f2_input = st.number_input("Input Mean Texture:", value=20.0)
        predict_btn = st.button("Predict")
        
    with col2:
        st.subheader("Prediction Result")
        model_path = os.path.join(MODEL_DIR, f"clf_{selected_model_name.replace(' ', '_')}.joblib")
        
        if predict_btn:
            if os.path.exists(model_path):
                # Load model sesuai dengan dropdown yang dipilih pengguna
                loaded_clf = joblib.load(model_path)
                
                # Make prediction
                new_data = np.array([[f1_input, f2_input]])
                prediction = loaded_clf.predict(new_data)[0]
                prob = loaded_clf.predict_proba(new_data)[0]
                
                class_color = "Red (Malignant)" if prediction == 0 else "Blue (Benign)"
                st.success(f"**Model Used:** {selected_model_name}")
                st.write(f"**Prediction:** Class {prediction} ({class_color})")
                st.write(f"**Confidence:** {prob[prediction]*100:.2f}%")
            else:
                st.error("Model file not found. Please click 'Train Models' first!")
    
    # Plot Decision boundary bergantung pada model yang dipilih di dropdown
    if os.path.exists(model_path):
        st.markdown("---")
        st.subheader(f"Visualizing Decision Boundary ({selected_model_name})")
        loaded_clf = joblib.load(model_path)
        fig = plot_decision_boundary(X, y, loaded_clf, title=f"Decision Boundary - {selected_model_name}")
        st.plotly_chart(fig, use_container_width=True)

elif task == "Regression":
    st.header("2. Regression: Predicting Continuous Numbers")
    st.write("Train and compare models using the Diabetes dataset (BMI feature) to predict disease progression.")
    
    X, y = get_regression_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
            
            # Hitung Metrics Evaluasi
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({"Model": name, "MAE": mae, "MSE": mse, "R-Squared": r2})
            joblib.dump(reg, os.path.join(MODEL_DIR, f"reg_{name.replace(' ', '_')}.joblib"))
        
        st.session_state['reg_results'] = pd.DataFrame(results)
        st.success("Models trained, evaluated, and saved successfully!")
        
    if 'reg_results' in st.session_state:
        # Menyorot Error terkecil (minimum) dan R2 tertinggi (maksimum)
        st.dataframe(st.session_state['reg_results'].style.highlight_min(subset=['MAE', 'MSE'], color='lightgreen').highlight_max(subset=['R-Squared'], color='lightgreen'), use_container_width=True)

    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Simulate Prediction")
        selected_model_name = st.selectbox("Select Model for Prediction:", ["Linear Regression", "Ridge Regression", "Random Forest Regressor"])
        x_input = st.number_input("Input BMI (Standardized):", value=0.05)
        predict_btn = st.button("Predict")
        
    with col2:
        st.subheader("Prediction Result")
        model_path = os.path.join(MODEL_DIR, f"reg_{selected_model_name.replace(' ', '_')}.joblib")
        
        if predict_btn:
            if os.path.exists(model_path):
                loaded_reg = joblib.load(model_path)
                new_data = np.array([[x_input]])
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
        fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Actual Data', marker=dict(color='blue', opacity=0.5)))
        
        # Garis prediksi berdasarkan model yang dipilih
        line_X = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        line_y = loaded_reg.predict(line_X)
        fig.add_trace(go.Scatter(x=line_X.flatten(), y=line_y, mode='lines', name=f'{selected_model_name} Prediction', line=dict(color='red', width=3)))
            
        fig.update_layout(xaxis_title="BMI Feature (Standardized)", yaxis_title="Disease Progression (Target)", height=500)
        st.plotly_chart(fig, use_container_width=True)

elif task == "Clustering":
    st.header("3. Clustering: Discovering Hidden Groups")
    st.write("Train and compare clustering models on the Iris dataset (Sepal Length & Width).")
    
    X = get_clustering_data()
    
    st.subheader("Train & Compare Models")
    k_value = st.slider("Select Number of Clusters (K)", min_value=2, max_value=6, value=3)
    
    if st.button("Train Models", type="primary"):
        models = {
            "K-Means": KMeans(n_clusters=k_value, random_state=42, n_init=10),
            "Gaussian Mixture": GaussianMixture(n_components=k_value, random_state=42)
        }
        results = []
        for name, clu in models.items():
            labels = clu.fit_predict(X)
            
            # Silhouette Score untuk mengevaluasi kualitas cluster
            sil_score = silhouette_score(X, labels)
            
            results.append({"Model": name, "Silhouette Score": sil_score})
            joblib.dump(clu, os.path.join(MODEL_DIR, f"clu_{name.replace(' ', '_')}.joblib"))
        
        st.session_state['clu_results'] = pd.DataFrame(results)
        st.success(f"Models trained with K={k_value} clusters!")
        
    if 'clu_results' in st.session_state:
        st.dataframe(st.session_state['clu_results'].style.highlight_max(subset=['Silhouette Score'], color='lightgreen'), use_container_width=True)

    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Simulate Prediction")
        selected_model_name = st.selectbox("Select Model for Prediction:", ["K-Means", "Gaussian Mixture"])
        f1_input = st.number_input("Input Sepal Length (cm):", value=5.5)
        f2_input = st.number_input("Input Sepal Width (cm):", value=3.0)
        predict_btn = st.button("Predict Cluster")
        
    with col2:
        st.subheader("Prediction Result")
        model_path = os.path.join(MODEL_DIR, f"clu_{selected_model_name.replace(' ', '_')}.joblib")
        
        if predict_btn:
            if os.path.exists(model_path):
                loaded_clu = joblib.load(model_path)
                new_data = np.array([[f1_input, f2_input]])
                predicted_cluster = loaded_clu.predict(new_data)[0]
                
                st.success(f"**Model Used:** {selected_model_name}")
                st.write(f"**Prediction:** Assigned to Cluster {predicted_cluster}")
            else:
                st.error("Model file not found. Please click 'Train Models' first!")

    if os.path.exists(model_path):
        st.markdown("---")
        st.subheader(f"Visualizing Clusters ({selected_model_name})")
        
        loaded_clu = joblib.load(model_path)
        labels = loaded_clu.predict(X)
        
        fig = go.Figure()
        
        # Add clustered points
        n_clusters = loaded_clu.n_components if hasattr(loaded_clu, 'n_components') else loaded_clu.n_clusters
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            fig.add_trace(go.Scatter(x=cluster_points[:, 0], y=cluster_points[:, 1], mode='markers', 
                                     name=f'Cluster {i}', marker=dict(size=8, opacity=0.7)))
        
        # Add centroids berdasarkan API dari K-Means vs GMM
        if hasattr(loaded_clu, 'cluster_centers_'):
            centroids = loaded_clu.cluster_centers_
            fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', name='Centroids',
                                     marker=dict(color='black', symbol='x', size=15, line=dict(width=2))))
        elif hasattr(loaded_clu, 'means_'):
            centroids = loaded_clu.means_
            fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', name='Means (Centroids)',
                                     marker=dict(color='black', symbol='x', size=15, line=dict(width=2))))
        
        fig.update_layout(xaxis_title="Sepal Length (cm)", yaxis_title="Sepal Width (cm)", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Plot raw unlabeled data if not trained
        fig = px.scatter(x=X[:, 0], y=X[:, 1], title="Raw Unlabeled Data (Iris dataset)")
        fig.update_layout(xaxis_title="Sepal Length (cm)", yaxis_title="Sepal Width (cm)")
        fig.update_traces(marker=dict(color='grey'))
        st.plotly_chart(fig, use_container_width=True)
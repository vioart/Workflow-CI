"""
modeling.py

File ini berisi proses pelatihan model machine learning menggunakan dataset Medical Learning Resources
yang telah diproses sebelumnya, dengan pelacakan menggunakan MLflow secara lokal.
"""

import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import sys

# Mengaktifkan autologging untuk Scikit-Learn
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

def load_preprocessed_data(file_path):
    """
    Memuat data yang telah diproses dari file CSV.
    
    Args:
        file_path (str): Path ke file CSV yang telah diproses.
    
    Returns:
        pd.DataFrame: Dataset yang dimuat.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan.")
    df = pd.read_csv(file_path)
    print("Kolom DataFrame:", df.columns)
    return df

def prepare_data(df):
    """
    Mempersiapkan data untuk pelatihan model, termasuk vektorisasi teks.
    
    Args:
        df (pd.DataFrame): Dataset yang dimuat.
    
    Returns:
        tuple: Fitur (X) dan target (y) yang telah diproses.
    """
    # Misalkan kita gunakan 'type' sebagai target (klasifikasi jenis sumber daya)
    X = df['content_soup']
    y = df['type']
    
    # Vektorisasi teks menggunakan TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(X)
    
    return X, y

def train_model(X, y):
    """
    Melatih model Random Forest Classifier dan melacak dengan MLflow.
    
    Args:
        X: Fitur yang telah diproses.
        y: Target untuk pelatihan.
    
    Returns:
        object: Model yang telah dilatih.
    """
    # Membagi data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Memulai run MLflow
    with mlflow.start_run(run_name="RandomForest_Training"):
        # Inisialisasi dan latih model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrik tambahan (meskipun autolog sudah mencatat)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)

        # Log laporan klasifikasi sebagai artefak
        classification_report_str = classification_report(y_test, y_pred)
        mlflow.log_text(classification_report_str, "classification_report.txt")
        
        print(f"Akurasi model: {accuracy:.2f}")
        print("\nLaporan Klasifikasi:\n", classification_report_str)
    
    return model

if __name__ == "__main__":
    try:
        base_dir = os.getcwd()
        preprocessed_file = os.path.join(base_dir, "MLProject", "Learning_Resources_Preprocessing.csv")
        
        print("MLflow Tracking URI:", mlflow.get_tracking_uri())
        print("Current working directory:", os.getcwd())
        print("Preprocessed file path:", preprocessed_file)
        print("File exists:", os.path.exists(preprocessed_file))
        
        # Mengatur tracking URI ke direktori lokal
        mlflow_tracking_dir = os.path.abspath(os.path.join(base_dir, "mlruns"))
        
        # Konversi path ke format file:/// dengan forward slash
        tracking_uri = f"file:///{mlflow_tracking_dir.replace(os.sep, '/')}"
        mlflow.set_tracking_uri(tracking_uri)
        
        print("MLflow Tracking URI:", mlflow.get_tracking_uri())
        print("Current working directory:", base_dir)
        
        # Buat eksperimen baru dengan nama unik
        experiment_name = "Learning_Resources_Experiment_New"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            print(f"Eksperimen baru dibuat: {experiment_name}")
        except mlflow.exceptions.MlflowException as e:
            if "already exists" in str(e):
                mlflow.set_experiment(experiment_name)
                print(f"Menggunakan eksperimen yang sudah ada: {experiment_name}")
            else:
                raise e
        
        df = load_preprocessed_data(preprocessed_file)
        X, y = prepare_data(df)
        trained_model = train_model(X, y)
        
        print("\nPelatihan selesai. Cek MLflow UI untuk detail pelacakan.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}", file=sys.stderr)
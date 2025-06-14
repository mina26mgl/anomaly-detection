import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeRegressor
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
# Set page config
st.set_page_config(
    page_title="Anomaly Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stFileUploader>div>div>div>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox>div>div>div {
        background-color: white;
    }
    .stNumberInput>div>div>input {
        background-color: white;
    }
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Define your model classes here (IsolationTree, IsolationForestScratch, LOF, Autoencoder, BayesianOrdinalEncoder, ShrinkageBoostingClassifier, AnomalyDetector)
# [PASTE ALL YOUR MODEL CLASSES HERE]

def preprocess_unsupervised(df):
    df = df.copy()

    all_ordinal_cols = ['full_date', 'lib_jour', 'lib_mois', 'Birth_date', 'Activation_Date',
                        'First_Call_Date', 'Last_Call_Date', 'status_date', 'Document_Validation_Date',
                        'DOC_SCN_DT']
    ordinal_cols = [col for col in all_ordinal_cols if col in df.columns]
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    counter_cols = [col for col in cat_cols if col not in ordinal_cols]

    # Numérique
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    df_num = pd.DataFrame(num_pipe.fit_transform(df[num_cols]), columns=num_cols, index=df.index).astype(np.float32)

    # Ordinal
    if ordinal_cols:
        ordinal_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        df_ord = pd.DataFrame(ordinal_pipe.fit_transform(df[ordinal_cols]), columns=ordinal_cols, index=df.index).astype(np.float32)
    else:
        df_ord = pd.DataFrame(index=df.index)

    # Count
    if counter_cols:
        count_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', ce.CountEncoder())
        ])
        df_count = pd.DataFrame(count_pipe.fit_transform(df[counter_cols]), columns=counter_cols, index=df.index)
        df_count = np.log1p(df_count).astype(np.float32)  # sécurisation
    else:
        df_count = pd.DataFrame(index=df.index)

    df_processed = pd.concat([df_num, df_ord, df_count], axis=1)

    # Nettoyage final
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_processed.fillna(0, inplace=True)

    return df_processed
# ----- Isolation Forest Modifié pour Silhouette Score -----
class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None
    @staticmethod
    def _c(n):
        if n <= 1: 
            return 0
        return 2 * (np.log(n - 1) + 0.5772) - (2 * (n - 1) / n)
    def fit(self, X, depth=0):
        if depth >= self.max_depth or len(X) <= 1:
            return {"size": len(X)}
        n_features = X.shape[1]
        feature = np.random.randint(0, n_features)
        min_val, max_val = np.min(X.iloc[:, feature]), np.max(X.iloc[:, feature])
        if min_val == max_val:
            return {"size": len(X)}
        split = np.random.uniform(min_val, max_val)
        left = X[X.iloc[:, feature] < split]
        right = X[X.iloc[:, feature] >= split]

        return {
            "feature": feature,
            "split": split,
            "left": self.fit(left, depth + 1),
            "right": self.fit(right, depth + 1)
        }
    
    def path_length(self, x, node=None, depth=0):
        if node is None:
            node = self.tree
        if "size" in node:
            return depth + self._c(node["size"])
        
        # Make sure we don't try to access non-existent features
        if node["feature"] >= len(x):
            return depth + self._c(1)  # Return a base case if feature is out of bounds
            
        value = x[node["feature"]]

        if value < node["split"]:
            return self.path_length(x, node["left"], depth + 1)
        else:
            return self.path_length(x, node["right"], depth + 1)
    

class IsolationForestScratch:
    def __init__(self, n_trees=100, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.X = None
        self.threshold = None
        self.scores = None
        self.contamination = 0.05  # Default contamination rate
    

    def anomaly_score(self, X=None):
        X = self.X if X is None else (X.values if isinstance(X, pd.DataFrame) else X)
        scores = []
        
        for i in range(len(X)):
            x = X[i]
            if isinstance(x, pd.Series):  # Convert Series to numpy array if needed
                x = x.values
                
            lengths = [t.path_length(x) for t in self.trees]
            avg = np.mean(lengths)
            score = 2 ** (-avg / IsolationTree._c(len(self.X)))  # Call through IsolationTree
            scores.append(score)
        return np.array(scores)
    def fit(self, X):
        # Convert to numpy array and ensure 2D
        self.X = np.asarray(X)
        if len(self.X.shape) == 1:
            self.X = self.X.reshape(-1, 1)
            
        height_limit = int(np.ceil(np.log2(len(self.X))))
        self.trees = []
        
        for _ in range(self.n_trees):
            sample_idx = np.random.choice(len(self.X), len(self.X) // 2, replace=False)
            sample = self.X[sample_idx]  # Use numpy array directly
            tree = IsolationTree(self.max_depth or height_limit)
            tree.tree = tree.fit(pd.DataFrame(sample))  # Convert to DataFrame only for the tree fitting
            self.trees.append(tree)
        
        # Calculate threshold based on contamination
        scores = self.anomaly_score()
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
    
    def get_distance_matrix(self):
        """Retourne une matrice de distance basée sur les scores d'anomalie"""
        scores = self.anomaly_score()
        return cdist(scores.reshape(-1, 1), scores.reshape(-1, 1))
    
    def get_cluster_labels(self, X, threshold=None):
        """
        Returns cluster labels (0=normal, 1=anomaly) based on anomaly scores
        If no threshold is provided, uses the one calculated during fit()
        """
        scores = self.anomaly_score(X)
        threshold = threshold or self.threshold
        
        if threshold is None:
            raise ValueError("Threshold not set. Call fit() first or provide a threshold.")
            
        return (scores > threshold).astype(int)


# ----- LOF Modifié pour Silhouette Score -----
class LOF:
    def __init__(self, k=10):
        self.k = k
        self.X = None
        self.distances = None
        self.threshold = 1.5 

    def fit(self, X, contamination=0.05):
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.distances = cdist(self.X, self.X)
        
        # Calculate threshold based on contamination
        scores = self.anomaly_score()
        self.threshold = np.percentile(scores, 100 * (1 - contamination))

    def _reach_dist(self, i, j):
        dist = self.distances[i, j]
        k_dist_j = np.partition(self.distances[j], self.k)[self.k]
        return max(dist, k_dist_j)

    def _lrd(self, i):
        neighbors = np.argsort(self.distances[i])[1:self.k+1]
        reach_dists = [self._reach_dist(i, j) for j in neighbors]
        return 1 / (np.mean(reach_dists) + 1e-10)

    def anomaly_score(self, X=None):
        if X is None:
            X = self.X
        else:
            X = X.values if isinstance(X, pd.DataFrame) else X
            distances = cdist(X, self.X)
        
        lrd_scores_train = [self._lrd(i) for i in range(len(self.X))]
        scores = []
        
        for i in range(len(X)):
            if X is self.X:
                neighbors = np.argsort(self.distances[i])[1:self.k+1]
            else:
                neighbors = np.argsort(distances[i])[:self.k]
            
            lrd_x = 1 / (np.mean([
                max(distances[i][j] if X is not self.X else self.distances[i][j], 
                np.partition(self.distances[j], self.k)[self.k])
                for j in neighbors
            ]) + 1e-10)
            
            ratios = [lrd_scores_train[j] / lrd_x for j in neighbors]
            scores.append(np.mean(ratios))
        
        return np.array(scores)
        
    
    def get_distance_matrix(self):
        """Retourne la matrice de distance originale"""
        return self.distances
    
    def get_cluster_labels(self, X, threshold=None):
        """Return labels for input data X"""
        scores = self.anomaly_score(X)
        threshold = threshold or self.threshold
        return (scores > threshold).astype(int)




# ----- One-Class SVM Simplifié -----
class OneClassSVM_RBF:
    def __init__(self, gamma=0.1):
        self.gamma = gamma

    def fit(self, X):
        self.X_train = X
        self.kernel = self._rbf_kernel(X, X)

    def _rbf_kernel(self, X1, X2):
        sq_dists = cdist(X1, X2, 'sqeuclidean')
        return np.exp(-self.gamma * sq_dists)

    def decision_function(self, X):
        K = self._rbf_kernel(X, self.X_train)
        return np.mean(K, axis=1)

    def anomaly_score(self, X):
        return -self.decision_function(X)

#------Autoencoder------
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    
    def fit(self, X):
        # === 1. Vérification et conversion des données ===
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        elif not isinstance(X, np.ndarray):
            raise ValueError("Input must be DataFrame or numpy array")
        
        # === 2. Initialisation du modèle ===
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        X_tensor = torch.FloatTensor(X).to(device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
        # === 3. Entraînement ===
        self.train()
        for epoch in range(30):
            epoch_loss = 0
            for batch in loader:
                x_batch = batch[0]
                optimizer.zero_grad()
                x_recon, _ = self(x_batch)
                loss = criterion(x_recon, x_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = epoch_loss / len(loader)
                print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")
        
        # === 4. Encodage dans l'espace latent ===
        with torch.no_grad():
            self.eval()
            _, latent = self(X_tensor)
            latent_np = latent.cpu().numpy()
        
        # === 5. Clustering ===
        kmeans = KMeans(n_clusters=2, random_state=42)
        y_pred = kmeans.fit_predict(latent_np)
        print(f" Kmeans Silhouette Score: {silhouette_score(latent_np, y_pred):.3f}")
        # LOF
        lof = LOF(k=20)
        lof.fit(latent_np)
        lof_scores = lof.anomaly_score()
        lof_labels = lof.get_cluster_labels(pd.DataFrame(latent_np))
        print(f"LOF Silhouette Score: {silhouette_score(lof.get_distance_matrix(), lof_labels, metric='precomputed'):.3f}")
        
        # Isolation Forest
        iso_forest = IsolationForestScratch(n_trees=100)
        iso_forest.fit(pd.DataFrame(latent_np))
        iso_scores = iso_forest.anomaly_score(pd.DataFrame(latent_np))
        iso_labels = iso_forest.get_cluster_labels(pd.DataFrame(latent_np))
        print(f"Isolation Forest Silhouette Score: {silhouette_score(iso_forest.get_distance_matrix(), iso_labels, metric='precomputed'):.3f}")
        self.latent_np = latent_np
        self.lof = lof
        self.iforest = iso_forest
        self.lof_labels = lof_labels
        self.iso_labels = iso_labels
        return {
            'latent': pd.DataFrame(latent_np),
            'cluster_labels': y_pred,
            'lof_scores': lof_scores,
            'iso_scores': iso_scores,
            'iso_labels': iso_labels,
            'lof_labels': lof_labels
        }
    def encode(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        elif not isinstance(X, np.ndarray):
            raise ValueError("Input must be DataFrame or numpy array")

        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            latent = self.encoder(X_tensor)
            return latent.cpu().numpy()

    def reconstruct(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)

        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            X_recon = self.decoder(self.encoder(X_tensor))
            return X_recon.cpu().numpy()

#--------META MODELE------
class BayesianOrdinalEncoder:
    def __init__(self, cols):
        self.cols = cols
        self.encoder = None

    def fit_transform(self, X, y):
        X_ = X.copy()
        self.encoder = ce.LeaveOneOutEncoder(cols=self.cols, random_state=42, sigma=0.1)
        return self.encoder.fit_transform(X_, y)

    def transform(self, X):
        X_ = X.copy()
        return self.encoder.transform(X_)
class ShrinkageBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, stack_model=None, fast_mode=True):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.tree_weights = []
        self.scaler = StandardScaler()
        self.encoder = None
        
        self.fast_mode = fast_mode
        
        if stack_model is None:
            from sklearn.ensemble import RandomForestClassifier
            self.stack_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.stack_model = stack_model

    def _preprocess(self, X):
        X = X.copy()
        for col in X.columns:
            col_str = str(col)
            if 'date' in col_str.lower():
                try:
                    X[col] = pd.to_datetime(X[col], errors='coerce').astype('int64') // 10**9
                except Exception:
                    pass
        
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna('missing')
            else:
                X[col] = X[col].fillna(X[col].median())
        return X

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        cat_cols = X.select_dtypes(include='object').columns.tolist()
        if cat_cols:
            self.encoder = BayesianOrdinalEncoder(cols=cat_cols)
            X = self.encoder.fit_transform(X, y)

        self.feature_names_ = X.columns.tolist()

        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        F = np.zeros(len(y))
        self.trees = []
        self.tree_weights = []

        for i in range(self.n_estimators):
            p = 1 / (1 + np.exp(-F))
            grad = y - p
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, grad)

            pred = tree.predict(X)
            delta = self.learning_rate * pred
            F += delta

            ll = log_loss(y, 1 / (1 + np.exp(-F)))
            weight = 1 / (ll + 1e-5)
            self.trees.append(tree)
            self.tree_weights.append(weight)

        stacked_preds = self._get_tree_outputs(X)
        if self.stack_model is None:
            from catboost import CatBoostClassifier
            self.stack_model = CatBoostClassifier(
                iterations=50,
                learning_rate=0.1,
                depth=4,
                verbose=0,
                random_seed=42,
                task_type='GPU' if self.fast_mode else 'CPU'
            )
        # Fit le stack model une seule fois
        self.stack_model.fit(stacked_preds, y)

        # Calibration des probabilités : "cv='prefit'" car CatBoost est déjà entraîné
        # self.stack_model = CalibratedClassifierCV(self.stack_model, method="sigmoid", cv="prefit")

        # # Fit du calibrateur uniquement (pas le modèle CatBoost)
        # self.stack_model.fit(stacked_preds, y)

    def _get_tree_outputs(self, X):
        outputs = [tree.predict(X) * w for tree, w in zip(self.trees, self.tree_weights)]
        return np.vstack(outputs).T

    def predict_proba(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        X = self._preprocess(X)
        if self.encoder:
            X = self.encoder.transform(X)

        X = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names_)
        stacked_preds = self._get_tree_outputs(X)
        return self.stack_model.predict_proba(stacked_preds)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
class AnomalyDetector:
    def __init__(self, autoencoder, contamination=0.05, features=None):
        """
        autoencoder: module PyTorch avec fit, encode, predict
        meta_model: modèle scikit-learn compatible (fit, predict, predict_proba)
        contamination: pour seuil des erreurs de reconstruction
        features: liste des colonnes à utiliser (subset de X.columns)
        """
        self.autoencoder = autoencoder
        self.meta_model = ShrinkageBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, stack_model=None, fast_mode=True)
        self.contamination = contamination
        self.features = features
        self.fit_outputs = None
    def fit(self, X):
        if self.features:
            X = X[self.features]

        self.fit_outputs = self.autoencoder.fit(X)
        
        lof_labels = self.autoencoder.lof_labels
        iso_labels = self.autoencoder.iso_labels

        X_np = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else X
        recon = self.autoencoder.reconstruct(X_np)

        ae_errors = np.mean((X_np - recon) ** 2, axis=1)
        ae_thresh = np.percentile(ae_errors, 100 * (1 - self.contamination))
        ae_labels = (ae_errors > ae_thresh).astype(int)

        df_meta = pd.DataFrame({
            "ae_label": ae_labels,
            "lof_label": lof_labels,
            "iso_label": iso_labels
        })

        df_meta["meta_label"] = (df_meta[["ae_label", "lof_label", "iso_label"]].sum(axis=1) >= 2).astype(int)

        self.meta_model.fit(df_meta[["ae_label", "lof_label", "iso_label"]], df_meta["meta_label"])

    def predict(self, X):
        if self.features:
            X = X[self.features]

        X_np = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else X
        recon = self.autoencoder.reconstruct(X_np)

        ae_errors = np.mean((X_np - recon) ** 2, axis=1)
        ae_thresh = np.percentile(ae_errors, 100 * (1 - self.contamination))
        ae_labels = (ae_errors > ae_thresh).astype(int)

        latent = self.autoencoder.encode(X_np)
        lof_labels = self.autoencoder.lof.get_cluster_labels(pd.DataFrame(latent))
        iso_labels = self.autoencoder.iforest.get_cluster_labels(pd.DataFrame(latent))

        df_pred = pd.DataFrame({
            "ae_label": ae_labels,
            "lof_label": lof_labels,
            "iso_label": iso_labels
        })

        return self.meta_model.predict(df_pred)

    def predict_proba(self, X):
        if self.features:
            X = X[self.features]

        X_np = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else X
        recon = self.autoencoder.reconstruct(X_np)

        ae_errors = np.mean((X_np - recon) ** 2, axis=1)
        ae_thresh = np.percentile(ae_errors, 100 * (1 - self.contamination))
        ae_labels = (ae_errors > ae_thresh).astype(int)

        latent = self.autoencoder.encode(X_np)
        lof_labels = self.autoencoder.lof.get_cluster_labels(pd.DataFrame(latent))
        iso_labels = self.autoencoder.iforest.get_cluster_labels(pd.DataFrame(latent))

        df_pred = pd.DataFrame({
            "ae_label": ae_labels,
            "lof_label": lof_labels,
            "iso_label": iso_labels
        })

        return self.meta_model.predict_proba(df_pred)
    def evaluate_unsupervised(self, X):
        """
        Évalue le métamodèle sans labels, à l'aide de critères non supervisés.
        Retourne un dictionnaire contenant les résultats d'évaluation.
        """
        results = {}
        
        if self.features:
            X = X[self.features]

        X_np = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else X
        recon = self.autoencoder.reconstruct(X_np)
        ae_errors = np.mean((X_np - recon) ** 2, axis=1)
        ae_thresh = np.percentile(ae_errors, 100 * (1 - self.contamination))
        ae_labels = (ae_errors > ae_thresh).astype(int)

        latent = self.autoencoder.encode(X_np)
        lof_labels = self.autoencoder.lof.get_cluster_labels(pd.DataFrame(latent))
        iso_labels = self.autoencoder.iforest.get_cluster_labels(pd.DataFrame(latent))

        df_meta = pd.DataFrame({
            "ae_label": ae_labels,
            "lof_label": lof_labels,
            "iso_label": iso_labels,
            "recon_error": ae_errors
        })

        df_meta["meta_pred"] = self.meta_model.predict(df_meta[["ae_label", "lof_label", "iso_label"]])

        # Score de silhouette sur l'espace latent
        try:
            sil_score = silhouette_score(latent, df_meta["meta_pred"])
            results["Silhouette Score"] = f"{sil_score:.4f}"
        except Exception as e:
            results["Silhouette Error"] = str(e)

        # Taux moyen de désaccord entre les modèles de base et le métamodèle
        df_meta["disagreement"] = (
            (df_meta["meta_pred"] != df_meta["ae_label"]).astype(int) +
            (df_meta["meta_pred"] != df_meta["lof_label"]).astype(int) +
            (df_meta["meta_pred"] != df_meta["iso_label"]).astype(int)
        )
        results["Mean Disagreement Rate"] = f"{df_meta['disagreement'].mean():.4f}"

        # Statistiques d'erreur de reconstruction
        error_stats = df_meta.groupby("meta_pred")["recon_error"].describe().to_dict()
        results["Reconstruction Error Stats"] = error_stats

        return results


        

def main():
    st.title("🛡️ Anomaly Detection System")
    st.markdown("""
    This system detects anomalies in your data using a hybrid approach combining:
    - Autoencoder reconstruction errors
    - Isolation Forest
    - Local Outlier Factor (LOF)
    - Meta-model ensemble
    """)
    
    # Sidebar for file upload and parameters
    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
        
        st.subheader("Model Parameters")
        contamination = st.slider("Contamination rate", 0.01, 0.5, 0.05, 0.01)
        latent_dim = st.slider("Latent dimension", 2, 32, 16, 2)
        n_trees = st.slider("Number of trees", 50, 500, 100, 50)
        
        if st.button("ℹ️ About"):
            st.info("""
            This anomaly detection system combines multiple techniques:
            1. Autoencoder for reconstruction errors
            2. Isolation Forest for isolation-based detection
            3. LOF for density-based detection
            4. Meta-model to combine predictions
            
            Adjust parameters in the sidebar and upload your data to begin.
            """)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file,sep=';')
            
            # Show data preview
            st.subheader("📊 Data Preview")
            st.write(f"Dataset shape: {df.shape}")
            st.dataframe(df.head())
            
            # Feature selection
            st.subheader("🔍 Feature Selection")
            all_features = df.columns.tolist()
            selected_features = st.multiselect(
                "Select features for anomaly detection",
                all_features,
                default=all_features[:min(9, len(all_features))]
            )
            
            if not selected_features:
                st.warning("Please select at least one feature!")
                return
                
            # Preprocess data
            with st.spinner("Preprocessing data..."):
                X_preprocessed = preprocess_unsupervised(df[selected_features])
                st.success("Data preprocessed successfully!")
                
            # Train-test split
            test_size = st.slider("Test set size (%)", 10, 50, 20, 5)
            X_train, X_test = train_test_split(X_preprocessed, test_size=test_size/100, random_state=42)
            
            # Model training
            if st.button("🚀 Train Model"):
                with st.spinner("Training anomaly detection model..."):
                    # Initialize and train model
                    autoencoder = Autoencoder(input_dim=len(selected_features), latent_dim=latent_dim)
                    anomaly_model = AnomalyDetector(
                        autoencoder=autoencoder,
                        contamination=contamination,
                        features=selected_features
                    )
                    
                    anomaly_model.fit(X_train)
                    st.success("Model trained successfully!")
                    
                    # Make predictions
                    with st.spinner("Detecting anomalies..."):
                        predictions = anomaly_model.predict(X_test)
                        proba = anomaly_model.predict_proba(X_test)
                        
                        # Add results to test set
                        results = X_test.copy()
                        results["Anomaly_Score"] = proba[:, 1]
                        results["Anomaly_Prediction"] = predictions
                        results["Anomaly_Prediction"] = results["Anomaly_Prediction"].map({0: "Normal", 1: "Anomaly"})
                        
                        # Show results
                        st.subheader("🔎 Detection Results")
                        st.write(f"Detected {sum(predictions)} anomalies ({sum(predictions)/len(predictions):.1%})")
                        
                        # Display anomaly table
                        st.dataframe(
                            results.sort_values("Anomaly_Score", ascending=False).head(50)
                        )
                        
                        # Download button
                        csv = results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="anomaly_detection_results.csv",
                            mime="text/csv"
                        )
                        
                        # # Visualizations
                        # st.subheader("📈 Visualizations")
                        
                        # col1, col2 = st.columns(2)
                        
                        # with col1:
                        #     st.bar_chart(results["Anomaly_Prediction"].value_counts())
                            
                        # with col2:
                        #     st.line_chart(results["Anomaly_Score"])
                        
                        # Model evaluation
                        # Model evaluation
            st.subheader("📝 Model Evaluation")
            with st.spinner("Evaluating model performance..."):
                eval_results = anomaly_model.evaluate_unsupervised(X_test)
                
                # Afficher les résultats dans l'interface
                st.write("### Model Performance Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Silhouette Score", eval_results.get("Silhouette Score", "N/A"))
                    st.metric("Mean Disagreement Rate", eval_results.get("Mean Disagreement Rate", "N/A"))
                
                with col2:
                    st.write("### Reconstruction Error Statistics")
                    if "Reconstruction Error Stats" in eval_results:
                        error_stats = pd.DataFrame(eval_results["Reconstruction Error Stats"])
                        st.dataframe(error_stats)
                    else:
                        st.warning("Could not compute reconstruction error statistics")
                
                if "Silhouette Error" in eval_results:
                    st.warning(f"Silhouette score could not be computed: {eval_results['Silhouette Error']}")
                            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload a CSV file to get started. Use the sidebar to configure the model.")

if __name__ == "__main__":
    main()
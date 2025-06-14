import streamlit as st
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import category_encoders as ce
from scipy.spatial.distance import cdist
st.set_page_config(
    page_title="Sélection de Features pour Détection d'Anomalies", 
    layout="wide"
)
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

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def preprocess_data(df, features):
    df_subset = df[features].copy()

    all_ordinal_cols = ['full_date', 'lib_jour', 'lib_mois', 'Birth_date', 'Activation_Date',
                        'First_Call_Date', 'Last_Call_Date', 'status_date', 'Document_Validation_Date',
                        'DOC_SCN_DT']
    ordinal_cols = [col for col in all_ordinal_cols if col in df_subset.columns]
    num_cols = df_subset.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_subset.select_dtypes(include='object').columns.tolist()
    counter_cols = [col for col in cat_cols if col not in ordinal_cols]

    # Pipeline numérique
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    df_num = pd.DataFrame(num_pipe.fit_transform(df_subset[num_cols]), columns=num_cols, index=df_subset.index).astype(np.float32)

    # Pipeline ordinal
    if ordinal_cols:
        ordinal_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        df_ord = pd.DataFrame(ordinal_pipe.fit_transform(df_subset[ordinal_cols]), columns=ordinal_cols, index=df_subset.index).astype(np.float32)
    else:
        df_ord = pd.DataFrame(index=df_subset.index)

    # Pipeline count encoding
    if counter_cols:
        count_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', ce.CountEncoder())
        ])
        df_count = pd.DataFrame(count_pipe.fit_transform(df_subset[counter_cols]), columns=counter_cols, index=df_subset.index)
        df_count = np.log1p(df_count).astype(np.float32)
    else:
        df_count = pd.DataFrame(index=df_subset.index)

    df_processed = pd.concat([df_num, df_ord, df_count], axis=1)

    # Nettoyage final
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_processed.fillna(0, inplace=True)

    return df_processed.values


def compute_sse(data, centers, labels):
    return np.sum((data - centers[labels]) ** 2)

def noise_based_kmeans(X, max_iter=50, rmin=0.1, rmax=0.5, K_range=(2, 10)):
    N, D = X.shape
    if N < 2:
        raise ValueError("Not enough data points for clustering.")

    best_sse = float('inf')
    best_labels = None
    best_centers = None

    NS = max(1, int(np.sqrt(N) * max_iter))  # Avoid zero
    sqrtN = max(1, int(np.sqrt(N)))
    restart = max(1, max_iter // sqrtN)
    decrease = (rmax - rmin) / max(1, (max_iter - 1))
    noise_radius = rmax

    for iteration in range(max_iter):
        K = random.randint(*K_range)
        if K >= N:
            K = N - 1  # Avoid having more clusters than data points

        init_indices = np.random.choice(range(N), K, replace=False)
        centers = X[init_indices]

        noise = np.random.normal(loc=0.0, scale=noise_radius, size=centers.shape)
        centers += noise

        distances = euclidean_distances(X, centers)
        labels = np.argmin(distances, axis=1)

        new_centers = []
        for i in range(K):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                # Fallback: reinitialize center if cluster is empty
                new_centers.append(X[random.randint(0, N - 1)])
            else:
                new_centers.append(cluster_points.mean(axis=0))
        new_centers = np.array(new_centers)

        sse = compute_sse(X, new_centers, labels)

        if sse < best_sse:
            best_sse = sse
            best_labels = labels
            best_centers = new_centers

        noise_radius = max(noise_radius - decrease, rmin)
        if iteration % restart == 0:
            noise_radius = rmax

    return best_labels, best_centers, best_sse



def visualize_clusters(X, labels, title="Cluster Visualization"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    fig.colorbar(scatter, ax=ax, label='Cluster')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)


# # Example Usage
# for name, features in feature_sets.items():
#     try:
#         print(f"\nRunning on {name}")
#         X = preprocess_data(df, features)
#         labels, centers, sse = noise_based_kmeans(X)
#         silhouette = silhouette_score(X, labels)
#         print(f"Silhouette Score: {silhouette:.4f}, SSE: {sse:.2f}, Clusters: {len(set(labels))}")
        
#         # Show cluster plot
#         visualize_clusters(X, labels, title=f"Clusters for {name}")

#     except Exception as e:
#         print(f"Failed on {name}: {e}")
        

class FeatureSelectorForAnomaly:
    def __init__(self, n_features=40, corr_threshold=0.85):
        self.n_features = n_features
        self.corr_threshold = corr_threshold
        self.selected_features_stage1 = []
        self.selected_features_final = []
        self.preprocessor = None
        self.all_features = []

    def _prepare_data(self, df):
        df = df.copy()

        all_ordinal_cols = ['full_date', 'lib_jour', 'lib_mois', 'Birth_date', 'Activation_Date',
                            'First_Call_Date', 'Last_Call_Date', 'status_date', 'Document_Validation_Date',
                            'DOC_SCN_DT']
        ordinal_cols = [col for col in all_ordinal_cols if col in df.columns]
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        counter_cols = [col for col in cat_cols if col not in ordinal_cols]

        # Pipeline numérique
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        df_num = pd.DataFrame(num_pipe.fit_transform(df[num_cols]), columns=num_cols, index=df.index).astype(np.float32)

        # Pipeline ordinal
        if ordinal_cols:
            ordinal_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            df_ord = pd.DataFrame(ordinal_pipe.fit_transform(df[ordinal_cols]), columns=ordinal_cols, index=df.index).astype(np.float32)
        else:
            df_ord = pd.DataFrame(index=df.index)

        # Pipeline count encoding
        if counter_cols:
            count_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', ce.CountEncoder())
            ])
            df_count = pd.DataFrame(count_pipe.fit_transform(df[counter_cols]), columns=counter_cols, index=df.index)
            df_count = np.log1p(df_count).astype(np.float32)
        else:
            df_count = pd.DataFrame(index=df.index)

        df_processed = pd.concat([df_num, df_ord, df_count], axis=1)

        # Nettoyage final
        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_processed.fillna(0, inplace=True)

        self.all_features = df_processed.columns.tolist()
        return df_processed.values, self.all_features


    def _remove_correlated(self, X_df, features, threshold=0.85):
        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        return [feat for feat in features if feat not in to_drop]

    
    def _unsupervised_feature_scores(self, X, features):
        scores = {}

        # 1. Variance
        variances = np.var(X, axis=0)
        scores['variance'] = variances / variances.max()

        # 2. Isolation Forest scratch
        iso = IsolationForestScratch(n_trees=100)
        iso.fit(X)
        scores['isoforest'] = np.abs(iso.anomaly_score(X)).mean(axis=0)

        # 3. LOF + PCA
        pca = PCA(n_components=min(10, X.shape[1]))
        X_pca = pca.fit_transform(X)
        
        lof = LOF(k=20)
        lof.fit(X_pca)
        lof_scores = lof.anomaly_score()

        # Seuil basé sur le 90e percentile
        labels = (lof_scores > np.percentile(lof_scores, 90)).astype(int)

        scores['pca_info'] = pd.Series(
            mutual_info_classif(X, labels, discrete_features=False),
            index=features
        )

        all_scores = pd.DataFrame(scores, index=features)
        final_score = all_scores.mean(axis=1)
        return final_score.sort_values(ascending=False)

    
    def select_top_features(self, df):
        print("[Étape 1] Préparation des données...")
        X, features = self._prepare_data(df)
        X_df = pd.DataFrame(X, columns=features)

        print("[Étape 2] Calcul des scores de features non supervisés...")
        score_series = self._unsupervised_feature_scores(X, features)

        print("[Étape 3] Sélection des", self.n_features, "meilleures features brutes...")
        top_feats = score_series.head(self.n_features).index.tolist()

        print("[Étape 4] Suppression des features corrélées (>", self.corr_threshold, ")...")
        reduced_feats = self._remove_correlated(X_df[top_feats], top_feats, threshold=self.corr_threshold)
        self.selected_features_stage1 = reduced_feats[:self.n_features]
        print("[Résultat] Features sélectionnées (étape 1) :", len(self.selected_features_stage1))
        return self.selected_features_stage1
        

    def _evaluate_who(self, X, individual):
        selected = np.where(individual == 1)[0]
        if len(selected) < 2:
            return np.inf  # Penalize small subsets

        subset = X[:, selected]
        try:
            # 1. IsolationForest-based clustering
            iso_labels = IsolationForest(random_state=0).fit_predict(subset)
            if len(np.unique(iso_labels)) < 2:
                return np.inf  # Penalize single-cluster solutions
            sil_iso = silhouette_score(subset, iso_labels)

            # 2. Noise-based KMeans clustering
            nbk_labels, _, _ = noise_based_kmeans(subset)
            if len(np.unique(nbk_labels)) < 2:
                return np.inf
            sil_nbk = silhouette_score(subset, nbk_labels)

            # Combine the two silhouette scores
            avg_sil = (sil_iso + sil_nbk) / 2.0

            return -avg_sil + 0.001 * len(selected)  # Minimize loss = -score + complexity_penalty

        except Exception as e:
            return np.inf


    def optimize_with_who(self, X, n_horses=40, n_generations=20, n_stallions=5, mutation_rate=0.1):
        print("[WHO] Optimisation via Wild Horse Optimizer")
        feature_indices = [self.all_features.index(f) for f in self.selected_features_stage1]
        n_features = len(feature_indices)
        # Convert to numpy array first if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            X_sel = X.iloc[:, feature_indices].values.astype(np.float32)
        else:
            X_sel = X[:, feature_indices].astype(np.float32)

        # 1. Initialize population (horses)
        population = np.random.binomial(1, 0.5, size=(n_horses, n_features))
        
        # 2. Evaluate initial population
        fitness = np.array([self._evaluate_who(X_sel, ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_score = fitness[best_idx]

        for gen in tqdm(range(n_generations), desc="WHO Generations"):
            # 3. Select stallions (best solutions)
            stallion_indices = np.argsort(fitness)[:n_stallions]
            stallions = population[stallion_indices]

            # 4. Group into harems (assign each horse to a stallion)
            harem_sizes = np.random.randint(1, n_horses - n_stallions + 1, size=n_stallions)
            harem_sizes = (harem_sizes / harem_sizes.sum() * (n_horses - n_stallions)).astype(int)
            harem_sizes[-1] = n_horses - n_stallions - harem_sizes[:-1].sum()  # Adjust last group

            # 5. Grazing behavior (exploration)
            new_population = []
            for i, stallion in enumerate(stallions):
                harem_size = harem_sizes[i]
                harem = np.random.binomial(1, 0.5, size=(harem_size, n_features))  # Random exploration
                harem = np.where(np.random.rand(harem_size, n_features) < 0.3, stallion, harem)  # Follow stallion
                new_population.append(harem)
            
            new_population = np.vstack([stallions] + new_population)
            
            # 6. Breeding (crossover between stallions)
            for i in range(1, n_stallions):
                crossover_point = np.random.randint(1, n_features)
                new_population[i, :crossover_point] = new_population[0, :crossover_point]  # Alpha stallion influence

            # 7. Mutation
            mutation_mask = np.random.rand(n_horses, n_features) < mutation_rate
            new_population = np.where(mutation_mask, 1 - new_population, new_population)

            # 8. Update population and fitness
            population = new_population
            fitness = np.array([self._evaluate_who(X_sel, ind) for ind in population])
            
            # 9. Track best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_score:
                best_solution = population[current_best_idx].copy()
                best_score = fitness[current_best_idx]
            print(f"[WHO] {len(self.selected_features_final)} features selected | Best score: {best_score:.4f}")
        self.selected_features_final = [self.selected_features_stage1[i] for i in range(n_features) if best_solution[i] == 1]
        print(f"[WHO] {len(self.selected_features_final)} features selected | Best score: {best_score:.4f}")
        return self.selected_features_final

    
    def transform_final(self, df):
        # Use the same preprocessing as in _prepare_data
        X_processed, _ = self._prepare_data(df)
        
        # Get indices of selected features
        feat_idx = [self.all_features.index(f) for f in self.selected_features_final]
        
        # Return only the selected columns
        return X_processed[:, feat_idx]
# Configuration de la page


# Titre de l'application
st.title("Système de Sélection de Features pour Détection d'Anomalies")

# Section pour uploader le fichier
st.header("1. Chargement des Données")
uploaded_file = st.file_uploader("Téléversez votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Lecture des données
        df = pd.read_csv(uploaded_file, sep = ';')
        st.success("Fichier chargé avec succès!")
        
        # Aperçu des données
        st.subheader("Aperçu des Données")
        st.write(df.head())
        
        # Paramètres de configuration
        st.header("2. Configuration des Paramètres")
        col1, col2 = st.columns(2)
        
        with col1:
            n_features = st.slider("Nombre de features à sélectionner", 10, 100, 40)
            corr_threshold = st.slider("Seuil de corrélation", 0.7, 0.99, 0.85)
        
        with col2:
            n_horses = st.slider("Nombre de chevaux (WHO)", 10, 100, 40)
            n_generations = st.slider("Nombre de générations (WHO)", 5, 50, 20)
        
        # Bouton pour lancer le processus
        if st.button("Lancer la Sélection de Features"):
            with st.spinner("Traitement en cours..."):
                # Initialisation du sélecteur
                selector = FeatureSelectorForAnomaly(
                    n_features=n_features, 
                    corr_threshold=corr_threshold
                )
                
                # Étape 1: Sélection initiale
                st.header("3. Résultats de Sélection")
                st.subheader("Étape 1: Sélection Initiale")
                
                selected_stage1 = selector.select_top_features(df)
                st.write(f"Nombre de features sélectionnées: {len(selected_stage1)}")
                st.write("Features sélectionnées:")
                st.write(selected_stage1)
                
                # Étape 2: Optimisation WHO
                st.subheader("Étape 2: Optimisation avec WHO")
                
                X_all, features = selector._prepare_data(df)
                selected_final = selector.optimize_with_who(
                    X_all,
                    n_horses=n_horses,
                    n_generations=n_generations
                )
                
                st.write(f"Nombre de features finales: {len(selected_final)}")
                st.write("Features finales sélectionnées:")
                st.write(selected_final)
                
                # Visualisation des résultats
                st.subheader("Visualisation des Résultats")
                
                try:
                    X_final = selector.transform_final(df)
                    
                    # Réduction de dimension pour visualisation
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_final)
                    
                    # Création du DataFrame pour plot
                    plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                    plot_df['Features'] = " | ".join(selected_final[:3]) + "..."  # Affiche les 3 premières
                    
                    # Plot
                    import plotly.express as px
                    fig = px.scatter(
                        plot_df, x='PC1', y='PC2', 
                        title="Projection des Données avec les Features Sélectionnées",
                        hover_name='Features'
                    )
                    st.plotly_chart(fig)
                    
                except Exception as e:
                    st.warning(f"Visualisation non disponible: {str(e)}")
                
                # Téléchargement des résultats
                st.subheader("Téléchargement des Résultats")
                
                # Création du rapport
                report = {
                    "Paramètres": {
                        "Nombre de features demandé": n_features,
                        "Seuil de corrélation": corr_threshold,
                        "Nombre de chevaux (WHO)": n_horses,
                        "Nombre de générations (WHO)": n_generations
                    },
                    "Features_initiales": selected_stage1,
                    "Features_finales": selected_final
                }
                
                # Conversion en DataFrame pour export
                report_df = pd.DataFrame.from_dict({
                    "Type": ["Paramètres"]*4 + ["Features initiales"]*len(selected_stage1) + ["Features finales"]*len(selected_final),
                    "Valeur": list(report["Paramètres"].values()) + selected_stage1 + selected_final
                })
                
                # Bouton de téléchargement
                st.download_button(
                    label="Télécharger le Rapport",
                    data=report_df.to_csv(index=False).encode('utf-8'),
                    file_name="rapport_selection_features.csv",
                    mime="text/csv"
                )
                
                st.success("Processus terminé avec succès!")
    
    except Exception as e:
        st.error(f"Une erreur est survenue: {str(e)}")

else:
    st.info("Veuillez téléverser un fichier CSV pour commencer.")
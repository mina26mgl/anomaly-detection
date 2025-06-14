import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_curve, auc)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import label_binarize
# Set page config
st.set_page_config(page_title="Logic Attention Network Trainer", layout="wide")

# --- Model Definitions ---
class Fuzzifier(nn.Module):
    def __init__(self, input_dim, centers=3):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(0, 1, centers).repeat(input_dim, 1), requires_grad=True)
        self.sigma = nn.Parameter(torch.ones(input_dim, centers) * 0.1, requires_grad=True)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, features, 1)
        diff = (x - self.centers) ** 2
        return torch.exp(-diff / (2 * self.sigma ** 2))  # (batch, features, centers)

class LogicAttention(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, embed_dim)
        self.key = nn.Linear(input_dim, embed_dim)
        self.value = nn.Linear(input_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)
        pooled = attended.mean(dim=1)  # Global average pooling
        return pooled, weights

class LogicAttentionNet(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_classes=5, centers=3):
        super().__init__()
        self.fuzzifier = Fuzzifier(input_dim, centers)
        self.linear_pre = nn.Linear(input_dim * centers, input_dim)
        self.logic_attention = LogicAttention(input_dim, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        fuzzified = self.fuzzifier(x)  # (batch, features, centers)
        flat_fuzz = fuzzified.flatten(start_dim=1)  # (batch, features * centers)
        embed = self.linear_pre(flat_fuzz)  # (batch, features)
        pooled, attn_weights = self.logic_attention(embed.unsqueeze(1))  # (batch, 1, features) -> pour compatibilité
        out = self.mlp(pooled)
        return out, attn_weights

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_pipe = None
        self.encoder = None
        self.num_cols = []
        self.cat_cols = []

    def fit(self, X, y=None):
        X = X.copy()
        self.num_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = X.select_dtypes(include='object').columns.tolist()

        # Pipeline pour colonnes numériques
        self.num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.num_pipe.fit(X[self.num_cols])

        # Encoder pour colonnes catégorielles
        df_cat = X[self.cat_cols].fillna(X[self.cat_cols].mode().iloc[0])
        if y is None:
            y = np.zeros(len(X))  # y factice
        self.encoder = ce.LeaveOneOutEncoder(cols=self.cat_cols)
        self.encoder.fit(df_cat, y)

        return self

    def transform(self, X):
        X = X.copy()

        # Colonnes numériques
        X_num = pd.DataFrame(self.num_pipe.transform(X[self.num_cols]), columns=self.num_cols, index=X.index)

        # Colonnes catégorielles
        df_cat = X[self.cat_cols].fillna(X[self.cat_cols].mode().iloc[0])
        X_cat = self.encoder.transform(df_cat)

        # Concaténation
        X_final = pd.concat([X_num, X_cat], axis=1)

        return X_final

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class DataPreprocessor:
    def __init__(self):
        self.X_processor = Preprocessor()
        self.y_encoder = LabelEncoder()

    def fit(self, X, y):
        # Fit Preprocessor et LabelEncoder
        self.X_processor.fit(X, y)
        self.y_encoder.fit(y)
        return self

    def transform(self, X, y=None):
        X_transformed = self.X_processor.transform(X)
        y_transformed = self.y_encoder.transform(y) if y is not None else None
        return X_transformed, y_transformed

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform_y(self, y_encoded):
        return self.y_encoder.inverse_transform(y_encoded)

class CSVDataset(Dataset):
    def __init__(self, X, y):
        # Convert DataFrame to numpy array first, then to tensor
        self.X = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, 
                             dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Training Functions ---
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs, _ = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    return total_loss / len(loader), acc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    return total_loss / len(loader), acc
def display_classification_report(y_true, y_pred, class_names):
    """Affiche un classification report stylisé dans Streamlit"""
    
    # Générer le rapport
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    
    # Styler le DataFrame
    styled_df = df.style.format("{:.2f}", na_rep="-").applymap(
        lambda x: "color: green" if isinstance(x, (int, float)) and x > 0.8 else "color: orange",
        subset=pd.IndexSlice[class_names, ['precision', 'recall', 'f1-score']]
    ).applymap(
        lambda x: "font-weight: bold" if isinstance(x, (int, float)) and x > 0.9 else "",
        subset=pd.IndexSlice[class_names, ['precision', 'recall', 'f1-score']]
    ).set_properties(
        **{'background-color': '#f0f2f6'},
        subset=pd.IndexSlice[['accuracy', 'macro avg', 'weighted avg'], :]
    )
    
    # Afficher dans Streamlit
    st.subheader("📊 Classification Report")
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=(len(df) + 1) * 35 + 3
    )
    
    # Ajouter des explications
    with st.expander("ℹ️ Comment interpréter ce tableau"):
        st.markdown("""
        - **Precision** : Parmi les prédictions positives, combien sont vraiment positives ?
        - **Recall** : Parmi les vrais positifs, combien ont été correctement identifiés ?
        - **F1-score** : Moyenne harmonique de la precision et du recall
        - **Support** : Nombre d'échantillons pour chaque classe
        """)


# --- Streamlit App ---
def main():
    st.title("Logic Attention Network Trainer")
    
    # Sidebar for file upload and parameters
    with st.sidebar:
        st.header("Data & Parameters")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, sep=';', engine='python')  # Auto-detect separator
                df.columns = df.columns.str.replace('\ufeff', '')
                df.columns = df.columns.str.strip()
                st.success("File successfully loaded!")
                
                # Display basic info
                st.subheader("Data Preview")
                st.write(f"Shape: {df.shape}")
                st.dataframe(df.head())
                
                # Column selection
                target_col = st.selectbox("Select target column", df.columns)
                feature_cols = st.multiselect("Select feature columns", 
                                            [col for col in df.columns if col != target_col],
                                            default=[col for col in df.columns if col != target_col])
                
                # Model parameters
                st.subheader("Model Parameters")
                embed_dim = st.slider("Embedding dimension", 16, 256, 64, 16)
                centers = st.slider("Number of fuzzy centers", 2, 10, 3)
                epochs = st.slider("Number of epochs", 1, 50, 10)
                batch_size = st.slider("Batch size", 16, 256, 64, 16)
                learning_rate = st.slider("Learning rate", 0.0001, 0.01, 0.001, 0.0001)
                
                # Training button
                train_button = st.button("Train Model")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.info("Please upload a CSV file to begin")
            return
    
    # Main content area
    if uploaded_file is not None and train_button:
        st.subheader("Training Progress")
        
        # Prepare data
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Handle class imbalance
        ros = RandomOverSampler()
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        
        # Preprocessing
        dp = DataPreprocessor()
        X_train_processed, y_train_encoded = dp.fit_transform(X_resampled, y_resampled)
        X_test_processed, y_test_encoded = dp.transform(X_test, y_test)
        
        # Create datasets and loaders
        train_ds = CSVDataset(X_train_processed, y_train_encoded)
        test_ds = CSVDataset(X_test_processed, y_test_encoded)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X_train_processed.shape[1]
        num_classes = len(np.unique(y_train_encoded))
        
        model = LogicAttentionNet(
            input_dim=input_dim, 
            embed_dim=embed_dim, 
            num_classes=num_classes, 
            centers=centers
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # Training loop with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        best_acc = 0.0
        best_logloss = float('inf')
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
            )
            
            # Save best model
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_logloss):
                best_acc = test_acc
                best_logloss = test_loss
                torch.save(model.state_dict(), "best_logic_attention_model.pt")
                joblib.dump(dp, "preprocessor.joblib")
        
        # Training complete
        st.success("Training complete!")
        
        # Display metrics
        st.subheader("Evaluation Metrics")
        
        # Get predictions
        # Dans votre fonction main(), remplacez la partie évaluation par ceci:


    # [...] (votre code existant jusqu'à la fin de l'entraînement)
    
        # Évaluation du modèle
        model.eval()
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits, _ = model(X_batch)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
                all_targets.append(y_batch.cpu())

        # Convertir les listes en arrays numpy
        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs).numpy()
        all_targets = torch.cat(all_targets).numpy()

        # Maintenant vous pouvez utiliser ces variables pour le classification report
        try:
            report_dict = classification_report(all_targets, all_preds, 
                                            target_names=dp.y_encoder.classes_, 
                                            output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()

            # Afficher le rapport stylisé
            st.subheader("Classification Report")
            st.dataframe(
                report_df.style.format("{:.2f}"),
                height=(len(report_df) + 1) * 35 + 3,
                width=800
            )
            
        except Exception as e:
            st.error(f"Erreur lors de la génération du rapport: {str(e)}")    
        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(all_targets, all_preds)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=dp.y_encoder.classes_, 
                    yticklabels=dp.y_encoder.classes_,
                    ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
        # ROC curves (for binary or multiclass)
        if num_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(all_targets, all_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            st.pyplot(fig)
        else:
            # Multiclass ROC
            y_test_bin = label_binarize(all_targets, classes=np.arange(num_classes))
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            for i in range(num_classes):
                ax.plot(fpr[i], tpr[i], 
                        label=f'Class {dp.y_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')
            
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves (One-vs-Rest)')
            ax.legend()
            st.pyplot(fig)
        
        # Training history plots
        st.subheader("Training History")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['test_loss'], label='Test Loss')
        ax1.set_title('Loss over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['test_acc'], label='Test Accuracy')
        ax2.set_title('Accuracy over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        st.pyplot(fig)
        
        # Download trained model
        st.subheader("Model Download")
        with open("best_logic_attention_model.pt", "rb") as f:
            st.download_button(
                label="Download Trained Model",
                data=f,
                file_name="logic_attention_model.pt",
                mime="application/octet-stream"
            )
        
        with open("preprocessor.joblib", "rb") as f:
            st.download_button(
                label="Download Preprocessor",
                data=f,
                file_name="preprocessor.joblib",
                mime="application/octet-stream"
            )
        # Ajoutez cette section dans votre fonction main(), après la partie évaluation du modèle

    st.subheader("🔮 Prédiction sur une Nouvelle Instance")

    # Créer un formulaire pour la prédiction
    # Dans votre fonction main() de Streamlit
    with st.form("full_prediction_form"):
        st.write("### Remplissez toutes les caractéristiques pour une prédiction")
        
        # Organisé par sections
        with st.expander("Informations personnelles"):
            col1, col2 = st.columns(2)
            with col1:
                status = st.selectbox("Status", ["ACT", "INACT"])
                gender = st.selectbox("Gender", ["MALE", "FEMALE"])
                age = st.number_input("Âge", min_value=0, max_value=100, value=22)
            with col2:
                nationality = st.selectbox("Nationalité", ["DZ", "OTHER"])
                subscriber_type = st.selectbox("Type d'abonné", ["INDIVIDUAL", "COMPANY"])
        
        with st.expander("Informations d'abonnement"):
            col1, col2 = st.columns(2)
            with col1:
                subscription_type = st.selectbox("Type d'abonnement", ["PREPAID", "POSTPAID"])
                connection_type = st.selectbox("Type de connexion", ["2G", "3G", "4G", "5G"])
            with col2:
                tarrif_profile = st.text_input("Profil tarifaire", "DJEZZY LEGEND_4G")
                channel = st.selectbox("Canal", ["SNOC", "RETAIL", "ONLINE"])
        
        with st.expander("Documentation et validation"):
            doc_val_4g = st.selectbox("Validation document 4G", ["YES", "NO"])
            
            doc_scan = st.selectbox("Document scanné", ["YES", "NO"])
        
        with st.expander("Données financières"):
            revenue = st.number_input("Revenue 2 mois (DZD)", min_value=0, value=3000)
            arpu = st.number_input("ARPU 2 mois (DZD)", min_value=0, value=15000)
            segment = st.text_input("Segment client", "H VHV]>4000[")
        with st.expander("Validation OCR"):
            NIN_ok = st.selectbox("NIN_ok", ["1", "0"])
            DOB_ok = st.selectbox("DOB_ok", ["1", "0"])
            minor_ok = st.selectbox("minor_ok", ["1", "0"])
            cn_valid = st.selectbox("cn_valid", ["1", "0"])
            similarity_score = st.slider("Score de similarité", 0, 100, 95)
            similarity_score_bin = st.selectbox("similarity_score_bin", ["1", "0"])
            score_conf = st.selectbox("score_conf", ["1", "0"])
        submit_button = st.form_submit_button("Prédire")


    if submit_button:
        # Créer le dictionnaire de données (similaire à votre exemple)
        new_data = {
        'id_fact': [6460],
        'cust_id': [1222401523278],
        'nin': [1111122223333444],
        'MSISDN': [774155131],
        'status': ['ACT'],
        'status_date': ['2025-01-01 00:00:00'],
        'Birth_date': ['2003-04-26 00:00:00'],
        'age_sub': [22],
        'Gender': ['FEMALE'],
        'Nationality': ['DZ'],
        'Subscriber_type': ['INDIVIDUAL'],
        'id_type': ['NATIONAL_I'],
        'subscription_type': ['PREPAID'],
        'tarrif_profile': ['DJEZZY LEGEND_4G'],
        'ICC': ['3243785502934991111112222_774155131'],
        'type_sim': ['USIM'],
        'category': ['New'],
        'connection_type': ['4G'],
        'source': ['Source2'],
        'CHANNEL': ['SNOC'],
        'Activation_Date': ['2025-01-08 00:00:00'],
        'First_Call_Date': ['2025-01-08 00:40:00'],
        'Last_Call_Date': ['2025-05-09 16:00:00'],
        'ID_doc': ['79944268739945444331748811111111'],
        'Document_Validation_2G': ['NO'],
        'Document_Validation_3G': ['NO'],
        'Document_Validation_4G': ['YES'],
        'Document_stamped': ['YES'],
        'document_scaned_status': ['YES'],
        'Document_Validation_Date': ['2025-01-08 00:10:00'],
        'DOC_SCN_DT': ['2025-01-08 00:05:00'],
        'pdv_sk': [223],
        'PoS_ID': ['CH033428368069'],
        'DOC_VAL_USR': ['CH033428368069VAL1'],
        'DOK_SCN_USR': ['CH033428368069SCN1'],
        'BU': ['ALGER'],
        'localisation_sk': ['218'],
        'Postal_ID': [16050],
        'Province': ['ALGER'],
        'City': ['DRARIA'],
        'Street': ['BOUDJEMA TEMIME DRARIA N57'],
        'id_date': [20250108000500],
        'full_date': ['2025-01-08 00:05:00'],
        'year': [2025],
        'mois': [1],
        'lib_mois': ['Janvier'],
        'jours': [8],
        'lib_jour': ['Mercredi'],
        'object_id': ['SNOC-1222401523278-774155131'],
        'NIN_ok': [NIN_ok],
        'DOB_ok': [DOB_ok],
        'minor_ok': [minor_ok],
        'cn_valid': [cn_valid],
        'similarity_score': [similarity_score],
        'similarity_score_bin': [similarity_score_bin],
        'temps_moyen_appel': [54000.21],
        'temps_moyen_traitement': [5.0],
        'reactivite_client': [40.00],
        'Revenue_Last_2_Months': [3000.00],
        'Revenue_Last_12_Months': [18000.00],
        'ARPU_Last_2_Months': [15000.00],
        'segment_value': ['H VHV]>4000['],
        'client_haut_revenue': [1],
        'doc_scan_avant_activation': [1],
        'score_confiance': [score_conf]
    }
        
        # Convertir en DataFrame
        new_df = pd.DataFrame(new_data)
        
        try:
            # Charger le préprocesseur
            dp = joblib.load("preprocessor.joblib")
            
            # Prétraiter les données
            X_processed, _ = dp.transform(new_df)
            
            # Convertir en tenseur
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_tensor = torch.tensor(X_processed.values, dtype=torch.float32).to(device)
            
            # Charger le modèle
            model = LogicAttentionNet(input_dim=X_processed.shape[1], 
                                    num_classes=len(dp.y_encoder.classes_)).to(device)
            model.load_state_dict(torch.load("best_logic_attention_model.pt", map_location=device))
            model.eval()
            
            # Faire la prédiction
            with torch.no_grad():
                outputs, attn_weights = model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = outputs.argmax(dim=1).cpu().numpy()[0]
                pred_prob = probs[0][pred_class].item()
                pred_label = dp.inverse_transform_y([pred_class])[0]
            
            # Afficher les résultats
            st.success(f"Prédiction réussie !")
            
            # Carte de résultat
            st.markdown(f"""
            <div style="background-color:#e8f5e9; padding:20px; border-radius:10px; margin:10px 0;">
                <h3 style="color:#2e7d32;">Résultat de la prédiction</h3>
                <p><b>Classe prédite :</b> {pred_label}</p>
                <p><b>Probabilité :</b> {pred_prob:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Détails des probabilités par classe
            st.subheader("Probabilités par classe")
            prob_df = pd.DataFrame({
                "Classe": dp.y_encoder.classes_,
                "Probabilité": probs.cpu().numpy()[0]
            }).sort_values("Probabilité", ascending=False)
            
            st.dataframe(
                prob_df.style.format({"Probabilité": "{:.2%}"}),
                height=(len(prob_df) * 35 + 35),
                use_container_width=True
            )
            
            # Visualisation des probabilités
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x="Probabilité", y="Classe", data=prob_df, palette="Blues_d", ax=ax)
            ax.set_title("Distribution des Probabilités")
            ax.set_xlim(0, 1)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")
if __name__ == "__main__":
    main()
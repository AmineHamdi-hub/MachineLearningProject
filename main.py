import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_curve, auc

from scipy.cluster.hierarchy import dendrogram, linkage

# Fonction pour charger et nettoyer les données
@st.cache_data
def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)
    data = data.dropna()  # Suppression des valeurs manquantes
    return data

# Chargement des données
data_path = "data/Auto.csv"  # Assurez-vous que le fichier Auto.csv est dans le dossier data/
data = load_and_clean_data(data_path)

st.title("Machine Learning sur Auto.csv")

# ---- Upper bar (ligne de configuration) ----
# On crée une rangée de colonnes pour placer les contrôles en haut de la page
col1, col2, col3 = st.columns(3)

with col1:
    algo_type = st.selectbox("Type d'algorithme", ["Supervisé", "Non supervisé"])

with col2:
    if algo_type == "Supervisé":
        model_choice = st.selectbox("Modèle Supervisé", 
                                    ["Régression Linéaire", "Régression Logistique", "Random Forest"])
    else:
        model_choice = st.selectbox("Modèle Non Supervisé", 
                                    ["K-means", "Agglomerative Clustering", "DBSCAN", "PCA"])

with col3:
    show_data = st.checkbox("Afficher les données & résumé")

if show_data:
    st.subheader("Aperçu des données")
    st.write(data.head())
    st.write(data.describe())
    st.subheader("Matrice de corrélation")
    fig_corr, ax_corr = plt.subplots(figsize=(10,8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)
    plt.clf()

# ===================== Modèles Supervisés =====================
if algo_type == "Supervisé":
    st.header("Apprentissage Supervisé")
    st.subheader("Sélection des variables")
    st.write("Variables disponibles :", list(data.columns))
    # Pour cet exemple, on prend 'mpg' comme cible
    features = st.multiselect("Sélectionnez les features", 
                              options=data.columns.drop("mpg"), 
                              default=["horsepower", "weight", "acceleration", "cylinders"])
    target = st.selectbox("Sélectionnez la variable cible", options=["mpg"])
    
    if len(features) > 0:
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- Régression Linéaire ---
        if model_choice == "Régression Linéaire":
            st.subheader("Régression Linéaire")
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"MSE: {mse:.2f}")
            st.write(f"R²: {r2:.2f}")
            
            # Courbe de prédiction : Réel vs Prédiction
            fig1, ax1 = plt.subplots()
            ax1.scatter(y_test, y_pred, alpha=0.7, color="blue")
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
            ax1.set_xlabel("Valeurs réelles")
            ax1.set_ylabel("Prédictions")
            ax1.set_title("Régression Linéaire : Réel vs Prédiction")
            st.pyplot(fig1)
            
            # Analyse des résidus
            residuals = y_test - y_pred
            fig2, ax2 = plt.subplots()
            ax2.scatter(y_pred, residuals, alpha=0.7, color="red")
            ax2.axhline(0, color="black", lw=2)
            ax2.set_xlabel("Prédictions")
            ax2.set_ylabel("Résidus")
            ax2.set_title("Analyse des Résidus")
            st.pyplot(fig2)
        
        # --- Régression Logistique (classification) ---
        elif model_choice == "Régression Logistique":
            st.subheader("Régression Logistique")
            # Transformation en classification binaire à partir de 'mpg'
            median_mpg = y.median()
            y_class = (y > median_mpg).astype(int)
            st.write("Transformation en variable binaire (0: faible, 1: élevé) basée sur la médiane de mpg.")
            X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {acc:.2f}")
            cm = confusion_matrix(y_test, y_pred)
            st.write("Matrice de confusion :")
            st.write(cm)
            # Courbe ROC
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax_roc.plot([0,1],[0,1], "k--")
                ax_roc.set_xlabel("Taux de faux positifs")
                ax_roc.set_ylabel("Taux de vrais positifs")
                ax_roc.set_title("Courbe ROC")
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
        
        # --- Random Forest ---
        elif model_choice == "Random Forest":
            st.subheader("Random Forest")
            n_estimators = st.slider("Nombre d'arbres", 10, 300, 100)
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"MSE: {mse:.2f}")
            st.write(f"R²: {r2:.2f}")
            
            # Visualisation : Réel vs Prédiction
            fig_rf, ax_rf = plt.subplots()
            ax_rf.scatter(y_test, y_pred, alpha=0.7, color="green")
            ax_rf.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
            ax_rf.set_xlabel("Valeurs réelles")
            ax_rf.set_ylabel("Prédictions")
            ax_rf.set_title("Random Forest : Réel vs Prédiction")
            st.pyplot(fig_rf)
            
            # Importance des caractéristiques
            importances = model.feature_importances_
            fig_imp, ax_imp = plt.subplots()
            ax_imp.bar(features, importances, color="orange")
            ax_imp.set_xlabel("Features")
            ax_imp.set_ylabel("Importance")
            ax_imp.set_title("Importance des Caractéristiques")
            st.pyplot(fig_imp)
        
        # Option pour télécharger les résultats
        if st.button("Télécharger les prédictions"):
            results = X_test.copy()
            results["Prédictions"] = y_pred
            results["Valeurs réelles"] = y_test.values
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger le CSV", data=csv, file_name="predictions.csv", mime="text/csv")
    else:
        st.warning("Veuillez sélectionner au moins une feature.")

# ===================== Modèles Non Supervisés =====================
else:
    st.header("Apprentissage Non Supervisé")
    features_cluster = st.multiselect("Sélectionnez les variables pour le clustering", 
                                      options=data.select_dtypes(include=[np.number]).columns.tolist(), 
                                      default=["horsepower", "weight"])
    if len(features_cluster) >= 2:
        X_cluster = data[features_cluster]
        
        # --- K-means ---
        if model_choice == "K-means":
            k = st.slider("Nombre de clusters (k)", 2, 10, 3)
            if st.checkbox("Afficher la courbe du coude"):
                inertias = []
                for k_val in range(2, 11):
                    km_temp = KMeans(n_clusters=k_val, random_state=42)
                    km_temp.fit(X_cluster)
                    inertias.append(km_temp.inertia_)
                fig_elbow, ax_elbow = plt.subplots()
                ax_elbow.plot(range(2, 11), inertias, marker="o")
                ax_elbow.set_xlabel("Nombre de clusters")
                ax_elbow.set_ylabel("Inertie")
                ax_elbow.set_title("Méthode du coude")
                st.pyplot(fig_elbow)
            model = KMeans(n_clusters=k, random_state=42)
            clusters = model.fit_predict(X_cluster)
            data["Cluster"] = clusters
            st.write(data[[*features_cluster, "Cluster"]].head())
            fig_km, ax_km = plt.subplots()
            ax_km.scatter(X_cluster.iloc[:, 0], X_cluster.iloc[:, 1], c=clusters, cmap="viridis", alpha=0.7)
            ax_km.set_xlabel(features_cluster[0])
            ax_km.set_ylabel(features_cluster[1])
            ax_km.set_title("K-means Clustering")
            st.pyplot(fig_km)
        
        # --- Agglomerative Clustering ---
        elif model_choice == "Agglomerative Clustering":
            n_clusters = st.slider("Nombre de clusters", 2, 10, 3)
            model = AgglomerativeClustering(n_clusters=n_clusters)
            clusters = model.fit_predict(X_cluster)
            data["Cluster"] = clusters
            st.write(data[[*features_cluster, "Cluster"]].head())
            fig_agg, ax_agg = plt.subplots()
            ax_agg.scatter(X_cluster.iloc[:, 0], X_cluster.iloc[:, 1], c=clusters, cmap="viridis", alpha=0.7)
            ax_agg.set_xlabel(features_cluster[0])
            ax_agg.set_ylabel(features_cluster[1])
            ax_agg.set_title("Agglomerative Clustering")
            st.pyplot(fig_agg)
            if st.checkbox("Afficher dendrogramme"):
                sample = X_cluster.sample(n=min(50, len(X_cluster)), random_state=42)
                Z = linkage(sample, method="ward")
                fig_dendro, ax_dendro = plt.subplots(figsize=(10, 5))
                dendrogram(Z, ax=ax_dendro)
                ax_dendro.set_title("Dendrogramme")
                st.pyplot(fig_dendro)
        
        # --- DBSCAN ---
        elif model_choice == "DBSCAN":
            eps = st.slider("Paramètre eps", 0.1, 10.0, 2.0)
            min_samples = st.slider("min_samples", 2, 20, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = model.fit_predict(X_cluster)
            data["Cluster"] = clusters
            st.write(data[[*features_cluster, "Cluster"]].head())
            fig_db, ax_db = plt.subplots()
            ax_db.scatter(X_cluster.iloc[:, 0], X_cluster.iloc[:, 1], c=clusters, cmap="viridis", alpha=0.7)
            ax_db.set_xlabel(features_cluster[0])
            ax_db.set_ylabel(features_cluster[1])
            ax_db.set_title("DBSCAN Clustering")
            st.pyplot(fig_db)
        
        # --- PCA ---
        elif model_choice == "PCA":
            n_components = st.slider("Nombre de composantes", 2, min(len(features_cluster), 5), 2)
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(X_cluster)
            st.write("Variance expliquée :", pca.explained_variance_ratio_)
            fig_scree, ax_scree = plt.subplots()
            ax_scree.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, color="purple")
            ax_scree.set_xlabel("Composante")
            ax_scree.set_ylabel("Variance expliquée")
            ax_scree.set_title("Scree Plot")
            st.pyplot(fig_scree)
            if n_components == 2:
                fig_pca, ax_pca = plt.subplots()
                ax_pca.scatter(components[:, 0], components[:, 1], alpha=0.7)
                ax_pca.set_xlabel("Composante 1")
                ax_pca.set_ylabel("Composante 2")
                ax_pca.set_title("PCA (2 composantes)")
                st.pyplot(fig_pca)
            else:
                st.write("Visualisation PCA pour plus de 2 composantes non supportée.")
        
        # Option pour télécharger les résultats du clustering
        if st.button("Télécharger les résultats du clustering"):
            csv = data[[*features_cluster, "Cluster"]].to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger le CSV", data=csv, file_name="clustering_results.csv", mime="text/csv")
    else:
        st.warning("Veuillez sélectionner au moins deux variables pour le clustering.")

st.info("Projet de Machine Learning avec fonctionnalités étendues")

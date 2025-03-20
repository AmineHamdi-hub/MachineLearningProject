import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage

import joblib
import os

# --- Chargement et prétraitement des données ---


@st.cache_data
def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)
    # Au lieu de supprimer les valeurs manquantes, on les impute (pour les numériques on utilise la moyenne)
    num_cols = data.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        data[col].fillna(data[col].mean(), inplace=True)
    return data


data_path = (
    "data/Auto.csv"  # Vérifiez que le fichier Auto.csv se trouve dans le dossier data/
)
data = load_and_clean_data(data_path)

st.title("Machine Learning sur Auto.csv")

# --- Barre latérale pour la configuration ---
st.sidebar.header("Configuration de l'analyse")
algo_type = st.sidebar.selectbox("Type d'algorithme", ["Supervisé", "Non supervisé"])
if algo_type == "Supervisé":
    model_choice = st.sidebar.selectbox(
        "Modèle Supervisé",
        ["Régression Linéaire", "Régression Logistique", "Random Forest"],
    )
else:
    model_choice = st.sidebar.selectbox(
        "Modèle Non Supervisé", ["K-means", "Agglomerative Clustering", "DBSCAN"]
    )
show_data = st.sidebar.checkbox("Afficher les données & résumé", value=True)

# --- Onglets pour organiser l'application ---
tabs = st.tabs(["Exploration", "Modélisation", "Prédiction"])

with tabs[0]:
    st.header("Analyse exploratoire")
    if show_data:
        st.subheader("Aperçu des données")
        st.write(data.head())
        st.write(data.describe())

        st.subheader("Matrice de corrélation")
        numeric_data = data.select_dtypes(include=[np.number])
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr
        )
        st.pyplot(fig_corr)
        plt.clf()

    if st.checkbox("Afficher l'analyse exploratoire via des graphiques", key="exp"):
        st.subheader("Histogrammes des variables numériques")
        for col in numeric_data.columns:
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(numeric_data[col], kde=True, ax=ax_hist, color="skyblue")
            ax_hist.set_title(f"Distribution de {col}")
            st.pyplot(fig_hist)
            plt.clf()

        if st.checkbox("Afficher le pairplot des variables numériques", key="pair"):
            st.subheader("Pairplot")
            pairgrid = sns.pairplot(numeric_data)
            st.pyplot(pairgrid.fig)
            plt.clf()

        st.subheader("Boxplots des variables numériques")
        for col in numeric_data.columns:
            fig_box, ax_box = plt.subplots()
            sns.boxplot(y=numeric_data[col], ax=ax_box, color="lightgreen")
            ax_box.set_title(f"Boxplot de {col}")
            st.pyplot(fig_box)
            plt.clf()

# --- Partie Supervisée ou Non Supervisée ---
if algo_type == "Supervisé":
    with tabs[1]:
        st.header("Apprentissage Supervisé")
        st.subheader("Sélection des variables")
        st.write("Variables disponibles :", list(data.columns))
        # On prend 'mpg' comme cible
        features = st.multiselect(
            "Sélectionnez les features",
            options=data.columns.drop("mpg"),
            default=["horsepower", "weight", "acceleration", "cylinders"],
            help="Choisissez les variables explicatives",
        )
        target = st.selectbox(
            "Sélectionnez la variable cible",
            options=["mpg"],
            help="La variable à prédire",
        )

        if len(features) > 0:
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Création d'un pipeline commun pour les régressions (imputation + scaling)
            pipeline_steps = [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]

            # --- Régression Linéaire ---
            if model_choice == "Régression Linéaire":
                st.subheader("Régression Linéaire")
                pipe_lr = Pipeline(
                    steps=pipeline_steps + [("model", LinearRegression())]
                )
                pipe_lr.fit(X_train, y_train)
                y_pred = pipe_lr.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"MSE: {mse:.2f}")
                st.write(f"R²: {r2:.2f}")

                if len(features) == 1:
                    feature = features[0]
                    st.subheader("Illustration de la régression linéaire")
                    X_range = np.linspace(
                        data[feature].min(), data[feature].max(), 100
                    ).reshape(-1, 1)
                    y_range_pred = pipe_lr.predict(np.array(X_range))
                    fig_line, ax_line = plt.subplots(figsize=(10, 6))
                    ax_line.scatter(
                        data[feature], data[target], alpha=0.5, label="Données"
                    )
                    ax_line.plot(
                        X_range,
                        y_range_pred,
                        color="red",
                        linewidth=2,
                        label="Ligne de régression",
                    )
                    ax_line.set_xlabel(feature)
                    ax_line.set_ylabel(target)
                    ax_line.set_title("Régression Linéaire")
                    ax_line.legend()
                    st.pyplot(fig_line)
                else:
                    fig1, ax1 = plt.subplots()
                    ax1.scatter(y_test, y_pred, alpha=0.7, color="blue")
                    ax1.plot(
                        [y_test.min(), y_test.max()],
                        [y_test.min(), y_test.max()],
                        "k--",
                        lw=2,
                    )
                    ax1.set_xlabel("Valeurs réelles")
                    ax1.set_ylabel("Prédictions")
                    ax1.set_title("Réel vs Prédiction")
                    st.pyplot(fig1)

                    residuals = y_test - y_pred
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(y_pred, residuals, alpha=0.7, color="red")
                    ax2.axhline(0, color="black", lw=2)
                    ax2.set_xlabel("Prédictions")
                    ax2.set_ylabel("Résidus")
                    ax2.set_title("Analyse des Résidus")
                    st.pyplot(fig2)

                # Enregistrement du modèle
                joblib.dump(pipe_lr, "model_linear_regression.pkl")
                st.info(
                    "Modèle de régression linéaire enregistré sous 'model_linear_regression.pkl'"
                )

            # --- Régression Logistique ---
            elif model_choice == "Régression Logistique":
                st.subheader("Régression Logistique")
                # Transformation en classification binaire
                median_mpg = y.median()
                y_class = (y > median_mpg).astype(int)
                st.write(
                    "Transformation en variable binaire (0: faible, 1: élevé) basée sur la médiane de mpg."
                )
                X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                    X, y_class, test_size=0.2, random_state=42
                )
                pipe_log = Pipeline(
                    steps=pipeline_steps
                    + [("model", LogisticRegression(max_iter=1000))]
                )
                pipe_log.fit(X_train_c, y_train_c)
                y_pred = pipe_log.predict(X_test_c)
                acc = accuracy_score(y_test_c, y_pred)
                st.write(f"Accuracy: {acc:.2f}")
                cm = confusion_matrix(y_test_c, y_pred)
                st.write("Matrice de confusion :")
                st.write(cm)
                if hasattr(pipe_log.named_steps["model"], "predict_proba"):
                    y_score = pipe_log.predict_proba(X_test_c)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test_c, y_score)
                    roc_auc = auc(fpr, tpr)
                    fig_roc, ax_roc = plt.subplots()
                    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    ax_roc.plot([0, 1], [0, 1], "k--")
                    ax_roc.set_xlabel("Taux de faux positifs")
                    ax_roc.set_ylabel("Taux de vrais positifs")
                    ax_roc.set_title("Courbe ROC")
                    ax_roc.legend(loc="lower right")
                    st.pyplot(fig_roc)

                joblib.dump(pipe_log, "model_logistic_regression.pkl")
                st.info(
                    "Modèle de régression logistique enregistré sous 'model_logistic_regression.pkl'"
                )

            # --- Random Forest avec GridSearchCV ---
            elif model_choice == "Random Forest":
                st.subheader("Random Forest")
                n_estimators = st.sidebar.slider("Nombre d'arbres", 10, 300, 100)
                # Pipeline pour Random Forest
                pipe_rf = Pipeline(
                    steps=pipeline_steps
                    + [("model", RandomForestRegressor(random_state=42))]
                )
                # Paramètres pour GridSearchCV
                param_grid = {
                    "model__n_estimators": [
                        n_estimators - 20,
                        n_estimators,
                        n_estimators + 20,
                    ],
                    "model__max_depth": [None, 5, 10],
                }
                grid_search = GridSearchCV(pipe_rf, param_grid, cv=5, scoring="r2")
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write("Meilleurs hyperparamètres :", grid_search.best_params_)
                st.write(f"MSE: {mse:.2f}")
                st.write(f"R²: {r2:.2f}")

                fig_rf, ax_rf = plt.subplots()
                ax_rf.scatter(y_test, y_pred, alpha=0.7, color="green")
                ax_rf.plot(
                    [y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    "k--",
                    lw=2,
                )
                ax_rf.set_xlabel("Valeurs réelles")
                ax_rf.set_ylabel("Prédictions")
                ax_rf.set_title("Réel vs Prédiction")
                st.pyplot(fig_rf)

                importances = best_model.named_steps["model"].feature_importances_
                fig_imp, ax_imp = plt.subplots()
                ax_imp.bar(features, importances, color="orange")
                ax_imp.set_xlabel("Features")
                ax_imp.set_ylabel("Importance")
                ax_imp.set_title("Importance des Caractéristiques")
                st.pyplot(fig_imp)

                joblib.dump(best_model, "model_random_forest.pkl")
                st.info(
                    "Modèle Random Forest enregistré sous 'model_random_forest.pkl'"
                )

            # Option pour télécharger les résultats
            if st.button("Télécharger les prédictions"):
                results = X_test.copy()
                results["Prédictions"] = y_pred
                results["Valeurs réelles"] = y_test.values
                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Télécharger le CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
        else:
            st.warning("Veuillez sélectionner au moins une feature.")

    # --- Formulaires de prédiction supervisée ---
    with tabs[2]:
        st.header("Formulaire de prédiction")
        if algo_type == "Supervisé":
            st.write(
                "Utilisez le formulaire ci-dessous pour obtenir une prédiction avec le modèle entraîné."
            )
            # Choisir le modèle en fonction de la sélection précédente
            pred_model = model_choice
            features_pred = st.multiselect(
                "Sélectionnez les features à utiliser pour la prédiction",
                options=data.columns.drop("mpg"),
                default=["horsepower", "weight", "acceleration", "cylinders"],
            )
            if len(features_pred) > 0:
                with st.form("form_prediction"):
                    input_data = {}
                    for feat in features_pred:
                        input_data[feat] = st.number_input(
                            f"Entrez la valeur pour {feat} :",
                            value=float(data[feat].mean()),
                            help=f"Moyenne de {feat} : {data[feat].mean():.2f}",
                        )
                    submit_pred = st.form_submit_button("Prédire")
                    if submit_pred:
                        input_df = pd.DataFrame([input_data])
                        # Charger le modèle sauvegardé en fonction du choix
                        if pred_model == "Régression Linéaire":
                            model_loaded = joblib.load("model_linear_regression.pkl")
                            prediction = model_loaded.predict(input_df)[0]
                            st.success(f"La prédiction de mpg est : {prediction:.2f}")
                        elif pred_model == "Régression Logistique":
                            model_loaded = joblib.load("model_logistic_regression.pkl")
                            proba = model_loaded.predict_proba(input_df)[0, 1]
                            st.success(
                                f"La probabilité de la classe '1' est : {proba:.2f}"
                            )
                        elif pred_model == "Random Forest":
                            model_loaded = joblib.load("model_random_forest.pkl")
                            prediction = model_loaded.predict(input_df)[0]
                            st.success(f"La prédiction de mpg est : {prediction:.2f}")
        else:
            st.info(
                "Les formulaires de prédiction sont disponibles pour l'apprentissage supervisé uniquement."
            )

# --- Partie Non Supervisée ---
if algo_type == "Non supervisé":
    with tabs[1]:
        st.header("Apprentissage Non Supervisé")
        features_cluster = st.multiselect(
            "Sélectionnez les variables pour le clustering",
            options=data.select_dtypes(include=[np.number]).columns.tolist(),
            default=["horsepower", "weight"],
            help="Au moins deux variables sont nécessaires",
        )
        if len(features_cluster) >= 2:
            X_cluster = data[features_cluster]

            # --- K-means ---
            if model_choice == "K-means":
                k = st.sidebar.slider("Nombre de clusters (k)", 2, 10, 3)
                if st.checkbox("Afficher la courbe du coude"):
                    inertias = []
                    silhouette_scores = []
                    for k_val in range(2, 11):
                        km_temp = KMeans(n_clusters=k_val, random_state=42)
                        clusters_temp = km_temp.fit_predict(X_cluster)
                        inertias.append(km_temp.inertia_)
                        silhouette_avg = silhouette_score(X_cluster, clusters_temp)
                        silhouette_scores.append(silhouette_avg)

                    fig_elbow, ax_elbow = plt.subplots(1, 2, figsize=(12, 6))
                    ax_elbow[0].plot(range(2, 11), inertias, marker="o")
                    ax_elbow[0].set_xlabel("Nombre de clusters")
                    ax_elbow[0].set_ylabel("Inertie")
                    ax_elbow[0].set_title("Méthode du coude")

                    ax_elbow[1].plot(
                        range(2, 11), silhouette_scores, marker="o", color="purple"
                    )
                    ax_elbow[1].set_xlabel("Nombre de clusters")
                    ax_elbow[1].set_ylabel("Silhouette Score")
                    ax_elbow[1].set_title("Silhouette Score")

                    st.pyplot(fig_elbow)

                model = KMeans(n_clusters=k, random_state=42)
                clusters = model.fit_predict(X_cluster)
                inertia = model.inertia_
                silhouette_avg = silhouette_score(X_cluster, clusters)

                st.write(f"Inertie : {inertia:.2f}")
                st.write(f"Silhouette Score : {silhouette_avg:.2f}")

                data["Cluster"] = clusters
                st.write(data[[*features_cluster, "Cluster"]].head())

                fig_km, ax_km = plt.subplots()
                ax_km.scatter(
                    X_cluster.iloc[:, 0],
                    X_cluster.iloc[:, 1],
                    c=clusters,
                    cmap="viridis",
                    alpha=0.7,
                )
                ax_km.set_xlabel(features_cluster[0])
                ax_km.set_ylabel(features_cluster[1])
                ax_km.set_title("K-means Clustering")
                st.pyplot(fig_km)

            # --- Agglomerative Clustering ---
            elif model_choice == "Agglomerative Clustering":
                n_clusters = st.sidebar.slider("Nombre de clusters", 2, 10, 3)
                model = AgglomerativeClustering(n_clusters=n_clusters)
                clusters = model.fit_predict(X_cluster)

                if n_clusters > 1:
                    silhouette_avg = silhouette_score(X_cluster, clusters)
                    st.write(f"Silhouette Score : {silhouette_avg:.2f}")

                data["Cluster"] = clusters
                st.write(data[[*features_cluster, "Cluster"]].head())

                fig_agg, ax_agg = plt.subplots()
                ax_agg.scatter(
                    X_cluster.iloc[:, 0],
                    X_cluster.iloc[:, 1],
                    c=clusters,
                    cmap="viridis",
                    alpha=0.7,
                )
                ax_agg.set_xlabel(features_cluster[0])
                ax_agg.set_ylabel(features_cluster[1])
                ax_agg.set_title("Agglomerative Clustering")
                st.pyplot(fig_agg)

            # --- DBSCAN ---
            elif model_choice == "DBSCAN":
                eps = st.sidebar.slider("Paramètre eps", 0.1, 10.0, 2.0)
                min_samples = st.sidebar.slider("min_samples", 2, 20, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = model.fit_predict(X_cluster)

                if len(set(clusters)) > 1:
                    silhouette_avg = silhouette_score(X_cluster, clusters)
                    st.write(f"Silhouette Score : {silhouette_avg:.2f}")

                data["Cluster"] = clusters
                st.write(data[[*features_cluster, "Cluster"]].head())

                fig_db, ax_db = plt.subplots()
                ax_db.scatter(
                    X_cluster.iloc[:, 0],
                    X_cluster.iloc[:, 1],
                    c=clusters,
                    cmap="viridis",
                    alpha=0.7,
                )
                ax_db.set_xlabel(features_cluster[0])
                ax_db.set_ylabel(features_cluster[1])
                ax_db.set_title("DBSCAN Clustering")
                st.pyplot(fig_db)

            if st.button("Télécharger les résultats du clustering"):
                csv = (
                    data[[*features_cluster, "Cluster"]]
                    .to_csv(index=False)
                    .encode("utf-8")
                )
                st.download_button(
                    "Télécharger le CSV",
                    data=csv,
                    file_name="clustering_results.csv",
                    mime="text/csv",
                )
        else:
            st.warning(
                "Veuillez sélectionner au moins deux variables pour le clustering."
            )

st.info("Projet de Machine Learning avec fonctionnalités étendues")

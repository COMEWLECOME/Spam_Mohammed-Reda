import cudf
import cuml
from cuml.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Générer des données de classification aléatoires
X, y = make_classification(n_samples=100000, n_features=20, random_state=42)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir les données en objets cuDF
X_train = cudf.DataFrame(X_train)
y_train = cudf.Series(y_train)

# Créer un modèle de régression logistique
lr = LogisticRegression()

# Entraîner le modèle sur GPU
lr.fit(X_train, y_train)

# Prédire les étiquettes pour l'ensemble de test
y_pred = lr.predict(X_test)

# Calculer la précision du modèle
accuracy = cuml.metrics.accuracy_score(y_test, y_pred)

# Afficher la précision
print("Précision: ", accuracy)

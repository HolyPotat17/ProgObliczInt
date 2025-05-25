import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(csv_file):
    """Wczytywanie danych z pliku CSV i przygotowywanie ich do klasyfikacji"""
    data = pd.read_csv(csv_file)

    # Wydzielenie cech i etykiet
    X = data.drop(['filename', 'class'], axis=1).values
    y = data['class'].values

    # Kodowanie etykiet na liczby
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Normalizacja cech
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, label_encoder


def train_and_evaluate(X, y, test_size=0.3, random_state=42):
    """Trenowanie i ocenianie klasyfikatorów"""
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Lista klasyfikatorów do porównania
    classifiers = [
        ('KNN (k=3)', KNeighborsClassifier(n_neighbors=3)),
        ('KNN (k=5)', KNeighborsClassifier(n_neighbors=5)),
        ('SVM (linear)', SVC(kernel='linear')),
        ('SVM (RBF)', SVC(kernel='rbf'))
    ]

    results = {}
    for name, clf in classifiers:
        # Trenowanie modelu
        clf.fit(X_train, y_train)

        # Predykcja na zbiorze testowym
        y_pred = clf.predict(X_test)

        # Obliczenie dokładności
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        # Wyświetlenie raportu klasyfikacji
        print(f"\n{name} - Dokładność: {accuracy:.4f}")
        print("Raport klasyfikacji:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    return results


if __name__ == "__main__":
    # Wczytywanie danych z Lab3-3
    csv_file = "texture_features.csv"
    X, y, label_encoder = load_and_prepare_data(csv_file)

    print(f"\nLiczba próbek: {X.shape[0]}")
    print(f"Liczba cech: {X.shape[1]}")
    print(f"Klasy: {label_encoder.classes_}")

    # Trenowanie i ocenianie klasyfikatorów
    results = train_and_evaluate(X, y)

    # Znajdywanie najlepszego klasyfikatora
    best_name = max(results, key=results.get)
    print(f"\nNajlepszy klasyfikator: {best_name} z dokładnością {results[best_name]:.4f}")
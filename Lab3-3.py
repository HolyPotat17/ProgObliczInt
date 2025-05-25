import os
import numpy as np
from skimage import io, color
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm



def reduce_gray_levels(image, levels=64):
    """Zmniejszanie głębi bitowej obrazu do określonej liczby poziomów"""
    max_val = image.max()
    return (image / (max_val / (levels - 1))).astype(np.uint8)


def extract_glcm_features(image_path, distances=[1, 3, 5], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    """
    Wyodrębnianie cechy tekstury na podstawie GLCM
    Zwracanie słownika z cechami dla każdej kombinacji odległości i kąta
    """
    # Wczytanie i konwersja do skali szarości
    image = io.imread(image_path)
    gray_image = color.rgb2gray(image)
    gray_image = img_as_ubyte(gray_image)

    # Redukcja głębi bitowej do 5 bitów (64 poziomów)
    gray_image = reduce_gray_levels(gray_image, levels=64)

    # Obliczenie GLCM
    glcm = graycomatrix(gray_image,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)

    # Wybrane właściwości GLCM do obliczenia
    properties = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']

    # Wyodrębnienie cech
    features = {}
    for prop in properties:
        features[prop] = graycoprops(glcm, prop).ravel()

    return features


def process_texture_samples(input_dir, output_file='texture_features.csv'):
    """
    Przetwarzanie wszystkich próbek tekstur w folderze i zapisywanie do pliku CSV
    """
    # Znajdywanie wszystkie pliki obrazów
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("Nie znaleziono plików obrazów w podanym folderze!")
        return

    # Przygotowywanie nagłówek kolumn
    distances = [1, 3, 5]
    angles = [0, 45, 90, 135]
    properties = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']

    # Generowanie nagłówek kolumn
    headers = ['filename', 'class']
    for d in distances:
        for a in angles:
            for prop in properties:
                headers.append(f"{prop}_d{d}_a{a}")

    # Przetwarzanie obrazów
    features_data = []
    for image_path in tqdm(image_files, desc="Przetwarzanie tekstur"):
        # Pobieranie nazw klas z nazwy folderu
        class_name = os.path.basename(os.path.dirname(image_path))

        try:
            features = extract_glcm_features(image_path)

            # Przygotowywanie wiersz danych
            row = [os.path.basename(image_path), class_name]
            for d in range(len(distances)):
                for a in range(len(angles)):
                    for prop in properties:
                        idx = d * len(angles) + a
                        row.append(features[prop][idx])

            features_data.append(row)
        except Exception as e:
            print(f"Błąd przetwarzania {image_path}: {str(e)}")

    # Zapisywanie do pliku CSV
    import csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(features_data)

    print(f"\nZapisano cechy do pliku: {output_file}")


if __name__ == "__main__":
    # Konfiguracja
    input_directory = "Tekstury_Próbki"  # Folder z próbkami tekstur
    output_csv = "texture_features.csv"  # Plik wynikowy

    # Przetwarzanie
    process_texture_samples(input_directory, output_csv)
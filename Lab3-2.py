import os
import cv2
from PIL import Image


def extract_texture_samples(input_base_dir, output_base_dir, sample_size=(128, 128)):
    """
    Wycinanie próbek tekstur z obrazów i zapisywanie ich do podkatalogów.
    """
    # Sprawdzanie czy folder wejściowy istnieje
    if not os.path.exists(input_base_dir):
        raise FileNotFoundError(f"Folder wejściowy {input_base_dir} nie istnieje!")


    # Tworzenie folderu głównego jeśli nie istnieje
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Folder wyjściowy: {os.path.abspath(output_base_dir)}")

    for texture_dir in os.listdir(input_base_dir):
        input_texture_path = os.path.join(input_base_dir, texture_dir)

        output_texture_path = os.path.join(output_base_dir, texture_dir)
        os.makedirs(output_texture_path, exist_ok=True)
        print(f"\nPrzetwarzanie: {texture_dir}")

        for file in os.listdir(input_texture_path):

            file_path = os.path.join(input_texture_path, file)
            img = cv2.imread(file_path)

            height, width = img.shape[:2]
            sample_w, sample_h = sample_size

            num_x = width // sample_w
            num_y = height // sample_h
            sample_count = 0

            for y in range(num_y):
                for x in range(num_x):
                    sample = img[y * sample_h:(y + 1) * sample_h, x * sample_w:(x + 1) * sample_w]

                    if sample.shape[:2] != (sample_h, sample_w):
                        continue

                    sample_path = os.path.join(output_texture_path,
                                               f"{os.path.splitext(file)[0]}_{y}_{x}.jpg")

                    Image.fromarray(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)).save(sample_path)
                    print(f"Zapisano przez PIL: {sample_path}")
                    sample_count += 1

            print(f"Zapisano {sample_count} próbek z {file}")


if __name__ == "__main__":

    extract_texture_samples("Tekstury", "Tekstury_Próbki")
    print("\nPrzetwarzanie zakończone!")
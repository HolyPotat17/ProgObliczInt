import open3d as o3d
import numpy as np
import os


# Funkcja do wczytania chmury punktów z pliku .txt
def load_point_cloud(file_path):
    # Załadowywanie chmur punktów z pliku .txt
    data = np.loadtxt(file_path)
    points = data[:, :3]
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


# Funkcja do iteracyjnego dopasowywania płaszczyzn
def iterative_plane_fitting(pcd, iterations=6, distance_threshold=0.01):
    remaining_pcd = pcd
    extracted_planes = []

    for i in range(iterations):
        # Dopasowanie płaszczyzny za pomocą RANSAC
        print(f"Iteracja {i + 1}")
        plane_model, inlier_indices = remaining_pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3,
                                                                  num_iterations=1000)

        # Wyodrębnianie punktów należących do płaszczyzny
        inlier_cloud = remaining_pcd.select_by_index(inlier_indices)
        extracted_planes.append(inlier_cloud)

        # Usuwanie punktów płaszczyznowych z chmury wejściowej
        remaining_pcd = remaining_pcd.select_by_index(inlier_indices, invert=True)

        # Wizualizacja
        o3d.visualization.draw_geometries([remaining_pcd, inlier_cloud], window_name=f"Iteracja {i + 1}")

    return extracted_planes, remaining_pcd


# Funkcja do wyświetlenia wyników w CloudCompare
def save_planes_for_cloudcompare(extracted_planes, output_directory):
    # Sprawdzanie, czy folder istnieje, jeśli nie to jest tworzony
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, plane in enumerate(extracted_planes):
        filename = f"{output_directory}/plane_{i + 1}.ply"

        # Sprawdzanie czy ścieżka do folderu jest poprawna
        try:
            o3d.io.write_point_cloud(filename, plane)
            print(f"Zapisano płaszczyznę {i + 1} jako {filename}")
        except Exception as e:
            print(f"[Błąd] Nie udało się zapisać pliku {filename}: {e}")



def main():
    # Ścieżka do pliku z chmurą punktów
    file_path = "conferenceRoom_1.txt"
    output_directory = "output_planes"  # Katalog, w którym zapisywane będą wyniki

    # Załadowywanie chmur punktów
    pcd = load_point_cloud(file_path)

    # Iteracyjne dopasowanie płaszczyzn
    extracted_planes, remaining_pcd = iterative_plane_fitting(pcd, iterations=6, distance_threshold=0.01)

    # Zapisywanie płaszczyzny jako pliki PLY, które można potem otworzyć w CloudCompare
    save_planes_for_cloudcompare(extracted_planes, output_directory)


if __name__ == "__main__":
    main()

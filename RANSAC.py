import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv

def load_xyz_file(filename):
    points = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if len(row) >= 3:
                points.append([float(row[0]), float(row[1]), float(row[2])])
    return np.array(points)

def fit_plane_ransac(points, threshold=0.05, iterations=1000):
    best_eq = None
    best_inliers = []

    for _ in range(iterations):
        sample = points[np.random.choice(points.shape[0], 3, replace=False)]
        p1, p2, p3 = sample

        # Oblicz wektor normalny
        normal = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(normal) == 0:
            continue
        normal = normal / np.linalg.norm(normal)

        # Równanie płaszczyzny: ax + by + cz + d = 0
        d = -np.dot(normal, p1)

        # Odległości wszystkich punktów od płaszczyzny
        distances = np.abs(np.dot(points, normal) + d)
        inliers = points[distances < threshold]

        if len(inliers) > len(best_inliers):
            best_eq = (normal, d)
            best_inliers = inliers

    return best_eq, best_inliers

def evaluate_plane(normal, tolerance=0.1):
    vertical_vector = np.array([0, 0, 1])
    angle = np.arccos(np.clip(np.dot(normal, vertical_vector), -1.0, 1.0)) * 180 / np.pi

    if abs(angle) < tolerance:
        return "pozioma"
    elif abs(angle - 90) < tolerance:
        return "pionowa"
    else:
        return "inna"

def is_plane(avg_distance, threshold=0.05):
    return avg_distance < threshold

def visualize_clusters(clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']
    for i, cluster in enumerate(clusters):
        ax.scatter(cluster[:,0], cluster[:,1], cluster[:,2], s=1, c=colors[i % 3])
    ax.set_title("Klastry punktów (k=3)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def main():
    # 1. Wczytanie danych
    all_points = load_xyz_file("combined_surfaces.xyz")

    # 2. Klasteryzacja k-średnich
    kmeans = KMeans(n_clusters=3, random_state=0).fit(all_points)
    labels = kmeans.labels_

    clusters = [all_points[labels == i] for i in range(3)]
    visualize_clusters(clusters)

    for i, cluster in enumerate(clusters):
        print(f"\n--- Klaster {i + 1} ---")
        model, inliers = fit_plane_ransac(cluster)

        if model is None:
            print("Nie udało się dopasować płaszczyzny.")
            continue

        normal, d = model
        avg_distance = np.mean(np.abs(np.dot(cluster, normal) + d))
        is_flat = is_plane(avg_distance)

        print(f"Wektor normalny: {normal}")
        print(f"Średnia odległość punktów od płaszczyzny: {avg_distance:.4f}")

        if is_flat:
            orientation = evaluate_plane(normal)
            print(f"Chmura reprezentuje płaszczyznę {orientation}.")
        else:
            print("Chmura nie reprezentuje dobrze płaszczyzny.")

if __name__ == "__main__":
    main()

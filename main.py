import numpy as np
import matplotlib.pyplot as plt

def generate_horizontal_plane(width, length, num_points, z=0):
    x = np.random.uniform(0, width, num_points)
    y = np.random.uniform(0, length, num_points)
    z = np.full(num_points, z)
    return np.column_stack((x, y, z))

def generate_vertical_plane(width, height, num_points, x=0):
    y = np.random.uniform(0, width, num_points)
    z = np.random.uniform(0, height, num_points)
    x = np.full(num_points, x)
    return np.column_stack((x, y, z))

def generate_cylindrical_surface(radius, height, num_points):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(0, height, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack((x, y, z))

def save_combined_xyz(points, filename):
    np.savetxt(filename, points, fmt='%.2f', delimiter=' ')

def visualize_points(points, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Parametry
num_points = 1000

# Generowanie
horizontal = generate_horizontal_plane(10, 10, num_points)
vertical = generate_vertical_plane(10, 5, num_points)
cylinder = generate_cylindrical_surface(3, 7, num_points)

# Łączenie w jedną chmurę
combined = np.vstack((horizontal, vertical, cylinder))

# Zapis do pliku
save_combined_xyz(combined, "combined_surfaces.xyz")

# Wizualizacja
visualize_points(combined, "Połączona chmura punktów (3 powierzchnie)")

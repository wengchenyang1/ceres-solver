import open3d as o3d
import numpy as np


def generate_chair_point_cloud():
    # Define chair points
    chair_points = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 0.5, 0.5],
        [1, 1, 0],
        [1, 1, 1],
        [1.5, 0, 0],
        [1.5, 0, 1],
        [2, 0, 0],
        [2, 0, 1],
        [2, 0, 0.5],
    ]

    # Convert to numpy array
    chair_points = np.array(chair_points)

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(chair_points)

    return pcd


def visualize_point_cloud(pcd):
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def main():
    # Generate chair point cloud
    chair_pcd = generate_chair_point_cloud()

    # Save point cloud as .ply file
    o3d.io.write_point_cloud("chair.ply", chair_pcd)
    print("Point cloud saved as chair.ply")

    # Visualize the point cloud
    visualize_point_cloud(chair_pcd)


if __name__ == "__main__":
    main()

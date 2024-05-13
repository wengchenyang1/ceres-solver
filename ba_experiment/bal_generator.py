import numpy as np
import open3d as o3d


def read_ply(file):
    pcd = o3d.io.read_point_cloud(file)
    return np.asarray(pcd.points)


def project_points(points_3d, intrinsics, extrinsics):
    points_3d_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    R = extrinsics[:3, :3]  # rotation matrix
    T = extrinsics[:3, 3].reshape(-1, 1)  # translation vector
    RT = np.hstack([R, T])  # augmented rotation-translation matrix
    RT = np.vstack([RT, [0, 0, 0, 1]])  # add homogeneous coordinate
    points_2d = intrinsics @ (RT @ points_3d_hom.T)
    return points_2d[:2, :] / points_2d[2, :]


def write_to_bal(file, points_2d, camera_poses):
    with open(file, "w") as f:
        f.write(
            f"{len(camera_poses)} {points_2d.shape[1]} {len(camera_poses) * points_2d.shape[1]}\n"
        )
        for pose in camera_poses:
            f.write(" ".join(map(str, pose.flatten())) + "\n")
        for point in points_2d.T:
            f.write(f"{point[0]} {point[1]}\n")


# Example usage
points_3d = read_ply("chair.ply")
intrinsics = np.array(
    [[800, 0, 320], [0, 800, 240], [0, 0, 1]]
)  # example pinhole camera intrinsics
extrinsics = np.eye(4)  # example camera pose
points_2d = project_points(points_3d, intrinsics, extrinsics)
write_to_bal("output.bal", points_2d, [extrinsics])

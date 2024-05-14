import numpy as np
import open3d as o3d


def read_ply(file):
    pcd = o3d.io.read_point_cloud(file)
    return np.asarray(pcd.points)


def project_points(points_3d, intrinsics, extrinsics):
    points_3d_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    RT = extrinsics[:3, :]  # rotation-translation matrix
    points_2d_hom = intrinsics @ (RT @ points_3d_hom.T)
    epsilon = 1e-8  # small constant to avoid division by zero
    points_2d = points_2d_hom[:2, :] / (points_2d_hom[2, :] + epsilon)
    return points_2d



def write_to_bal(file, points_3d, points_2d_list, camera_poses, intrinsics):
    num_cameras = len(camera_poses)
    num_points = points_3d.shape[0]  # Number of 3D points observed by the cameras
    num_observations = sum([points.shape[1] for points in points_2d_list])

    with open(file, "w") as f:
        f.write(f"{num_cameras} {num_points} {num_observations}\n")

        # Writing observations
        for i, points_2d in enumerate(points_2d_list):
            for j in range(points_2d.shape[1]):
                f.write(f"{i} {j} {points_2d[0, j]} {points_2d[1, j]}\n")

        # Writing camera parameters
        for pose in camera_poses:
            R = pose[:, :3]
            t = pose[:, 3]
            f.write(
                f"{R[0,0]} {R[0,1]} {R[0,2]} {t[0]} {R[1,0]} {R[1,1]} {R[1,2]} {t[1]} {R[2,0]} {R[2,1]} {R[2,2]} {t[2]} {intrinsics[0, 0]} {intrinsics[1, 1]} {intrinsics[0, 2]} {intrinsics[1, 2]} 0 0\n"
            )

        # Writing 3D points
        for point in points_3d:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


# Example usage
points_3d = read_ply("chair.ply")
intrinsics = np.array(
    [[800, 0, 320], [0, 800, 240], [0, 0, 1]]
)  # example pinhole camera intrinsics
extrinsics1 = np.eye(4)  # example camera pose 1
extrinsics2 = np.eye(4)  # example camera pose 2
points_2d_1 = project_points(points_3d, intrinsics, extrinsics1)
points_2d_2 = project_points(points_3d, intrinsics, extrinsics2)
write_to_bal("output.bal", points_3d, [points_2d_1, points_2d_2], [extrinsics1, extrinsics2], intrinsics)

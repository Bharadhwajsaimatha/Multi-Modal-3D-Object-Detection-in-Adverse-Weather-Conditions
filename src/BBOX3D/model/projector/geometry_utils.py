import torch

def project_pcd_to_img(points_3D, intrinsics, extrinsics):
    N = points_3D.shape[0]

    points_hom = torch.cat([points_3D, torch.ones((N, 1), device=points_3D.device)], dim=1)  # (N, 4)

    points_cam = (extrinsics @ points_hom.T).T  # (N, 4)
    points_cam = points_cam[:, :3]


    z = points_cam[:, 2]
    valid_mask = z > 1e-5

    points_proj = (intrinsics @ points_cam.T).T  # (N, 3)
    u = points_proj[:, 0] / z
    v = points_proj[:, 1] / z
    points_2d = torch.stack([u, v], dim=1)

    return points_2d, valid_mask




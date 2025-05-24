import os
import os.path as osp
import cv2 as cv
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for extracting and storing calibration data')
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help='The dataset directory',
        dest='DATA_DIR'
    )
    parser.add_argument(
        "--use_identity_extrinsics",
        action='store_true',
        help='Whether to use estimated extrinsics or identity matrix'
    )
    args = parser.parse_args()
    return args

def estimate_intrinsics(pc):
    H, W = pc.shape[1:]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))  # (H, W)

    X = pc[0].flatten()
    Y = pc[1].flatten()
    Z = pc[2].flatten()
    u = us.flatten()
    v = vs.flatten()

    valid = Z > 0.001  # filter out invalid points
    X, Y, Z, u, v = X[valid], Y[valid], Z[valid], u[valid], v[valid]

    fx = np.median((u - W / 2) * Z / X)
    fy = np.median((v - H / 2) * Z / Y)
    cx = W / 2
    cy = H / 2

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    return K

def estimate_extrinsics(bbox3d, mask, pc, K):
    for i in range(len(bbox3d)):
        obj_mask = mask[i]
        ys, xs = np.where(obj_mask)
        if len(xs) < 50:
            continue

        object_points = []
        image_points = []

        for corner in bbox3d[i]:
            dists = np.linalg.norm(pc[:, ys, xs].transpose(1, 0) - corner, axis=1)
            min_idx = np.argmin(dists)
            x2d, y2d = xs[min_idx], ys[min_idx]
            image_points.append([x2d, y2d])
            object_points.append(corner)

        object_points = np.array(object_points).astype(np.float32)
        image_points = np.array(image_points).astype(np.float32)

        success, rvec, tvec, _ = cv.solvePnPRansac(object_points, image_points, K, None)
        if success:
            return rvec, tvec
    return None, None

def project_bboxes(bbox3d, rvec, tvec, K):
    bbox2d_list = []
    for box in bbox3d:
        proj, _ = cv.projectPoints(box, rvec, tvec, K, None)
        proj = proj.squeeze()  # shape: (8, 2)
        x_min, y_min = np.min(proj, axis=0)
        x_max, y_max = np.max(proj, axis=0)
        bbox2d = [x_min, y_min, x_max, y_max]
        bbox2d_list.append(bbox2d)
    return np.array(bbox2d_list)

def process_dir(dir, use_idty):
    pcd_data = np.load(osp.join(dir, 'pc.npy'))
    bbox3d = np.load(osp.join(dir, 'bbox3d.npy'))
    mask = np.load(osp.join(dir, 'mask.npy'))
    # image = Image.open(osp.join(dir,'rgb.jpg'))

    mat_K = estimate_intrinsics(pcd_data)
    rvec, tvec = estimate_extrinsics(bbox3d, mask, pcd_data, mat_K)
    if rvec is None or tvec is None:
        print(f"[WARNING] Could not estimate extrinsics for {dir}")
        return None, None, None

    rot_mat, _ = cv.Rodrigues(rvec)
    mat_E = np.eye(4)
    mat_E[:3, :3] = rot_mat
    mat_E[:3, 3] = tvec.squeeze()

    if use_idty:
        rvec = np.array([0.0,0.0,0.0])
        tvec = np.array([0.0,0.0,0.0])
        mat_E = np.eye(4,4)

    bbox2d = project_bboxes(bbox3d, rvec, tvec, mat_K)

    return mat_K, mat_E, bbox2d

def main():
    args = parse_args()
    data_root = args.DATA_DIR
    USE_IDTY = False

    for dir_name in tqdm(sorted(os.listdir(data_root))):
        dir_path = osp.join(data_root, dir_name)
        if not osp.isdir(dir_path):
            continue

        print(f"Processing... {dir_name}")
        if args.use_identity_extrinsics:
            USE_IDTY = True

        mat_K, mat_E, bbox2d = process_dir(dir_path, USE_IDTY)

        if mat_K is None:
            continue
        with open(osp.join(dir_path, 'calib.txt'), 'w') as f:
            f.write(' '.join(map(str, mat_K.flatten())) + '\n')
            f.write(' '.join(map(str, mat_E.flatten())) + '\n')

        # Save 2D bounding boxes
        np.save(osp.join(dir_path, 'bbox2d.npy'), bbox2d)

if __name__ == "__main__":
    main()

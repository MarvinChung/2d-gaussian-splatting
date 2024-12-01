import trimesh

import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt

def align(model, data):

    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3,-1))
    data_zerocentered = data - data.mean(1).reshape((3,-1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error

def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(float))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp

def trimesh_to_o3d_pc(mesh):
    """
    Convert a Trimesh mesh to an Open3D point cloud.
    """
    points = np.asarray(mesh.vertices)
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(points)
    return o3d_pc

def get_align_transformation(rec_mesh, gt_mesh):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """

    # Convert Trimesh meshes to Open3D point clouds
    o3d_rec_pc = trimesh_to_o3d_pc(rec_mesh)
    o3d_gt_pc = trimesh_to_o3d_pc(gt_mesh)
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    transformation = reg_p2p.transformation
    return transformation

def evaluate_mesh_3d_metric(rec_meshfile, gt_meshfile, align=True):
    """
    3D reconstruction metric.

    """
    try:
        mesh_rec = trimesh.load(rec_meshfile, process=False)
    except:
        print(f"[Evaluator] rec_meshfile: {rec_meshfile} is not exist")

    try:
        mesh_gt = trimesh.load(gt_meshfile, process=False)
    except:
        print(f"[Evaluator] gt_meshfile: {gt_meshfile} is not exist")

    if align:
        transformation = get_align_transformation(mesh_rec, mesh_gt)
        print(transformation)
        mesh_rec = mesh_rec.apply_transform(transformation)

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(
        gt_pc_tri.vertices, rec_pc_tri.vertices)
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %

    return accuracy_rec, completion_rec, completion_ratio_rec


import argparse

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate 3D reconstruction metrics.")
    parser.add_argument(
        "--gt_mesh", type=str, required=True,
        help="Path to the ground truth mesh file."
    )
    parser.add_argument(
        "--rec_mesh", type=str, required=True,
        help="Path to the reconstructed mesh file."
    )
    parser.add_argument(
        "--align", action="store_true",
        help="Whether to align the reconstructed mesh to the ground truth."
    )
    args = parser.parse_args()

    # Call the evaluation function
    accuracy_rec, completion_rec, completion_ratio_rec = evaluate_mesh_3d_metric(
        rec_meshfile=args.rec_mesh, gt_meshfile=args.gt_mesh, align=args.align
    )

    # Print the results
    print(f"Accuracy: {accuracy_rec:.2f} cm")
    print(f"Completion: {completion_rec:.2f} cm")
    print(f"Completion Ratio: {completion_ratio_rec:.2f} %")

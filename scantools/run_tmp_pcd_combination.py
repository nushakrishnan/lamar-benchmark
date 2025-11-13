import open3d as o3d
from pathlib import Path
import argparse
from tqdm import tqdm
import os
import numpy as np



def merge_pointclouds_navvis(pcd_folder: Path) -> o3d.t.geometry.PointCloud:
    ply_files = sorted(pcd_folder.glob("*.ply"))
    if len(ply_files) == 0:
        raise ValueError(f"No PLY files found in {pcd_folder}")
    if len(ply_files) == 1:
        return o3d.t.io.read_point_cloud(str(ply_files[0]))

    all_points = []
    all_origin_x = []
    all_origin_y = []
    all_origin_z = []
    all_intensity = []

    has_intensity = None

    for ply_file in tqdm(ply_files, desc="Reading tiles"):
        pcd = o3d.t.io.read_point_cloud(str(ply_file))

        pts = pcd.point["positions"].numpy()
        ox = pcd.point["origin_x"].numpy()
        oy = pcd.point["origin_y"].numpy()
        oz = pcd.point["origin_z"].numpy()

        all_points.append(pts)
        all_origin_x.append(ox)
        all_origin_y.append(oy)
        all_origin_z.append(oz)

        if has_intensity is None:
            has_intensity = "intensity" in list(pcd.point)

        if has_intensity:
            if "intensity" not in list(pcd.point):
                raise ValueError(
                    f"File {ply_file} is missing 'intensity' while others have it."
                )
            all_intensity.append(pcd.point["intensity"].numpy())

    points = np.concatenate(all_points, axis=0)
    origin_x = np.concatenate(all_origin_x, axis=0)
    origin_y = np.concatenate(all_origin_y, axis=0)
    origin_z = np.concatenate(all_origin_z, axis=0)

    device = o3d.core.Device("CPU:0")
    merged = o3d.t.geometry.PointCloud(device)

    merged.point["positions"] = o3d.core.Tensor(points, o3d.core.Dtype.Float32)
    merged.point["origin_x"] = o3d.core.Tensor(origin_x, o3d.core.Dtype.Float32)
    merged.point["origin_y"] = o3d.core.Tensor(origin_y, o3d.core.Dtype.Float32)
    merged.point["origin_z"] = o3d.core.Tensor(origin_z, o3d.core.Dtype.Float32)

    if has_intensity:
        intensity = np.concatenate(all_intensity, axis=0)
        merged.point["intensity"] = o3d.core.Tensor(intensity, o3d.core.Dtype.Float32)

    return merged


def main(pcd_folder: Path, out_ply: Path):
    print("Parent exists:", out_ply.parent.exists())
    print("Parent writable:", os.access(out_ply.parent, os.W_OK))

    merged_pcd = merge_pointclouds_navvis(pcd_folder)
    o3d.t.io.write_point_cloud(str(out_ply), merged_pcd)
    print("Wrote merged cloud to:", out_ply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge NavVis tilecloud PLYs into one"
    )
    parser.add_argument("pcd_folder", type=Path, help="Folder containing PLY tiles.")
    parser.add_argument("output_ply", type=Path, help="Output merged PLY file path.")
    args = parser.parse_args()

    main(args.pcd_folder, args.output_ply)
#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple
import good

import numpy as np
from numpy.core.numeric import indices
import sortednp as snp

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    TriangulationParameters,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    _remove_correspondences_with_ids
)

import _corners

import cv2        

def filter_inliers(corner_storage: CornerStorage, inliers):
    mask = np.zeros(corner_storage.max_corner_id)
    mask[inliers] = 1
    def predicate(corners):
        return mask[corners]

    _corners.StorageFilter(corner_storage)

def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    point_cloud_builder = PointCloudBuilder()

    min_track_length = np.quantile(_corners.calc_track_len_array_mapping(corner_storage), 0.7)
    print(f"Min track length: {min_track_length}")
    corner_storage = _corners.without_short_tracks(corner_storage, min_track_length)


    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])

    triang_params = TriangulationParameters(2, 30, 1)
    
    points3d, ids, median_cos = triangulate_correspondences(correspondences, view_mats[known_view_1[0]], view_mats[known_view_2[0]], intrinsic_mat, triang_params)
    while points3d.shape[0] / correspondences.ids.shape[0] < 0.8 and triang_params.min_triangulation_angle_deg > 0.01:
        triang_params = TriangulationParameters(triang_params.max_reprojection_error + 0.02, triang_params.min_triangulation_angle_deg / 1.6, 0.4)
        points3d, ids, median_cos = triangulate_correspondences(correspondences, view_mats[known_view_1[0]], view_mats[known_view_2[0]], intrinsic_mat, triang_params)

    print(triang_params)

    point_cloud_builder.add_points(ids, points3d)

    unprocessed_frames = list(range(len(corner_storage)))
    unprocessed_frames.remove(known_view_1[0])
    unprocessed_frames.remove(known_view_2[0])
    processed_frames = [known_view_1[0], known_view_2[0]]

    good_indices = point_cloud_builder.ids.flatten()

    start_inliers = good_indices.shape[0]

    while len(unprocessed_frames) > 0:
        best_frame, points_idxs = None, None
        
        for frame in unprocessed_frames:
            cur_intersection = snp.intersect(corner_storage[frame].ids.flatten(), good_indices)
            if best_frame is None or len(cur_intersection) > len(points_idxs):
                best_frame = frame
                points_idxs = cur_intersection

        print(f"Calculating pose for the frame #{best_frame}. Used 3D points: {len(points_idxs)}")

        frame_good_indices = points_idxs

        _, (points_frame_idxs, _) = snp.intersect(corner_storage[best_frame].ids.flatten(), frame_good_indices, indices=True)
        _, (points_cloud_idxs, _) = snp.intersect(point_cloud_builder.ids.flatten(), frame_good_indices, indices=True)
        points3d_for_pnp = point_cloud_builder.points[points_cloud_idxs]

        success, r_vec, t_vec, inliers = cv2.solvePnPRansac(points3d_for_pnp, corner_storage[best_frame].points[points_frame_idxs], intrinsic_mat, None,
            reprojectionError=triang_params.max_reprojection_error,
            flags=cv2.SOLVEPNP_ITERATIVE,
            confidence=0.9999
        )

        inliers = np.array([]).astype(np.int32) if inliers is None else np.array(inliers).flatten()
        inliers = corner_storage[best_frame].ids[points_frame_idxs][inliers].flatten()

        if inliers.shape[0] > max(20, 0.05 * start_inliers):
            good_indices = inliers

        print(f"Inliers: {inliers.shape}, Good indices: {good_indices.shape}")

        view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
        view_mats[best_frame] = view_mat

        total_added = 0
        for frame in processed_frames[-200:][:150]:
            correspondences = build_correspondences(corner_storage[best_frame], corner_storage[frame], ids_to_remove=good_indices)

            if correspondences.ids.shape[0] > 0:
                triang_params_frame = TriangulationParameters(3, 20, 0.4)
                points3d, ids, median_cos = triangulate_correspondences(correspondences, view_mats[best_frame], view_mats[frame], intrinsic_mat, triang_params_frame)

                total_added += ids.shape[0]
                point_cloud_builder.add_points(ids, points3d)
                good_indices = np.hstack((good_indices, ids))
                good_indices = np.sort(good_indices)

        print(f"New 3d points from the frame #{best_frame}: {total_added}")


        unprocessed_frames.remove(best_frame)
        processed_frames.append(best_frame)


    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()

#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple
from collections import namedtuple

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
    eye3x4,
    TriangulationParameters,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    _remove_correspondences_with_ids,
    check_inliers_mask,
    Correspondences
)

import _corners
import cv2

from dataclasses import dataclass

@dataclass
class Params:
    frame_prefix: int
    triangulate_params: TriangulationParameters
    reprojection_error: float
    prob: float
    inliers: int
    inliers_ratio: float

PARAMS = Params(
    prob=0.9999,
    frame_prefix=3,
    triangulate_params=TriangulationParameters(0.9, 6, 0.1),
    reprojection_error=3.0,
    inliers=10,
    inliers_ratio=0.35,
)

class Tracker():
    def __init__(self, intrinsic_mat, corner_storage, params: Params = PARAMS):
        self.intrinsic_mat = intrinsic_mat
        self.corner_storage = corner_storage

        self.n = len(self.corner_storage)

        self.params = params

    def find_essential(self, i, j, hom_ratio):
        correspondences = build_correspondences(self.corner_storage[i], self.corner_storage[j])

        essential, inliers = cv2.findEssentialMat(
            correspondences.points_1,
            correspondences.points_2,
            self.intrinsic_mat,
            method=cv2.RANSAC,
            prob=self.params.prob,
            threshold=1.0
        )
        inliers = inliers.flatten()

        _, inliers_h = cv2.findHomography(
            correspondences.points_1,
            correspondences.points_2,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.params.reprojection_error,
            confidence=self.params.prob
        )
        inliers_h = inliers_h.flatten()

        if inliers_h.sum() / inliers.sum() > hom_ratio or check_inliers_mask(inliers, self.params.inliers, self.params.inliers_ratio) == False:
            return None, None

        correspondences = Correspondences(
            correspondences.ids[inliers],
            correspondences.points_1[inliers],
            correspondences.points_2[inliers]
        )

        return essential, correspondences

    def find_view(self, i, j, hom_ratio, min_points):
        essential, correspondences = self.find_essential(i, j, hom_ratio)

        if essential is None:
            return None

        rotation1, rotation2, translation = cv2.decomposeEssentialMat(essential)
        best_view, best_point_count = None, min_points
        for R in [rotation1, rotation2]:
            for t in [-translation, translation]:
                view = np.hstack((R, t.reshape(-1, 1)))
                _, corr_ids, _ = triangulate_correspondences(
                    correspondences, eye3x4(), view,
                    self.intrinsic_mat, self.params.triangulate_params
                )
                point_count = len(corr_ids)
                if point_count > best_point_count:
                    best_view = view
                    best_point_count = point_count

        return best_view

    def find_among_all(self, max_hom_ratio, min_points):
        for i in range(self.params.frame_prefix):
            for j in range(self.n - 1, i + 19, -1):
                print(f"Trying frames: {i}, {j}")
                view = self.find_view(i, j, max_hom_ratio, min_points)
                if view is not None:
                    return ((i, view_mat3x4_to_pose(eye3x4())), (j, view_mat3x4_to_pose(view)))
        return None

    def calculate_init_poses(self):    
        max_hom_ratio = 0.35
        min_points = 250
        while True:
            res = self.find_among_all(max_hom_ratio, min_points)
            if res is not None:
                return res
            max_hom_ratio += 0.1
            min_points //= 2
            min_points -= 1
            self.params.inliers_ratio -= 0.08
            self.params.inliers -= 2

            self.params.triangulation_params = TriangulationParameters(max_hom_ratio * 2.0, self.params.triangulate_params.min_triangulation_angle_deg / 2.0, 0.9)

                
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
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    min_track_length = np.quantile(_corners.calc_track_len_array_mapping(corner_storage), 0.6)
    print(f"Min track length: {min_track_length}")
    corner_storage = _corners.without_short_tracks(corner_storage, min_track_length)

    frame_count = len(corner_storage)

    triang_params = TriangulationParameters(0.9, 5, 0.4)

    tracker = Tracker(intrinsic_mat, corner_storage)

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = tracker.calculate_init_poses()

    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    point_cloud_builder = PointCloudBuilder()

    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])

    print(f"Init frames: {known_view_1[0]}, {known_view_2[0]}")

    points3d, ids, median_cos = triangulate_correspondences(correspondences, view_mats[known_view_1[0]], view_mats[known_view_2[0]], intrinsic_mat, triang_params)
    while points3d.shape[0] / correspondences.ids.shape[0] < 0.7 and triang_params.min_triangulation_angle_deg > 0.9:
        triang_params = TriangulationParameters(triang_params.max_reprojection_error + 0.02, triang_params.min_triangulation_angle_deg / 1.6, 0.4)
        points3d, ids, median_cos = triangulate_correspondences(correspondences, view_mats[known_view_1[0]], view_mats[known_view_2[0]], intrinsic_mat, triang_params)

    print(triang_params)

    point_cloud_builder.add_points(ids.astype(np.int64), points3d)

    unprocessed_frames = list(range(len(corner_storage)))
    processed_frames = []

    good_indices = point_cloud_builder.ids.flatten()

    start_inliers = good_indices.shape[0]

    fast = 0

    while len(unprocessed_frames) > 0:
        best_frame, best_inliers, points_idxs = None, -1, None
        
        for idx, frame in enumerate(unprocessed_frames):
            if best_frame is not None and (unprocessed_frames[0] + 70 < frame or (((idx > 6 and fast >= 7) or (len(processed_frames) > 0 and abs(frame - processed_frames[-1]) > 100)))):
                continue
            cur_intersection = snp.intersect(corner_storage[frame].ids.flatten(), good_indices)

            _, (points_frame_idxs, _) = snp.intersect(corner_storage[frame].ids.flatten(), cur_intersection, indices=True)
            _, (points_cloud_idxs, _) = snp.intersect(point_cloud_builder.ids.flatten(), cur_intersection, indices=True)
            points3d_for_pnp = point_cloud_builder.points[points_cloud_idxs]

            success, r_vec, t_vec, inliers = cv2.solvePnPRansac(points3d_for_pnp, corner_storage[frame].points[points_frame_idxs], intrinsic_mat, None,
                reprojectionError=triang_params.max_reprojection_error,
                flags=cv2.SOLVEPNP_EPNP,
                confidence=0.999
            )
            inliers = np.array([]).astype(np.int64) if inliers is None else np.array(inliers).flatten()

            if inliers.sum() > best_inliers:
                best_frame = frame
                best_inliers = inliers.sum()
                points_idxs = cur_intersection

        if len(processed_frames) > 0 and abs(processed_frames[-1] - best_frame) <= 6:
            fast += 1
        else:
            fast = 0

        print(f"#{best_frame} ({len(processed_frames)}/{frame_count}).\tUsed 3D points: {len(points_idxs)}",end="\t")

        frame_good_indices = points_idxs

        _, (points_frame_idxs, _) = snp.intersect(corner_storage[best_frame].ids.flatten(), frame_good_indices, indices=True)
        _, (points_cloud_idxs, _) = snp.intersect(point_cloud_builder.ids.flatten(), frame_good_indices, indices=True)
        points3d_for_pnp = point_cloud_builder.points[points_cloud_idxs]

        success, r_vec, t_vec, inliers = cv2.solvePnPRansac(points3d_for_pnp, corner_storage[best_frame].points[points_frame_idxs], intrinsic_mat, None,
            reprojectionError=triang_params.max_reprojection_error,
            flags=cv2.SOLVEPNP_ITERATIVE,
            confidence=0.9999
        )
        print(f"Success: {success}", end="\t")

        inliers = np.array([]).astype(np.int64) if inliers is None else np.array(inliers).flatten()
        inliers = corner_storage[best_frame].ids[points_frame_idxs][inliers].flatten()

        if inliers.shape[0] > max(20, 0.025 * start_inliers):
            good_indices = inliers

        print(f"Inliers: {inliers.shape}, Good indices: {good_indices.shape}", end="\t")

        view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
        view_mats[best_frame] = view_mat

        total_added = 0

        for frame in processed_frames[::-1]:
            correspondences = build_correspondences(corner_storage[best_frame], corner_storage[frame], ids_to_remove=good_indices)

            if correspondences.ids.shape[0] > 0:
                triang_params_frame = TriangulationParameters(1.0, 6, 0.4)
                points3d, ids, median_cos = triangulate_correspondences(correspondences, view_mats[best_frame], view_mats[frame], intrinsic_mat, triang_params_frame)

                if 0.75 < median_cos:
                    total_added += ids.shape[0]
                    point_cloud_builder.add_points(ids, points3d)
                    good_indices = np.hstack((good_indices, ids))
                    good_indices = np.sort(good_indices)
        
        print(f"New 3d points: {total_added};\tLast median_cos: {median_cos}")

        if (success and inliers.shape[0] > 12) or (total_added == 0):
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

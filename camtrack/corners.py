#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))

def convert(img):
    return cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    # TODO


    flow_params = dict(winSize=(15, 15), maxLevel=3, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.01))

    image_0 = frame_sequence[0]

    DST = 15
    CONST = 10
    N = image_0.shape[0] * image_0.shape[1] // DST // DST // 10

    corners_search_params = dict(blockSize=5, qualityLevel=0.02, minDistance=DST)

    corners_raw_0 = np.squeeze(cv2.goodFeaturesToTrack(image_0, N, **corners_search_params), axis=1)

    corners_0 = FrameCorners(
        np.arange(0, corners_raw_0.shape[0]),
        corners_raw_0,
        np.full_like(corners_raw_0, DST)
    )

    builder.set_corners_at_frame(0, corners_0)

    max_corner_id = corners_0.ids.max()

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        cr1, continued, _ = cv2.calcOpticalFlowPyrLK(convert(image_0), convert(image_1), corners_0.points, None, **flow_params)

        idxs = (cr1[:, 1] < image_1.shape[0] - 1) & (cr1[:, 0] < image_1.shape[1] - 1) & (cr1[:, 0] >= 0) & (cr1[:, 1] >= 0) & (continued.flatten() == 1)

        visited_pixels = np.zeros_like(image_1, dtype=np.uint8)
        for i, (fx, fy) in enumerate(cr1):
            x, y = int(fx), int(fy)
            if 0 <= y < image_1.shape[0] and 0 <= x < image_1.shape[1] and visited_pixels[y, x] == 1:
                idxs[i] = 0
            else:
                cv2.circle(visited_pixels, (x, y), DST, 1, -1)


        cr1 = cr1[idxs]

        corners_ids_1 = corners_0.ids[idxs]

        corners_dilated_1 = np.zeros_like(image_1, dtype=np.uint8)
        corners_dilated_1[cr1.astype(np.int32)[:, 1], cr1.astype(np.int32)[:, 0]] = 1

        kernel = np.ones((2 * DST, 2 * DST), np.uint8)
        corners_raw_ext = 1 - cv2.dilate(corners_dilated_1, kernel=kernel, iterations=1)


        corners_raw_extra = cv2.goodFeaturesToTrack(image_1, max(1, N - cr1.shape[0]), **corners_search_params, mask=corners_raw_ext)
        if len(corners_raw_extra.shape) > 0:
            corners_raw_extra = np.squeeze(corners_raw_extra, axis=1)
        
        ids_extra = np.arange(0, corners_raw_extra.shape[0]) + max_corner_id

        max_corner_id += corners_raw_extra.shape[0]

        all_ids = np.concatenate((corners_ids_1.flatten(), ids_extra))

        corners_1 = FrameCorners(
            all_ids,
            np.concatenate((cr1, corners_raw_extra)),
            np.full_like(all_ids, DST)
        )

        builder.set_corners_at_frame(frame, corners_1)

        corners_0 = corners_1
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter

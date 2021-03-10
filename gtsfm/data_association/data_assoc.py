""" Create 2D-3D data association as a precursor to Bundle Adjustment.
1. Forms feature tracks from verified correspondences and global poses.
2. Estimates 3D landmark for each track (Ransac and simple triangulation modes available)
3. Filters tracks based on reprojection error.

References: 
1. Richard I. Hartley and Peter Sturm. Triangulation. Computer Vision and Image Understanding, Vol. 68, No. 2,
   November, pp. 146–157, 1997

Authors: Sushmita Warrier, Xiaolong Wu
"""
from typing import Dict, List, NamedTuple, Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import PinholeCameraCal3Bundler, SfmData, SfmTrack

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.sfm_result import SfmResult
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.data_association.point3d_initializer import (
    Point3dInitializer,
    TriangulationParam,
)


logger = logger_utils.get_logger()


class DataAssociation(NamedTuple):
    """Class to form feature tracks; for each track, call LandmarkInitializer.

    Args:
        reproj_error_thresh: the maximum reprojection error allowed.
        min_track_len: min length required for valid feature track / min nb of supporting views required for a landmark
                       to be valid.
        mode: triangulation mode, which dictates whether or not to use robust estimation.
        num_ransac_hypotheses (optional): number of hypothesis for RANSAC-based triangulation.
    """

    reproj_error_thresh: float
    min_track_len: int
    mode: TriangulationParam
    num_ransac_hypotheses: Optional[int] = None

    def __validate_track(self, sfm_track: Optional[SfmTrack]) -> bool:
        """Validate the track by checking its length."""
        return sfm_track is not None and sfm_track.number_measurements() >= self.min_track_len

    def run(
        self,
        cameras: Dict[int, PinholeCameraCal3Bundler],
        corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
        keypoints_list: List[Keypoints],
    ) -> SfmData:
        """Perform the data association.

        Args:
            cameras: dictionary, with image index -> camera mapping.
            corr_idxs_dict: dictionary, with key as image pair (i1,i2) and value as matching keypoint indices.
            keypoints_list: keypoints for each image.

        Returns:
            cameras and tracks as SfmData.
        """
        # generate tracks for 3D points using pairwise correspondences
        tracks_2d = SfmTrack2d.generate_tracks_from_pairwise_matches(corr_idxs_dict, keypoints_list)

        # metrics on tracks w/o triangulation check
        num_tracks_2d = len(tracks_2d)
        track_lengths = list(map(lambda x: x.number_measurements(), tracks_2d))
        mean_2d_track_length = np.mean(track_lengths)

        logger.debug("[Data association] input number of tracks: %s", num_tracks_2d)
        logger.debug("[Data association] input avg. track length: %s", mean_2d_track_length)

        # initializer of 3D landmark for each track
        point3d_initializer = Point3dInitializer(
            cameras,
            self.mode,
            self.reproj_error_thresh,
            self.num_ransac_hypotheses,
        )

        num_tracks_w_cheirality_exceptions = 0
        per_accepted_track_avg_errors = []
        per_rejected_track_avg_errors = []
        # form SFMdata object after triangulation
        triangulated_data = SfmData()
        for track_2d in tracks_2d:
            # triangulate and filter based on reprojection error
            sfm_track, avg_track_reproj_error, is_cheirality_failure = point3d_initializer.triangulate(track_2d)
            if is_cheirality_failure:
                num_tracks_w_cheirality_exceptions += 1

            if sfm_track is not None and self.__validate_track(sfm_track):
                triangulated_data.add_track(sfm_track)
                per_accepted_track_avg_errors.append(avg_track_reproj_error)
            else:
                per_rejected_track_avg_errors.append(avg_track_reproj_error)

        num_accepted_tracks = triangulated_data.number_tracks()
        accepted_tracks_ratio = num_accepted_tracks / len(tracks_2d)
        track_cheirality_failure_ratio = num_tracks_w_cheirality_exceptions / len(tracks_2d)

        # TODO: improve dropped camera handling
        num_cameras = len(cameras.keys())
        expected_camera_indices = np.arange(num_cameras)
        # add cameras to landmark_map
        for i, cam in cameras.items():
            if i != expected_camera_indices[i]:
                raise RuntimeError("Some cameras must have been dropped ")
            triangulated_data.add_camera(cam)

        mean_3d_track_length, median_3d_track_length, track_lengths_3d = SfmResult(
            triangulated_data, None
        ).get_track_length_statistics()

        logger.debug("[Data association] output number of tracks: %s", num_accepted_tracks)
        logger.debug("[Data association] output avg. track length: %s", mean_3d_track_length)

        # dump the 3d point cloud before Bundle Adjustment for offline visualization
        points_3d = [list(triangulated_data.track(j).point3()) for j in range(num_accepted_tracks)]
        # bin edges are halfway between each integer
        track_lengths_histogram, _ = np.histogram(track_lengths_3d, bins=np.linspace(-0.5, 10.5, 12))

        # min possible track len is 2, above 10 is improbable
        histogram_dict = {f"num_len_{i}_tracks": int(track_lengths_histogram[i]) for i in range(2, 11)}

        data_assoc_metrics = {
            "mean_2d_track_length": np.round(mean_2d_track_length, 3),
            "accepted_tracks_ratio": np.round(accepted_tracks_ratio, 3),
            "track_cheirality_failure_ratio": np.round(track_cheirality_failure_ratio, 3),
            "num_accepted_tracks": num_accepted_tracks,
            "3d_tracks_length": {
                "median": median_3d_track_length,
                "mean": mean_3d_track_length,
                "min": int(track_lengths_3d.min()),
                "max": int(track_lengths_3d.max()),
                "track_lengths_histogram": histogram_dict,
            }
        }

        # placing long lists at the end of dictionary, since they are less easily interpreted
        data_assoc_metrics.update(
            {
                "mean_accepted_track_avg_error": np.array(per_accepted_track_avg_errors).mean(),
                "per_rejected_track_avg_errors": per_rejected_track_avg_errors,
                "per_accepted_track_avg_errors": per_accepted_track_avg_errors,
                "points_3d": points_3d,
            }
        )

        return triangulated_data, data_assoc_metrics

    def create_computation_graph(
        self,
        cameras: Delayed,
        corr_idxs_graph: Dict[Tuple[int, int], Delayed],
        keypoints_graph: List[Delayed],
    ) -> Delayed:
        """Creates a computation graph for performing data association.

        Args:
            cameras: list of cameras wrapped up as Delayed.
            corr_idxs_graph: dictionary of correspondence indices, each value wrapped up as Delayed.
            keypoints_graph: list of wrapped up keypoints for each image.

        Returns:
            SfmData object wrapped up using dask.delayed.
        """
        return dask.delayed(self.run)(cameras, corr_idxs_graph, keypoints_graph)

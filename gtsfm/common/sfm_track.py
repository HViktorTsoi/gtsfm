"""Utilities to generate and store tracks. Uses the Union-Find algorithm, with image ID and keypoint index for that
image as the unique keys.

A track is defined as a 2d measurement of a single 3d landmark seen in multiple different images.

References:
1. P. Moulon, P. Monasse. Unordered Feature Tracking Made Fast and Easy, 2012, HAL Archives.
   https://hal-enpc.archives-ouvertes.fr/hal-00769267/file/moulon_monasse_featureTracking_CVMP12.pdf

Authors: Ayush Baid, Sushmita Warrier, John Lambert
"""
from typing import Dict, List, NamedTuple, Tuple

import gtsam
import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints

logger = logger_utils.get_logger()


class SfmMeasurement(NamedTuple):
    """2d measurements (points in images)."""

    i: int  # camera index
    uv: np.ndarray  # 2d measurement

    def __eq__(self, other: object) -> bool:
        """Checks equality with the other object."""
        if not isinstance(other, SfmMeasurement):
            return False

        if self.i != other.i:
            return False

        return np.allclose(self.uv, other.uv)

    def __ne__(self, other: object) -> bool:
        """Checks inequality with the other object."""
        return not self == other


class SfmTrack2d(NamedTuple):
    """Track containing 2D measurements associated with a single 3D point.

    Note: Equivalent to gtsam.SfmTrack, but without the 3d measurement. This class holds data temporarily before 3D
          point is initialized.
    """

    measurements: List[SfmMeasurement]

    def number_measurements(self) -> int:
        """Returns the number of measurements."""
        return len(self.measurements)

    def measurement(self, idx: int) -> SfmMeasurement:
        """Getter for measurement at a particular index.

        Args:
            idx: index to fetch.

        Returns:
            measurement at the requested index.
        """
        return self.measurements[idx]

    def select_subset(self, idxs: Tuple[int, ...]) -> "SfmTrack2d":
        """Generates a new track with the subset of measurements.

        Returns:
            Track with the subset of measurements.
        """
        inlier_measurements = [self.measurements[j] for j in idxs]

        return SfmTrack2d(inlier_measurements)

    def __eq__(self, other: object) -> bool:
        """Checks equality with the other object."""

        # check object type
        if not isinstance(other, SfmTrack2d):
            return False

        # check number of measurements
        if len(self.measurements) != len(other.measurements):
            return False

        # check the individual measurements (order insensitive)
        # inefficient implementation but wont be used a lot
        for measurement in self.measurements:
            if measurement not in other.measurements:
                return False

        return True

    def __ne__(self, other: object) -> bool:
        """Checks inequality with the other object."""
        return not self == other

    def validate_unique_cameras(self) -> bool:
        """Validates the track by checking that no two measurements are from the same camera.

        Returns:
            boolean result of the validation.
        """
        track_cam_idxs = [measurement.i for measurement in self.measurements]
        return len(set(track_cam_idxs)) == len(track_cam_idxs)

    def generate_pairs(self) -> List[Tuple[int, int, "SfmTrack2d"]]:
        """Generete all possible pair from the measurements.

        Returns:
            List of all the pair subtracks.
        """
        pairs: List[Tuple[int, int, "SfmTrack2d"]] = []

        # loop over all pairs (k1, k2) with contstraints k1 < k2.
        for k1 in range(self.number_measurements()):
            for k2 in range(k1 + 1, self.number_measurements()):
                pairs.append((k1, k2, self.select_subset([k1, k2])))

        return pairs

    def generate_triplets(self) -> List[Tuple[int, int, int, "SfmTrack2d"]]:
        """Generete all possible triplets from the measurements.

        Returns:
            List of all the triplet subtracks.
        """
        triplets: List[Tuple[int, int, int, "SfmTrack2d"]] = []

        # loop over all triplets (k1, k2, k3) with constraints k1 < k2 < k3.
        for k1 in range(self.number_measurements()):
            for k2 in range(k1 + 1, self.number_measurements()):
                for k3 in range(k2 + 1, self.number_measurements()):
                    triplets.append((k1, k2, k3, self.select_subset([k1, k2, k3])))

        return triplets

    @staticmethod
    def generate_tracks_from_pairwise_matches(
        matches_dict: Dict[Tuple[int, int], np.ndarray],
        keypoints_list: List[Keypoints],
    ) -> List["SfmTrack2d"]:
        """Factory function that creates a list of tracks from 2d point correspondences.

        Creates a disjoint-set forest (DSF) and 2d tracks from pairwise matches. We create a singleton for union-find
        set elements from camera index of a detection and the index of that detection in that camera's keypoint list,
        i.e. (i,k).

        Args:
            matches_dict: Dict of pairwise matches of type:
                    key: pose indices for the matched pair of images
                    val: feature indices, as array of Nx2 shape; N being nb of features. A row is (feature_idx1,
                         feature_idx2).
            keypoints_list: List of keypoints for each image.

        Returns:
            list of all valid SfmTrack2d generated by the matches.
        """
        # check to ensure dimensions of coordinates are correct
        dims_valid = all([kps.coordinates.ndim == 2 for kps in keypoints_list])
        if not dims_valid:
            raise Exception("Dimensions for Keypoint coordinates incorrect. Array needs to be 2D")

        # Generate the DSF to form tracks
        dsf = gtsam.DSFMapIndexPair()
        track_2d_list = []
        # for DSF finally
        # measurement_idxs represented by ks
        for (i1, i2), k_pairs in matches_dict.items():
            for (k1, k2) in k_pairs:
                dsf.merge(gtsam.IndexPair(i1, k1), gtsam.IndexPair(i2, k2))

        key_set = dsf.sets()

        erroneous_track_count = 0
        # create a landmark map: a list of tracks
        # Each track is represented as a list of (camera_idx, measurements)
        for set_id in key_set:
            index_pair_set = key_set[set_id]  # key_set is a wrapped C++ map, so this unusual syntax is required

            # Initialize track from measurements
            track_measurements = []
            for index_pair in gtsam.IndexPairSetAsArray(index_pair_set):
                # camera_idx is represented by i
                # measurement_idx is represented by k
                i = index_pair.i()
                k = index_pair.j()
                # add measurement in this track
                track_measurements += [SfmMeasurement(i, keypoints_list[i].coordinates[k])]

            track_2d_list.append(SfmTrack2d(track_measurements))

        return track_2d_list

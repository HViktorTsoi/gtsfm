"""Utility functions for comparing different types related to geometry.

Authors: Ayush Baid, John Lambert
"""
import copy
import math
from typing import List, Optional, Tuple

import numpy as np
from gtsam import Point3, Pose3, Pose3Pairs, Rot3, Similarity3, Unit3
from scipy.spatial.transform import Rotation

from gtsfm.utils.logger import get_logger

EPSILON = np.finfo(float).eps
DEFAULT_RANSAC_ALIGNMENT_DELETE_FRAC = 0.10

logger = get_logger()


def align_rotations(aRi_list: List[Rot3], bRi_list: List[Rot3]) -> List[Rot3]:
    """Aligns the list of rotations to the reference list by shifting origin.

    TODO (John): replace later with Karcher mean to account for noisy estimates

    Args:
        aRi_list: reference rotations in frame "a" which are the targets for alignment
        bRi_list: input rotations which need to be aligned to frame "a"

    Returns:
        aRi_list_: transformed input rotations previously "bRi_list" but now which
            have the same origin as reference (now living in "a" frame)
    """
    aRi0 = aRi_list[0]

    bRi0 = bRi_list[0]
    i0Rb = bRi0.inverse()

    # origin_transform -- map the origin of the input list to the reference list
    aRb = aRi0.compose(i0Rb)

    # apply the coordinate shift to all entries in input
    return [aRb.compose(bRi) for bRi in bRi_list]


def ransac_align_poses_sim3_ignore_missing(
    aTi_list_ref: List[Optional[Pose3]],
    bTi_list_est: List[Optional[Pose3]],
    num_iters: int = 1000,
    delete_frac: float = DEFAULT_RANSAC_ALIGNMENT_DELETE_FRAC,
) -> Tuple[List[Optional[Pose3]], Similarity3]:
    """Align pose graphs by estimating a Similarity(3) transformation, while accounting for outliers in the pose graph.

    Args:
        aTi_list_ref: pose graph 1 (reference/target to align to).
        bTi_list_est: pose graph 2 (to be aligned).
        num_iters: number of RANSAC iterations to execture.
        delete_frac: what percent of data to remove when fitting a single hypothesis.

    Returns:
        best_aligned_bTi_list_est_full: transformed input poses previously "bTi_list" but now which
            have the same origin and scale as reference (now living in "a" frame). Rather than
            representing an aligned subset, this represents the entire aligned pose graph.
        best_aSb: Similarity(3) object that aligns the two pose graphs (best among all hypotheses).
    """
    best_aSb = None
    best_trans_error = float("inf")
    best_rot_error = float("inf")

    valid_idxs = [i for i, bTi in enumerate(bTi_list_est) if bTi is not None]
    num_to_delete = math.ceil(delete_frac * len(valid_idxs))
    if len(valid_idxs) - num_to_delete < 2:
        # run without RANSAC! want at least 2 frame pairs for good alignment.
        aligned_bTi_list_est, aSb = align_poses_sim3_ignore_missing(
            aTi_list_ref, bTi_list_est
        )
        return aligned_bTi_list_est, aSb

    # randomly delete some elements
    for _ in range(num_iters):

        aTi_list_ref_subset = copy.deepcopy(aTi_list_ref)
        bTi_list_est_subset = copy.deepcopy(bTi_list_est)

        # randomly delete 33% of the poses
        delete_idxs = np.random.choice(a=valid_idxs, size=num_to_delete, replace=False)
        for del_idx in delete_idxs:
            bTi_list_est_subset[del_idx] = None

        aligned_bTi_list_est, aSb = align_poses_sim3_ignore_missing(
            aTi_list_ref_subset, bTi_list_est_subset
        )

        # evaluate inliers.
        rot_error, trans_error, _, _ = compute_pose_errors_3d(aTi_list_ref, aligned_bTi_list_est)
        logger.debug("Deleted " + str(delete_idxs) + f" -> trans_error {trans_error:.1f}, rot_error {rot_error:.1f}")

        if trans_error <= best_trans_error and rot_error <= best_rot_error:
            logger.debug(f"\tFound better trans error {trans_error:.2f} < {best_trans_error:.2f}")
            logger.debug(f"\tFound better rot error {rot_error:.2f} < {best_rot_error:.2f}")
            best_aSb = aSb
            best_trans_error = trans_error
            best_rot_error = rot_error

    # now go back and transform the full, original list (not just a subset).
    best_aligned_bTi_list_est_full = [None] * len(bTi_list_est)
    for i, bTi_ in enumerate(bTi_list_est):
        if bTi_ is None:
            continue
        best_aligned_bTi_list_est_full[i] = best_aSb.transformFrom(bTi_)
    return best_aligned_bTi_list_est_full, best_aSb


def compute_pose_errors_3d(
    aTi_list_gt: List[Pose3], aligned_bTi_list_est: List[Optional[Pose3]]
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute average pose errors over all cameras (separately in rotation and translation).

    Note: pose graphs must already be aligned.

    Args:
        aTi_list_gt: ground truth 3d pose graph.
        aligned_bTi_list_est: estimated pose graph aligned to the ground truth's "a" frame.

    Returns:
        mean_rot_err: average rotation error per camera, measured in degrees.
        mean_trans_err: average translation error per camera.
        rot_errors: array of (K,) rotation errors, measured in degrees.
        trans_errors: array of (K,) translation errors.
    """
    logger.debug("aTi_list_gt: " + str(aTi_list_gt))
    logger.debug("aligned_bTi_list_est" + str(aligned_bTi_list_est))

    rotation_errors = []
    translation_errors = []
    for (aTi, aTi_) in zip(aTi_list_gt, aligned_bTi_list_est):
        if aTi is None or aTi_ is None:
            continue
        rot_err = compute_relative_rotation_angle(aTi.rotation(), aTi_.rotation())
        trans_err = np.linalg.norm(aTi.translation() - aTi_.translation())

        rotation_errors.append(rot_err)
        translation_errors.append(trans_err)

    rotation_errors = np.array(rotation_errors)
    translation_errors = np.array(translation_errors)

    logger.debug("Rotation Errors: " + str(np.round(rotation_errors, 1)))
    logger.debug("Translation Errors: " + str(np.round(translation_errors, 1)))
    mean_rot_err = np.mean(rotation_errors)
    mean_trans_err = np.mean(translation_errors)
    return mean_rot_err, mean_trans_err, rotation_errors, translation_errors


def align_poses_sim3_ignore_missing(
    aTi_list: List[Optional[Pose3]], bTi_list: List[Optional[Pose3]]
) -> Tuple[List[Optional[Pose3]], Similarity3]:
    """Align by similarity transformation, but allow missing estimated poses in the input.

    Note: this is a wrapper for align_poses_sim3() that allows for missing poses/dropped cameras.
    This is necessary, as align_poses_sim3() requires a valid pose for every input pair.

    We force SIM(3) alignment rather than SE(3) alignment.
    We assume the two trajectories are of the exact same length.

    Args:
        aTi_list: reference poses in frame "a" which are the targets for alignment
        bTi_list: input poses which need to be aligned to frame "a"

    Returns:
        aTi_list_: transformed input poses previously "bTi_list" but now which
            have the same origin and scale as reference (now living in "a" frame)
        aSb: Similarity(3) object that aligns the two pose graphs.
    """
    assert len(aTi_list) == len(bTi_list)

    # only choose target poses for which there is a corresponding estimated pose
    corresponding_aTi_list = []
    valid_camera_idxs = []
    valid_bTi_list = []
    for i, bTi in enumerate(bTi_list):
        if bTi is not None:
            valid_camera_idxs.append(i)
            valid_bTi_list.append(bTi)
            corresponding_aTi_list.append(aTi_list[i])

    valid_aTi_list_, aSb = align_poses_sim3(aTi_list=corresponding_aTi_list, bTi_list=valid_bTi_list)

    num_cameras = len(aTi_list)
    # now at valid indices
    aTi_list_ = [None] * num_cameras
    for i in range(num_cameras):
        if i in valid_camera_idxs:
            aTi_list_[i] = valid_aTi_list_.pop(0)

    return aTi_list_, aSb


def align_poses_sim3(aTi_list: List[Pose3], bTi_list: List[Pose3]) -> Tuple[List[Pose3], Similarity3]:
    """Align two pose graphs via similarity transformation. Note: poses cannot be missing/invalid.

    We force SIM(3) alignment rather than SE(3) alignment.
    We assume the two trajectories are of the exact same length.

    Args:
        aTi_list: reference poses in frame "a" which are the targets for alignment
        bTi_list: input poses which need to be aligned to frame "a"

    Returns:
        aTi_list_: transformed input poses previously "bTi_list" but now which
            have the same origin and scale as reference (now living in "a" frame)
        aSb: Similarity(3) object that aligns the two pose graphs.
    """
    n_to_align = len(aTi_list)
    assert len(aTi_list) == len(bTi_list)
    assert n_to_align >= 2, "SIM(3) alignment uses at least 2 frames"

    ab_pairs = Pose3Pairs(list(zip(aTi_list, bTi_list)))

    aSb = Similarity3.Align(ab_pairs)

    if np.isnan(aSb.scale()) or aSb.scale() == 0:
        # we have run into a case where points have no translation between them (i.e. panorama).
        # We will first align the rotations and then align the translation by using centroids.
        # TODO: handle it in GTSAM

        # align the rotations first, so that we can find the translation between the two panoramas
        aSb = Similarity3(aSb.rotation(), np.zeros((3,)), 1.0)
        aTi_list_rot_aligned = [aSb.transformFrom(bTi) for bTi in bTi_list]

        # fit a single translation motion to the centroid
        aTi_centroid = np.array([aTi.translation() for aTi in aTi_list]).mean(axis=0)
        aTi_rot_aligned_centroid = np.array([aTi.translation() for aTi in aTi_list_rot_aligned]).mean(axis=0)

        # construct the final SIM3 transform
        aSb = Similarity3(aSb.rotation(), aTi_centroid - aTi_rot_aligned_centroid, 1.0)

    # provide a summary of the estimated alignment transform
    aRb = aSb.rotation().matrix()
    atb = aSb.translation()
    rz, ry, rx = Rotation.from_matrix(aRb).as_euler("zyx", degrees=True)
    logger.info("Sim(3) Rotation `aRb`: rz=%.2f deg., ry=%.2f deg., rx=%.2f deg.", rz, ry, rx)
    logger.info(f"Sim(3) Translation `atb`: [tx,ty,tz]={str(np.round(atb,2))}")
    logger.info("Sim(3) Scale `asb`: %.2f", float(aSb.scale()))

    aTi_list_ = []
    for i in range(n_to_align):
        bTi = bTi_list[i]

        aTi_list_.append(aSb.transformFrom(bTi))

    logger.info("Pose graph Sim(3) alignment complete.")

    return aTi_list_, aSb


def compare_rotations(
    aRi_list: List[Optional[Rot3]], bRi_list: List[Optional[Rot3]], angular_error_threshold_degrees: float
) -> bool:
    """Helper function to compare two lists of global Rot3, after aligning them.

    Notes:
    1. The input lists have the rotations in the same order, and can contain None entries.

    Args:
        aTi_list: 1st list of rotations.
        bTi_list: 2nd list of rotations.
        angular_error_threshold_degrees: the threshold for angular error between two rotations.
    Returns:
        Result of the comparison.
    """
    if len(aRi_list) != len(bRi_list):
        return False

    # check the presense of valid Rot3 objects in the same location
    aRi_valid = [i for (i, aRi) in enumerate(aRi_list) if aRi is not None]
    bRi_valid = [i for (i, bRi) in enumerate(bRi_list) if bRi is not None]
    if aRi_valid != bRi_valid:
        return False

    if len(aRi_valid) <= 1:
        # we need >= two entries going forward for meaningful comparisons
        return False

    aRi_list = [aRi_list[i] for i in aRi_valid]
    bRi_list = [bRi_list[i] for i in bRi_valid]

    # frame 'a' is the target/reference, and bRi_list will be transformed
    aRi_list_ = align_rotations(aRi_list, bRi_list)

    return all(
        [
            compute_relative_rotation_angle(aRi, aRi_) < angular_error_threshold_degrees
            for (aRi, aRi_) in zip(aRi_list, aRi_list_)
        ]
    )


def compare_global_poses(
    aTi_list: List[Optional[Pose3]],
    bTi_list: List[Optional[Pose3]],
    rot_angular_error_thresh_degrees: float = 2,
    trans_err_atol: float = 1e-2,
    trans_err_rtol: float = 1e-1,
    verbose: bool = True,
) -> bool:
    """Helper function to compare two lists of Point3s using L2 distances at each index.

    Notes:
    1. The input lists have the poses in the same order, and can contain None entries.
    2. To resolve global origin ambiguity, we fit a Sim(3) transformation and align the two pose graphs.

    Args:
        aTi_list: 1st list of poses.
        bTi_list: 2nd list of poses.
        rot_angular_error_threshold_degrees (optional): angular error threshold for rotations. Defaults to 2.
        trans_err_atol (optional): absolute error threshold for translation. Defaults to 1e-2.
        trans_err_rtol (optional): relative error threshold for translation. Defaults to 1e-1.

    Returns:
        result of the comparison.
    """

    # check the length of the input lists
    if len(aTi_list) != len(bTi_list):
        return False

    # check the presense of valid Pose3 objects in the same location
    aTi_valid = [i for (i, aTi) in enumerate(aTi_list) if aTi is not None]
    bTi_valid = [i for (i, bTi) in enumerate(bTi_list) if bTi is not None]
    if aTi_valid != bTi_valid:
        return False

    if len(aTi_valid) <= 1:
        # we need >= two entries going forward for meaningful comparisons
        return False

    # align the remaining poses
    aTi_list = [aTi_list[i] for i in aTi_valid]
    bTi_list = [bTi_list[i] for i in bTi_valid]

    #  We set frame "a" the target/reference
    aTi_list_, _ = align_poses_sim3(aTi_list, bTi_list)

    rotations_equal = all(
        [
            compute_relative_rotation_angle(aTi.rotation(), aTi_.rotation()) < rot_angular_error_thresh_degrees
            for (aTi, aTi_) in zip(aTi_list, aTi_list_)
        ]
    )
    translations_equal = all(
        [
            np.allclose(aTi.translation(), aTi_.translation(), atol=trans_err_atol, rtol=trans_err_rtol)
            for (aTi, aTi_) in zip(aTi_list, aTi_list_)
        ]
    )
    if verbose:
        rotation_errors = np.array(
            [
                compute_relative_rotation_angle(aTi.rotation(), aTi_.rotation())
                for (aTi, aTi_) in zip(aTi_list, aTi_list_)
            ]
        )
        translation_errors = np.array(
            [np.linalg.norm(aTi.translation() - aTi_.translation()) for (aTi, aTi_) in zip(aTi_list, aTi_list_)]
        )
        logger.info("Comparison Rotation Errors (degrees): " + str(np.round(rotation_errors, 2)))
        logger.info("Comparison Translation Errors: " + str(np.round(translation_errors, 2)))

    return rotations_equal and translations_equal


def compute_relative_rotation_angle(R_1: Optional[Rot3], R_2: Optional[Rot3]) -> Optional[float]:
    """Compute the angle between two rotations.

    Note: the angle is the norm of the angle-axis representation.

    Args:
        R_1: the first rotation.
        R_2: the second rotation.

    Returns:
        the angle between two rotations, in degrees
    """

    if R_1 is None or R_2 is None:
        return None

    relative_rot = R_1.between(R_2)
    # TODO(johnwlambert): we are using scipy.spatial.transform as a hotfix until GTSAM axisAngle() is patched.
    # See https://github.com/borglab/gtsam/issues/886
    scaled_axis = Rotation.from_matrix(relative_rot.matrix()).as_rotvec()
    relative_rot_angle_rad = np.linalg.norm(scaled_axis)
    relative_rot_angle_deg = np.rad2deg(relative_rot_angle_rad)
    return relative_rot_angle_deg


def compute_relative_unit_translation_angle(U_1: Optional[Unit3], U_2: Optional[Unit3]) -> Optional[float]:
    """Compute the angle between two unit-translations.

    Args:
        U_1: the first unit-translation.
        U_2: the second unit-translation.

    Returns:
        the angle between the two unit-vectors, in degrees
    """
    if U_1 is None or U_2 is None:
        return None

    # TODO: expose Unit3's dot function and use it directly
    dot_product = np.dot(U_1.point3(), U_2.point3())
    dot_product = np.clip(dot_product, -1, 1)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg


def compute_translation_to_direction_angle(
    i2Ui1: Optional[Unit3], wTi2: Optional[Pose3], wTi1: Optional[Pose3]
) -> Optional[float]:
    """Compute angle between a unit translation and the relative translation between 2 poses.

    Given a unit translation measurement from i2 to i1, the estimated poses of
    i1 and i2, returns the angle between the relative position of i1 wrt i2
    and the unit translation measurement.

    Args:
        i2Ui1: Unit translation measurement.
        wTi2: Pose of camera i2.
        wTi1: Pose of camera i1.

    Returns:
        Angle between measurement and relative estimated translation in degrees.
    """
    if i2Ui1 is None or wTi2 is None or wTi1 is None:
        return None

    i2Ti1 = wTi2.between(wTi1)
    i2Ui1_estimated = Unit3(i2Ti1.translation())
    return compute_relative_unit_translation_angle(i2Ui1, i2Ui1_estimated)


def compute_points_distance_l2(wti1: Optional[Point3], wti2: Optional[Point3]) -> Optional[float]:
    """Computes the L2 distance between the two input 3D points.

    Assumes the points are in the same coordinate frame. Returns None if either
    point is None.

    Args:
        wti1: Point1 in world frame
        wti2: Point2 in world frame

    Returns:
        L2 norm of wti1 - wti2
    """
    if wti1 is None or wti2 is None:
        return None
    return np.linalg.norm(wti1 - wti2)


def compute_cyclic_rotation_error(i1Ri0: Rot3, i2Ri1: Rot3, i2Ri0: Rot3) -> float:
    """Computes the cycle error in degrees after composing the three input rotations.

    The cyclic error is the angle between identity and the rotation obtained by composing the three input relative 
    rotations, i.e., (I - inv(i2Ri0) * i2Ri1 * i1Ri0).

    Args:
        i1Ri0: Relative rotation of camera i0 with respect to i1.
        i2Ri1: Relative rotation of camera i1 with respect to i2.
        i2Ri0: Relative rotation of camera i0 with respect to i2.

    Returns:
        Cyclic rotation error in degrees.
    """
    i0Ri0_from_cycle = i2Ri0.inverse().compose(i2Ri1).compose(i1Ri0)
    return compute_relative_rotation_angle(Rot3(), i0Ri0_from_cycle)


def get_points_within_radius_of_cameras(
    wTi_list: List[Pose3], points_3d: np.ndarray, radius: float = 50
) -> Optional[np.ndarray]:
    """Return those 3d points that fall within a specified radius from any camera.

    Args:
        wTi_list: camera poses
        points_3d: array of shape (N,3) representing 3d points
        radius: distance threshold, in meters

    Returns:
        nearby_points_3d: array of shape (M,3), where M <= N
    """
    if len(wTi_list) == 0 or points_3d.size == 0 or radius <= 0:
        return None

    num_points = points_3d.shape[0]
    num_poses = len(wTi_list)
    # each row represents attributes for a single point
    # each column represents
    is_nearby_matrix = np.zeros((num_points, num_poses), dtype=bool)
    for j, wTi in enumerate(wTi_list):
        is_nearby_matrix[:, j] = np.linalg.norm(points_3d - wTi.translation(), axis=1) < radius

    is_nearby_to_any_cam = np.any(is_nearby_matrix, axis=1)
    nearby_points_3d = points_3d[is_nearby_to_any_cam]
    return nearby_points_3d

import abc
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple, Optional

import dask
import numpy as np
import cv2

import gtsam
from data_association.tracks import FeatureTracks

class DataAssociation(FeatureTracks):
    """
    workflow: form feature tracks; for each track, call LandmarkInitialization
    """
    def __init__(self, matches, num_poses, global_poses, calibrationFlag, calibration, camera_list) -> None:
        print("received matches", matches)
        self.calibrationFlag = calibrationFlag
        self.calibration = calibration
        super().__init__(matches, num_poses, global_poses)
        filtered_map = self.filtered_landmark_map
        self.triangulated_landmark = gtsam.Point3
        # filtered_map = filtered_landmark_map
        

        for landmark_key, feature_track in filtered_map.items():
            if self.calibrationFlag == True:
                LMI = LandmarkInitialization(calibrationFlag, feature_track, calibration,global_poses)
            else:
                LMI = LandmarkInitialization(calibrationFlag, feature_track, camera_list)
            self.triangulated_landmark = LMI.triangulate_landmark(feature_track)
            # Replace landmark_key with triangulated landmark
        

class LandmarkInitialization(metaclass=abc.ABCMeta):
    """
    Class to initialize landmark points via triangulation
    """

    def __init__(self, 
    calibrationFlag: bool,
    obs_list: List,
    calibration: Optional[gtsam.Cal3_S2] = None, 
    track_poses: Optional[List[gtsam.Pose3]] = None, 
    track_cameras: Optional[List[gtsam.Cal3_S2]] = None) -> None:
        """
        Args:
            calibrationFlag: check if shared calibration exists(True) or each camera has individual calibration(False)
            obs_list: Feature track of type [(img_idx, img_measurement),..]
            calibration: Shared calibration
            track_poses: List of poses in a feature track
            track_cameras: List of cameras in a feature track
        """
        self.sharedCal_Flag = calibrationFlag
        self.observation_list = obs_list
        self.calibration = calibration
        # for shared calibration
        if track_poses is not None:
            self.track_pose_list = track_poses
        # for multiple cameras with individual calibrations
        if track_cameras is not None:
            self.track_camera_list = track_cameras
    
    
    def extract_end_measurements(self, track) -> Tuple[gtsam.Pose3Vector, List, gtsam.Point2Vector]:
        """
        Extract first and last measurements in a track for triangulation.
        Args:
            track: feature track from which measurements are to be extracted
        Returns:
            pose_estimates: Poses of first and last measurements in track
            camera_list: Individual camera calibrations for first and last measurement
            img_measurements: Observations corresponding to first and last measurements
        """
        pose_estimates_track = gtsam.Pose3Vector()
        pose_estimates = gtsam.Pose3Vector()
        cameras_list_track = []
        cameras_list = []
        img_measurements_track = gtsam.Point2Vector()
        img_measurements = gtsam.Point2Vector()
        for (img_idx, img_Pt) in track:
            if self.sharedCal_Flag:
                pose_estimates_track.append(self.track_pose_list[img_idx])
            else:
                cameras_list_track.append(self.track_camera_list[img_idx]) 
            img_measurements_track.append(img_Pt)
            if pose_estimates_track:
                pose_estimates.append(pose_estimates_track[0]) 
                pose_estimates.append(pose_estimates_track[-1])
            else:
                cameras_list = [cameras_list_track[0], cameras_list_track[-1]]
            img_measurements.append(img_measurements_track[0])
            img_measurements.append(img_measurements_track[-1])
        
        return pose_estimates, cameras_list, img_measurements


    def triangulate_landmark(self, track) -> gtsam.Point3:
        """
        Args:
            track: List of (img_idx, observations(Point2))
        Returns:
            triangulated_landmark: triangulated landmark
        """
        pose_estimates, camera_values, img_measurements = self.extract_end_measurements(track)
        optimize = False
        rank_tol = 1e-9
        # for (img_idx, img_measurements) in track:
            # if shared calibration provided for all cameras
        if self.sharedCal_Flag:
            if self.track_pose_list == None or not pose_estimates:
                raise Exception('track_poses arg or pose estimates missing')
            return gtsam.triangulatePoint3(pose_estimates, self.calibration, img_measurements, rank_tol, optimize)
        else:
            if self.track_camera_list == None or not camera_values:
                raise Exception('track_cameras arg or camera values missing')
            return gtsam.triangulatePoint3(camera_values, img_measurements, rank_tol, optimize)
    

"""Factor-graph based formulation of Bundle adjustment and optimization.

Authors: Xiaolong Wu, John Lambert, Ayush Baid
"""

import logging
import sys

import dask
import gtsam
from dask.delayed import Delayed
from gtsam import GeneralSFMFactor2Cal3Bundler, SfmData, symbol_shorthand

from gtsfm.common.sfm_result import SfmResult

# TODO: any way this goes away?
C = symbol_shorthand.C  # camera
P = symbol_shorthand.P  # 3d point
X = symbol_shorthand.X  # camera pose
K = symbol_shorthand.K  # calibration


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class BundleAdjustmentOptimizer:
    """Bundle adjustment using factor-graphs in GTSAM.

    This class refines global pose estimates and intrinsics of cameras, and also refines 3D point cloud structure given tracks from triangulation."""

    def __init__(self, shared_calib: bool = False):
        """Initializes the parameters for bundle adjustment module.

        Args:
            shared_calib (optional): Flag to enable shared calibration across
                                     all cameras. Defaults to False.
        """

        self._shared_calib = shared_calib

    def run(self, initial_data: SfmData) -> SfmResult:
        """Run the bundle adjustment by forming factor graph and optimizing using Levenberg–Marquardt optimization.

        Args:
            initial_data: initialized cameras, tracks w/ their 3d landmark from triangulation.
        Results:
            optimized camera poses, 3D point w/ tracks, and error metrics.
        """
        logging.info(
            f"Input: {initial_data.number_tracks()} tracks on {initial_data.number_cameras()} cameras\n"
        )

        # Create a factor graph
        graph = gtsam.NonlinearFactorGraph()

        # We share *one* noiseModel between all projection factors
        # TODO: move params to init.
        noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v

        # Add measurements to the factor graph
        j = 0
        for t_idx in range(initial_data.number_tracks()):
            track = initial_data.track(t_idx)  # SfmTrack
            # retrieve the SfmMeasurement objects
            for m_idx in range(track.number_measurements()):
                # i represents the camera index, and uv is the 2d measurement
                i, uv = track.measurement(m_idx)

                graph.add(
                    GeneralSFMFactor2Cal3Bundler(
                        uv, noise, X(i), P(j), K(0 if self._shared_calib else i)
                    )
                )
            j += 1

        # Add a prior on camera x0's pose. This indirectly specifies where the origin is.
        graph.push_back(
            gtsam.PriorFactorPose3(
                X(0),
                initial_data.camera(0).pose(),
                gtsam.noiseModel.Isotropic.Sigma(6, 0.1),
            )
        )

        # Add prior on camera x0's intrinsics.
        graph.push_back(
            gtsam.PriorFactorCal3Bundler(
                K(0),
                initial_data.camera(0).calibration(),
                gtsam.noiseModel.Isotropic.Sigma(3, 0.1),
            )
        )

        # Also add a prior on the position of the first landmark to fix the scale
        graph.push_back(
            gtsam.PriorFactorPoint3(
                P(0),
                initial_data.track(0).point3(),
                gtsam.noiseModel.Isotropic.Sigma(3, 0.1),
            )
        )

        # Create initial estimate
        initial = gtsam.Values()

        i = 0
        # add each PinholeCameraCal3Bundler
        for cam_idx in range(initial_data.number_cameras()):
            camera = initial_data.camera(cam_idx)
            initial.insert(X(i), camera.pose())
            i += 1

        if self._shared_calib:
            camera = initial_data.camera(0)
            initial.insert(K(0), camera.calibration())

        else:
            i = 0
            # add each PinholeCameraCal3Bundler
            for cam_idx in range(initial_data.number_cameras()):
                camera = initial_data.camera(cam_idx)
                initial.insert(K(i), camera.calibration())
                i += 1

        j = 0
        # add each SfmTrack
        for t_idx in range(initial_data.number_tracks()):
            track = initial_data.track(t_idx)
            initial.insert(P(j), track.point3())
            j += 1

        # Optimize the graph and print results
        try:
            params = gtsam.LevenbergMarquardtParams()
            params.setVerbosityLM("ERROR")
            lm = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
            result = lm.optimize()
        except Exception as e:
            logging.exception("LM Optimization failed")
            return

        # Error drops from ~2764.22 to ~0.046
        logging.info(f"initial error: {graph.error(initial):.2f}")
        logging.info(f"final error: {graph.error(result):.2f}")

        # construct the results
        sfm_result = SfmResult(
            initial_data, result, graph.error(result), self._shared_calib
        )

        return sfm_result

    def create_computation_graph(self, sfm_data_graph: Delayed) -> Delayed:
        """Create the computation graph for performing bundle adjustment.

        Args:
            sfm_data_graph: an SfmData object wrapped up using dask.delayed
        Results:
            SfmResult wrapped up using dask.delayed
        """
        return dask.delayed(self.run)(sfm_data_graph)

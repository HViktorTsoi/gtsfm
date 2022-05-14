"""Translation averaging using 1DSFM for a camera rig.

This file creates a class that extends the 1DSfM module to work with camera rigs (currently Hilti).

References:
- https://research.cs.cornell.edu/1dsfm/
- https://github.com/borglab/gtsam/blob/develop/gtsam/sfm/MFAS.h
- https://github.com/borglab/gtsam/blob/develop/gtsam/sfm/TranslationRecovery.h
- https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/TranslationAveragingExample.py

Authors: Akshay Krishnan
"""
import copy
from typing import Dict, List, Optional, Tuple

import gtsam
from gtsam import Rot3, Unit3

import gtsfm.utils.logger as logger_utils
from gtsfm.averaging.translation.averaging_1dsfm import TranslationAveraging1DSFM
from gtsfm.common.pose_prior import PosePrior, PosePriorType

logger = logger_utils.get_logger()

NUM_CAMERAS_IN_RIG = 5
BODY_FRAME_CAMERA = 2
VALID_HARD_CONSTRAINT_EDGES = [(0, 2), (1, 2), (2, 3), (2, 4)]


class RigTranslationAveraging1DSFM(TranslationAveraging1DSFM):
    """A special case of the 1DSFM implementation that pre-processes the relative prior for the Hilti rig."""

    def _filter_priors_and_measurements(relative_measurements, relative_priors: Dict[Tuple[int, int], PosePrior]):
        """Removes all priors that are not between the """
        filtered_measurements = copy.deepcopy(relative_measurements)
        filtered_relative_priors = {}

        def needs_measurement(c1, c2, prior: PosePrior):
            return (prior.type == PosePriorType.HARD_CONSTRAINT and (c1, c2) in VALID_HARD_CONSTRAINT_EDGES) or (prior.type == PosePriorType.SOFT_CONSTRAINT and c1 == BODY_FRAME_CAMERA and c2 == BODY_FRAME_CAMERA)

        for (i1, i2), prior in relative_priors.items():
            c1 = i1 % NUM_CAMERAS_IN_RIG
            c2 = i2 % NUM_CAMERAS_IN_RIG
            if needs_measurement(c1, c2, prior) and (i1, i2) not in relative_measurements:
                filtered_measurements[(i1, i2)] = Unit3(prior.translation())
                filtered_relative_priors[(i1, i2)] = prior
        return filtered_measurements, filtered_relative_priors

    def _get_prior_measurements_in_world_frame(
        self,
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        wRi_list: List[Optional[Rot3]],
    ) -> gtsam.BinaryMeasurementsPoint3:
        """Converts the priors from relative Pose3 priors to relative Point3 priors in world frame.

        If the priors are hard constraints (in the same rig), a hard-coded noise model is used.
        If the priors are soft constraints, the covariance from the PosePrior is used.

        Soft constraints are only added between the 3rd rig cameras.
        Hard constraints are only added between the 3rd camera and other cameras in same rig.

        Args:
            relative_pose_priors: Relative pose priors between cameras, could be a hard or soft prior.
            wRi_list: Absolute rotation estimates from Shonan averaging.

        Returns:
            gtsam.BinaryMeasurementsPoint3 containing Point3 priors in world frame.
        """
        if len(relative_pose_priors) == 0:
            return gtsam.BinaryMeasurementsPoint3()

        def get_prior_in_world_frame(i2, i2Ti1_prior):
            return wRi_list[i2].rotate(i2Ti1_prior.value.translation())

        HARD_CONSTRAINT_NOISE_MODEL = gtsam.noiseModel.Constrained.All(3)

        w_i2ti1_priors = gtsam.BinaryMeasurementsPoint3()
        for (i1, i2), i2Ti1_prior in relative_pose_priors.items():
            c1 = i1 % NUM_CAMERAS_IN_RIG
            c2 = i2 % NUM_CAMERAS_IN_RIG
            if i2Ti1_prior.type == PosePriorType.HARD_CONSTRAINT:
                if (c1, c2) in VALID_HARD_CONSTRAINT_EDGES:
                    w_i2ti1_priors.append(
                        gtsam.BinaryMeasurementPoint3(
                            i2,
                            i1,
                            get_prior_in_world_frame(i2, i2Ti1_prior),
                            HARD_CONSTRAINT_NOISE_MODEL,
                        )
                    )
            else:
                if c1 == BODY_FRAME_CAMERA and c2 == BODY_FRAME_CAMERA:
                    # TODO(akshay-krishnan): Use the translation covariance, transform to world frame.
                    # noise_model = gtsam.noiseModel.Gaussian.Covariance(i2Ti1_prior.covariance)
                    noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 3e-2)
                    w_i2ti1_priors.append(
                        gtsam.BinaryMeasurementPoint3(
                            i2,
                            i1,
                            get_prior_in_world_frame(i2, i2Ti1_prior),
                            noise_model,
                        )
                    )
        logger.info("Added {} priors in rig translation averaging".format(len(w_i2ti1_priors)))
        return w_i2ti1_priors

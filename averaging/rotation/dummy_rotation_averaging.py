"""A dummy rotation averaging class which is used for testing.

Authors: Ayush Baid
"""

from typing import Dict, List, Tuple

import numpy as np
from averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsam import Rot3


class DummyRotationAveraging(RotationAveragingBase):
    """Assigns random rotation matrices to each pose."""

    def run(self,
            num_poses: int,
            relative_rotations: Dict[Tuple[int, int], Rot3]
            ) -> List[Rot3]:
        """Run the rotation averaging.

        Args:
            num_poses: number of poses.
            relative_rotations: relative rotations between camera poses.

        Returns:
            List[Rot3]: global rotations for each camera pose.
        """

        # create the random seed using relative rotations
        seed_rotation = Rot3()
        for rotation in relative_rotations.values():
            seed_rotation = seed_rotation.compose(rotation)

        np.random.seed(
            int(1000*np.sum(seed_rotation.xyz(), axis=None) % (2 ^ 32))
        )

        # generate dummy rotations
        results = []
        for _ in range(num_poses):
            random_vector = np.random.rand(3)*2*np.pi
            results.append(Rot3.Rodrigues(
                random_vector[0], random_vector[1], random_vector[2]))

        return results

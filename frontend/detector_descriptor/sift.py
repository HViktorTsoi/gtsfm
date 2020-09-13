"""SIFT Detector-Descriptor implementation.

The method was proposed in 'Distinctive Image Features from Scale-Invariant
Keypoints' and is implemented by wrapping over OpenCV's API

References:
- https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
- https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

Authors: Ayush Baid
"""
from typing import Tuple

import cv2 as cv
import numpy as np

import utils.features as feature_utils
import utils.images as image_utils
from common.image import Image
from frontend.detector_descriptor.detector_descriptor_base import \
    DetectorDescriptorBase


class SIFTDetectorDescriptor(DetectorDescriptorBase):
    """SIFT detector-descriptor using OpenCV's implementation."""

    def detect_and_describe(self,
                            image: Image) -> Tuple[np.ndarray, np.ndarray]:
        """Perform feature detection as well as their description in a single
        step.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for
        details about the output format.

        Args:
            image (Image): the input image

        Returns:
            Tuple[np.ndarray, np.ndarray]: detected features and their
                                           descriptions as two numpy arrays
        """

        # conert to grayscale
        gray_image = image_utils.rgb_to_gray_cv(image.image_array)

        # Creating OpenCV object
        opencv_obj = cv.xfeatures2d.SIFT_create()

        # Run the opencv code
        cv_keypoints, descriptors = opencv_obj.detectAndCompute(
            gray_image, None)

        # convert keypoints to features
        features = feature_utils.convert_to_numpy_array(cv_keypoints)

        # sort the features and descriptors by the score
        sort_idx = np.argsort(-features[:, 3])
        features = features[sort_idx]
        descriptors = descriptors[sort_idx]

        return features, descriptors

"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from .postprocessor import Postprocessor
from ..representation import BrainTumorSegmentationPrediction, BrainTumorSegmentationAnnotation


class SegmentationPredictionToGT(Postprocessor):
    __provider__ = "segmentation_prediction_to_gt"

    prediction_types = (BrainTumorSegmentationPrediction, )
    annotation_types = (BrainTumorSegmentationAnnotation, )

    def process_image(self, annotation, prediction):
        if not len(annotation) == len(prediction) == 1:
            raise RuntimeError('Postprocessor {} does not support multiple annotation and/or prediction.'
                               .format(self.__provider__))

        annotation_ = annotation[0]
        prediction_ = prediction[0]

        annotation_shape = annotation_.mask.shape
        prediction_shape = prediction_.mask.shape

        target_size = (128,128,128)
        annotation_.mask = annotation_.mask[0:target_size[0], 0:target_size[1], 0:target_size[2]]
        #correct labels from BRATS_2017 dataset
        ed = annotation_.mask == 1
        ncr = annotation_.mask == 2
        et = annotation_.mask == 3
        annotation_.mask[ncr] = 1
        annotation_.mask[ed] = 2

        prediction_.mask = (prediction_.mask > 0.5).astype(bool)
      
        wt = prediction_.mask[0]
        tc = prediction_.mask[1]
        et = prediction_.mask[2]

        pp = np.zeros(annotation_.mask.shape)
        pp[wt] = 2
        pp[tc] = 1
        pp[et] = 3
        
        prediction[0].mask = pp

        return annotation, prediction

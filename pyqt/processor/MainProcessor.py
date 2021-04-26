# Copyright © 2020, Yingping Liang. All Rights Reserved.

# Copyright Notice
# Yingping Liang copyrights this specification.
# No part of this specification may be reproduced in any form or means,
# without the prior written consent of Yingping Liang.


# Disclaimer
# This specification is preliminary and is subject to change at any time without notice.
# Yingping Liang assumes no responsibility for any errors contained herein.

import imutils


class VideoPlayer(object):

    def __init__(self):
        self.firstFrame = None

    def feedCap(self, frame):

        self.frame = imutils.resize(frame, height=300)

        pack = {'frame': frame, 'faces': [], 'face_bboxes': []}

        return pack


class MainProcessor(object):

    def __init__(self, model_type='openvino', tracker_type='deep_sort'):
        from .AIDetector_pytorch import Detector as FaceTracker
        self.processor = FaceTracker(tracker_type)
        # self.processor = VideoPlayer()
        self.face_id = 0

    def getProcessedImage(self, frame, isChecked, region_bbox2draw=None):
        dicti = self.processor.feedCap(frame, isChecked, region_bbox2draw)
        return dicti

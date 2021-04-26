from .tracker_deep import update_tracker
import cv2


class baseDet(object):

    def __init__(self, tracker_type):

        self.img_size = 640
        self.threshold = 0.2
        self.max_frame = 160
        self.stride = 2
        self.illegal_num = 0
        self.tracker_type = tracker_type

    def build_config(self):

        self.faceTracker = {}
        self.illegals = []
        self.longstay=[]
        self.faceClasses = {}
        self.faceregister = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def feedCap(self, im, isChecked, region_bbox2draw=None):

        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': [],
            'num':None,
            'time':None,
            'danger':None,
            'long':None
        }
        self.frameCounter += 1

        im, faces, face_bboxes,id_num,use_time,danger_num,long_num = update_tracker(
            self, im, region_bbox2draw, isChecked)

        retDict['frame'] = im
        retDict['faces'] = faces
        retDict['face_bboxes'] = face_bboxes
        retDict['num'] = id_num
        retDict['time'] = use_time
        retDict['danger'] = danger_num
        retDict['long'] = long_num
        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")

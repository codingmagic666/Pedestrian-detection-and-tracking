from numpy.core.records import record
from deep_sort.deep_sort.sort import track
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import time

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


id_num = 0
use_time=0

def is_poi_in_poly(pt, poly):
    """
    判断点是否在多边形内部的 pnpoly 算法
    :param pt: 点坐标 [x,y]
    :param poly: 点多边形坐标 [[x1,y1],[x2,y2],...]
    :return: 点是否在多边形之内
    """
    nvert = len(poly)
    vertx = []
    verty = []
    testx = pt[0]
    testy = pt[1]
    for item in poly:
        vertx.append(item[0])
        verty.append(item[1])

    j = nvert - 1
    res = False
    for i in range(nvert):
        if (verty[j] - verty[i]) == 0:
            j = i
            continue
        x = (vertx[j] - vertx[i]) * (testy - verty[i]) / \
            (verty[j] - verty[i]) + vertx[i]
        if ((verty[i] > testy) != (verty[j] > testy)) and (testx < x):
            res = not res
        j = i
    return res


def cluster_bboxes(image, bboxes, line_thickness):
    total_num = len(bboxes)
    recorded = []
    tf = max(line_thickness - 1, 1)
    for i in range(total_num):
        cluster = 0
        x1, y1, x2, y2, _, _ = bboxes[i]
        xc = int((x1+x2)/2)
        yc = int((y1+y2)/2)
        for j in range(i+1, total_num):
            x3, y3, x4, y4, _, _ = bboxes[j]
            xc_ = int((x3+x4)/2)
            yc_ = int((y3+y4)/2)
            if abs(xc - xc_) < (x2 - x1 + x4 - x3) / 1.5:
                if abs(yc - yc_) < (y2 - y1 + y4 - y3) / 3:
                    #cv2.line(image, (xc, yc), (xc_, yc_),
                             #(100, 80, 240), thickness=line_thickness)
                    cluster += 1
                    recorded.append(j)

        if cluster and not i in recorded:
            recorded.append(i)
            if cluster >= 5:
                color = [0, 0, 255]
            else:
                color = [200, 10, 100]
            #cv2.putText(image, 'Clusrers: {}'.format(cluster+1), (x1, y1 - 30), 0, line_thickness / 3,
                        #color, thickness=tf, lineType=cv2.LINE_AA)


def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = (0, 0, 255)
        if cls_id == 'person':
            color = (0, 255, 0)
        elif cls_id == 'Long Stay':
            color = (255, 0, 255)
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        #cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        if cls_id == 'person':
            cv2.putText(image, '{}'.format(pos_id), (c1[0], c1[1] - 2), 0, tl / 3,[255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        elif cls_id == 'Long Stay':
            cv2.putText(image, '{}{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,[255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        else:
            cv2.putText(image, '{}{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,[0, 0, 255], thickness=tf, lineType=cv2.LINE_AA) 
    cluster_bboxes(image, bboxes, tl)
    return image


def update_tracker(target_detector, image, region_bbox2draw, isChecked):
    time1=time.time()
    new_faces = []
    _, bboxes = target_detector.detect(image)

    bbox_xywh = []
    confs = []
    clss = []
    bboxes2draw = []
    face_bboxes = []
    idforlongstay=[] 
    idfordanger=[] 
    danger_num = 0
    long_num = 0
    for x1, y1, x2, y2, cls_id, conf in bboxes:

        obj = [
            int((x1+x2)/2), int((y1+y2)/2),
            x2-x1, y2-y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    if len(bbox_xywh):
        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, clss, image)

        current_ids = []
        for value in list(outputs):
            x1, y1, x2, y2, cls_, track_id = value
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            flag1 = 0
            flag2 = 0


            if track_id in target_detector.faceTracker:
                if target_detector.faceTracker[track_id] > target_detector.max_frame:
                    cls_ = 'Long Stay'
                    flag1 = 2;
                    long_num += 1

            
            if flag1 is 2:
                if track_id in target_detector.faceTracker:
                    if not track_id in target_detector.longstay:
                        #target_detector.faceTracker.pop(track_id)
                        #target_detector.illegals.append(track_id)
                        idforlongstay.append(track_id)
                        target_detector.longstay.append(track_id)

            if not isChecked['region']:
                if is_poi_in_poly([xc, yc], region_bbox2draw[0]):
                    cls_ = 'Danger Region'
                    flag2 = 1
                    danger_num += 1

            if flag2 is 1:
                if track_id in target_detector.faceTracker:
                    if not track_id in target_detector.illegals:
                        target_detector.faceTracker.pop(track_id)
                        target_detector.illegals.append(track_id)
                    

            bboxes2draw.append(
                (x1, y1, x2, y2, cls_, track_id)
            )
            current_ids.append(track_id)
            if not track_id in target_detector.faceTracker:
                target_detector.faceTracker[track_id] = 0
                face = image[y1:y2, x1:x2].copy()
                new_faces.append((face, track_id, cls_))
                if not cls_ == 'person':
                    target_detector.illegal_num += 1

            if track_id in idforlongstay:
                #target_detector.faceTracker[track_id] = 0
                face = image[y1:y2, x1:x2].copy()
                new_faces.append((face, track_id, cls_))

            face_bboxes.append(
                (x1, y1, x2, y2)
            )

        ids2delete = []
        for history_id in target_detector.faceTracker:
            if history_id in current_ids:
                if target_detector.faceTracker[history_id] < 0:
                    target_detector.faceTracker[history_id] = target_detector.faceregister[history_id]
                target_detector.faceTracker[history_id] += 1
            else:
                if target_detector.faceTracker[history_id] > 0:
                    target_detector.faceregister[history_id] = target_detector.faceTracker[history_id]
                    target_detector.faceTracker[history_id] = 0
                target_detector.faceTracker[history_id] -= 1
            if target_detector.faceTracker[history_id] < -5:
                ids2delete.append(history_id)

        for ids in ids2delete:
            target_detector.faceTracker.pop(ids)
            target_detector.faceregister.pop(ids)
            print('-[INFO] Delete track id:', ids)

        image = plot_bboxes(image, bboxes2draw)
    time2=time.time()

    id_num = len(bbox_xywh)
    use_time = 1.0/(time2-time1)
    #cv2.putText(image, '  illegals: {},num:{},time:{}'.format(target_detector.illegal_num,id_num,use_time), (50, 50), 0, 1,
                #[200, 10, 100], thickness=2, lineType=cv2.LINE_AA)
    return image, new_faces, face_bboxes,id_num,use_time,danger_num,long_num



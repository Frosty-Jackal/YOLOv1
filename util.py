import cv2
import numpy as np
import os
from configure import COLOR, GL_CLASSES,iou_thresh,cls_num,bound_confidence

def calculate_iou(bbox1, bbox2):
    if bbox1[2]<=bbox1[0] or bbox1[3]<=bbox1[1] or bbox2[2]<=bbox2[0] or bbox2[3]<=bbox2[1]:
        return 0
    intersect_bbox = [0., 0., 0., 0.] 
    intersect_bbox[0] = max(bbox1[0],bbox2[0])
    intersect_bbox[1] = max(bbox1[1],bbox2[1])
    intersect_bbox[2] = min(bbox1[2],bbox2[2])
    intersect_bbox[3] = min(bbox1[3],bbox2[3])
    w = max(intersect_bbox[2] - intersect_bbox[0], 0)
    h = max(intersect_bbox[3] - intersect_bbox[1], 0)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = w * h  # 交集面积
    iou = area_intersect / (area1 + area2 - area_intersect + 1e-6)  # 防止除0
    return iou

def labels2bbox(matrix):
    matrix = matrix.numpy()
    bboxes = np.zeros((98, 6))
    matrix = matrix.reshape(49,-1)
    bbox = matrix[:, :10].reshape(98, 5)
    r_grid = np.array(list(range(7)))
    r_grid = np.repeat(r_grid, repeats=14, axis=0)  # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ...]
    c_grid = np.array(list(range(7)))
    c_grid = np.repeat(c_grid, repeats=2, axis=0)[np.newaxis, :]
    c_grid = np.repeat(c_grid, repeats=7, axis=0).reshape(-1)  # [0 0 1 1 2 2 3 3 4 4 5 5 6 6 0 0 1 1 2 2 3 3 4 4 5 5 6 6...]
    bboxes[:, 0] = np.maximum((bbox[:, 0] + c_grid) / 7.0 - bbox[:, 2] / 2.0, 0)
    bboxes[:, 1] = np.maximum((bbox[:, 1] + r_grid) / 7.0 - bbox[:, 3] / 2.0, 0)
    bboxes[:, 2] = np.minimum((bbox[:, 0] + c_grid) / 7.0 + bbox[:, 2] / 2.0, 1)
    bboxes[:, 3] = np.minimum((bbox[:, 1] + r_grid) / 7.0 + bbox[:, 3] / 2.0, 1)
    bboxes[:, 4] = bbox[:, 4]
    cls = np.argmax(matrix[:, 10:], axis=1)
    cls = np.repeat(cls, repeats=2, axis=0)
    bboxes[:, 5] = cls
    keepid = nms_multi_cls(bboxes, thresh=iou_thresh, n_cls=cls_num)
    ids = []
    for x in keepid:
        ids = ids + list(x)
    ids = sorted(ids)
    return bboxes[ids, :]

def nms_multi_cls(dets, thresh, n_cls):
    keeps_index = []
    for i in range(n_cls):
        order_i = np.where(dets[:,5]==i)[0]
        det = dets[dets[:, 5] == i, 0:5]
        if det.shape[0] == 0:
            keeps_index.append([])
            continue
        keep = nms_1cls(det, thresh)
        keeps_index.append(order_i[keep])
    return keeps_index

def nms_1cls(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= thresh)[0]
        order = order[inds+1]
    return keep

def draw_bbox(img, bbox):
    h, w = img.shape[0:2]
    n = bbox.shape[0]
    for i in range(n):
        confidence = bbox[i, 4]
        if confidence<bound_confidence:
            continue
        p1 = (int(w * bbox[i, 0]), int(h * bbox[i, 1]))
        p2 = (int(w * bbox[i, 2]), int(h * bbox[i, 3]))
        cls_name = GL_CLASSES[int(bbox[i, 5])]
        print(cls_name, p1, p2)
        cv2.rectangle(img, p1, p2, COLOR[int(bbox[i, 5])])
        cv2.putText(img, cls_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(img, str(confidence), (p1[0],p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.imshow("bbox", img)
    cv2.waitKey(0)

def draw_bbox_app(bbox,filename):
    filepath = os.path.join('uploads', filename)
    img=cv2.imread(filepath)
    h, w = img.shape[0:2]
    n = bbox.shape[0]
    for i in range(n):
        confidence = bbox[i, 4]
        if confidence<bound_confidence:
            continue
        p1 = (int(w * bbox[i, 0]), int(h * bbox[i, 1]))
        p2 = (int(w * bbox[i, 2]), int(h * bbox[i, 3]))
        cls_name = GL_CLASSES[int(bbox[i, 5])]
        print(cls_name, p1, p2)
        cv2.rectangle(img, p1, p2, COLOR[int(bbox[i, 5])])
        cv2.putText(img, cls_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(img, str(confidence), (p1[0],p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.imwrite('predict/'+filename, img)
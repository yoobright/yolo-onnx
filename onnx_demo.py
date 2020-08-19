import numpy as np
import onnxruntime
import cv2


def sort_score_index(scores, threshold=0.0, top_k=0, descending=True):
    score_index = []
    for i, score in enumerate(scores):
        if (threshold > 0) and (score > threshold):
            score_index.append([score, i])
    # print(score_index)
    if not score_index:
        return []

    np_scores = np.array(score_index)
    if descending:
        np_scores = np_scores[np_scores[:, 0].argsort()[::-1]]
    else:
        np_scores = np_scores[np_scores[:, 0].argsort()]

    if top_k > 0:
        np_scores = np_scores[0:top_k]
    return np_scores.tolist()


def nms(boxes, scores, iou_threshold=0.5, score_threshold=0.3, top_k=0):
    if scores is not None:
        scores = sort_score_index(scores, score_threshold, top_k)
        if scores:
            order = np.array(scores, np.int32)[:, 1]
        else:
            return []
    else:
        y2 = boxes[:3]
        order = np.argsort(y2)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []

    while len(order) > 0:
        idx_self = order[0]
        idx_other = order[1:]
        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h
        over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= iou_threshold)[0]
        # over -> order
        order = order[inds + 1]

    return keep


def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


if __name__ == '__main__':
    # session = onnxruntime.InferenceSession('weights/yolov4-tiny.onnx', None)
    session = onnxruntime.InferenceSession('yolov4_1_3_416_416_static.onnx', None)
    input_name = session.get_inputs()[0].name
    print('Input Name:', input_name)

    img_bgr = cv2.imread("dog.jpg")
    h, w, _ = img_bgr.shape
    img = cv2.resize(img_bgr, (416, 416))
    img = img.astype('float32') / 255.
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, 3, 416, 416)
    print(img.shape)

    raw_result = session.run([], {input_name: img})

    # boxes = raw_result[1].reshape(-1, 4)
    # boxes = xywh2xyxy(boxes)
    # classes_score = raw_result[0].reshape(-1, 80)

    boxes = raw_result[0].reshape(-1, 4)
    classes_score = raw_result[1].reshape(-1, 80)
    num_cls = classes_score.shape[1]

    det_result = []
    for cls in range(num_cls):
        scores = classes_score[:, cls].flatten()
        pick = nms(boxes, scores, 0.6, 0.4)
        for i in range(len(pick)):
            det_result.append([cls, scores[pick][i], boxes[pick][i]])
    for det in det_result:
        print(det)
        # print(w, h)
        c1 = (int(det[2][0] * w), int(det[2][1] * h))
        c2 = (int(det[2][2] * w), int(det[2][3] * h))
        cv2.rectangle(img_bgr, c1, c2, (0, 255, 0), 2)

    cv2.imshow("test", img_bgr)
    cv2.waitKey()


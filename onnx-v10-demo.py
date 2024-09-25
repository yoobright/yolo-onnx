import argparse
import math
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


def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))



def plot_boxes_cv2(img, det_result, class_names=None):
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    w = img.shape[1] / 640.
    h = img.shape[0] / 640.

    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    for det in det_result:
        c1 = (int(det[2][0] * w), int(det[2][1] * h))
        c2 = (int(det[2][2] * w), int(det[2][3] * h))

        rgb = (255, 0, 0)

        if class_names:
            cls_conf = det[1]
            cls_id = det[0]
            label = class_names[cls_id]
            print("{}: {}".format(label, cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)

            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cc2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, cc2, rgb, -1)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
        img = cv2.rectangle(img, c1, c2, rgb, tl)
    return img

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="yolov10s.onnx",
                        help='model file')
    parser.add_argument('-i', '--input', type=str, default="dog.jpg",
                        help='input image')
    parser.add_argument('--names', type=str, default='coco.names', 
                        help='*.names path')
    args = parser.parse_args()
    print(args)
    model_file  = args.model
    image_file = args.input

    names = load_classes(args.names)

    # session = onnxruntime.InferenceSession('weights/yolov4-tiny.onnx', None)
    session = onnxruntime.InferenceSession(model_file, None)
    input_name = session.get_inputs()[0].name
    # print('Input Name:', input_name)
    input_shape = session.get_inputs()[0].shape
    print("The model expects input shape: ", input_shape)
    input_h = input_shape[2]
    input_w = input_shape[3]

    img_bgr = cv2.imread(image_file)
    h, w, _ = img_bgr.shape
    img = cv2.resize(img_bgr, (input_h, input_w))
    img = img.astype('float32') / 255.
    img = img.transpose(2, 0, 1)
    img = img.reshape(*input_shape)
    print(img.shape)

    prediction = session.run([session.get_outputs()[0].name], {input_name: img})[0]
    print(prediction.shape)

    boxes = prediction[0][:, :4]
    confidences = prediction[0][:, 4]
    class_ids =prediction[0][:, 5]

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), 0.5, 0.4)
    
    if len(indices) > 0:
        indices = indices.flatten()
        boxes = boxes[indices]
        confidences = confidences[indices]
        class_ids = class_ids[indices]

    # print(boxes, confidences, class_ids)

    det_result = []
    for i in range(len(boxes)):
        det_result.append([int(class_ids[i]), confidences[i], boxes[i]])

    img_show = plot_boxes_cv2(img_bgr, det_result, names)
    cv2.imshow("test", img_show)
    cv2.waitKey()

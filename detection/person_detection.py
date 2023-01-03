import numpy as np
import cv2
import insightface

assert insightface.__version__>='0.4'


def detect_person(img, detection_model='scrfd_person_2.5g.onnx', nms_thresh=0.5, input_size=(640, 640)):
    print('image shape', img.shape)
    detector = insightface.model_zoo.get_model(detection_model, download=True)
    detector.prepare(0, nms_thresh=nms_thresh, input_size=input_size)
    #img = cv2.imread(img_path)
    bboxes, vbboxes = detect_person_bboxes_vbboxes(img, detector)
    print(bboxes)

    return bboxes, vbboxes


def detect_person_bboxes_vbboxes(img, detector):
    bboxes, kpss = detector.detect(img)
    bboxes = np.round(bboxes[:,:4]).astype(int)
    kpss = np.round(kpss).astype(int)
    kpss[:,:,0] = np.clip(kpss[:,:,0], 0, img.shape[1])
    kpss[:,:,1] = np.clip(kpss[:,:,1], 0, img.shape[0])
    vbboxes = bboxes.copy()
    vbboxes[:,0] = kpss[:, 0, 0]
    vbboxes[:,1] = kpss[:, 0, 1]
    vbboxes[:,2] = kpss[:, 4, 0]
    vbboxes[:,3] = kpss[:, 4, 1]
    return bboxes, vbboxes
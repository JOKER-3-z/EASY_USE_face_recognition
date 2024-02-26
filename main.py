import cv2
import json
import torch
from model.YOLOV8 import YOLOv8_face
from model.Arcface import Face_re
from time import process_time

def get_id_from_feature(feature,threshold):
    identify="unKnow"
    with open('face_datasets.json') as f:
        dataset = json.load(f)
    max_sam=-999
    for stu_id, stored_features in dataset.items():
        similarity = feature.dot(torch.tensor(stored_features)).item()
        if max_sam < similarity:
            identify=stu_id
            max_sam=similarity
    if max_sam > threshold:
        identify=identify
        return identify
    else:
        return "unKnow"

def get_id_from_site(id_site_map,newsitelist):
    pass



if __name__ == "__main__":
    parameterRead=None
    with open('config.json') as f:
        parameterRead = json.load(f)
    Yolodefine=parameterRead['Yolov8']
    Arcfacedefine=parameterRead['Arcface']
    Identify_detect_threshold=Arcfacedefine['detect_Threshold']
    device_id=parameterRead['DEVICE']
    DEVICE=torch.device(device_id)

    # Initialize YOLOv8_face object detector
    YOLOv8_face_detector = YOLOv8_face(Yolodefine['modelpath'], conf_thres=Yolodefine['confThreshold'], iou_thres=Yolodefine['nmsThreshold'])
    FaceReModel=Face_re(Arcfacedefine['modelpath'],DEVICE)

    cap = cv2.VideoCapture(0)
    count=0
    while(True):
        # 一帧一帧捕捉
        ret, srcimg = cap.read()
        t1_time=process_time()
        if count%10==0:
            boxes, scores, classids, kpts = YOLOv8_face_detector.detect(srcimg)
            crop_img, landmarks = FaceReModel.crop_face_from_img(srcimg, boxes, kpts)
            name = ["UnKnow"] * len(crop_img)
            for i, (single_face, kp) in enumerate(zip(crop_img, landmarks)):
                if single_face is not None:
                    facial_features = FaceReModel.recognition(single_face, kp)
                    name[i] = get_id_from_feature(facial_features, Identify_detect_threshold)
        if len(scores) == 0: #对于未检测到人脸情况直接显示
            dstimg=srcimg
        else:
            # 显示返回的每帧
            dstimg = YOLOv8_face_detector.draw_detections_with_nopoint(srcimg, boxes,name)

        count += 1
        cv2.imshow('frame',dstimg)
        t2=process_time()
        print(t1_time , t2)
        print(t2-t1_time)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
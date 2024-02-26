import os
import numpy as np
import cv2
from PIL import Image
import torch
import json
from model.YOLOV8 import YOLOv8_face
from model.Arcface import Face_re

def load_folder(folder):
    stu_name=[]
    stu_imgpath=[]
    Add_list=os.listdir(folder)
    for single_stu in Add_list:
        stu=single_stu.split('.')[0]
        img_file=os.path.join(folder,single_stu)
        stu_imgpath.append(img_file)
        stu_name.append(stu)
    return stu_name,stu_imgpath

def Add_feature2dataset(feature_dict):
    with open('face_datasets.json','w', encoding='utf-8') as f:
        json.dump(feature_dict,f,ensure_ascii=False)
    print("Add "+str(len(feature_dict))+" student successful...")


if __name__ == "__main__":
    parameterRead=None
    print("-------------------------Load the configuration-------------------------")
    with open('config.json') as f:
        parameterRead = json.load(f)
    Yolodefine=parameterRead['Yolov8']
    Arcfacedefine=parameterRead['Arcface']
    device_id=parameterRead['DEVICE']
    Add_img_list=parameterRead['Add_folder']
    DEVICE=torch.device(device_id)

    # Initialize YOLOv8_face object detector
    YOLOv8_face_detector = YOLOv8_face(Yolodefine['modelpath'], conf_thres=Yolodefine['confThreshold'], iou_thres=Yolodefine['nmsThreshold'])
    FaceReModel=Face_re(Arcfacedefine['modelpath'],DEVICE)
    print("-------------------------Load the image list-------------------------")
    stu_name,stu_img_path=load_folder(Add_img_list)
    print("-------------------------     Get features     -------------------------")
    feature_dict={}
    for stu_n,stu_img in zip(stu_name,stu_img_path):
        srcimg=cv2.imdecode(np.fromfile(file=stu_img, dtype=np.uint8), cv2.IMREAD_COLOR)
        boxes, scores, classids, kpts = YOLOv8_face_detector.detect(srcimg)
        if len(scores) == 0: #对于未检测到人脸情况直接显示
            print("undetect face , please change image "+stu_n)
            continue
        else:
            crop_img,trans_landmark = FaceReModel.crop_face_from_img(srcimg, boxes,kpts)
            if crop_img[0] is not None:
                facial_features = FaceReModel.recognition(crop_img[0], trans_landmark[0])
                feature_dict[stu_n]=facial_features.tolist()
    Add_feature2dataset(feature_dict)
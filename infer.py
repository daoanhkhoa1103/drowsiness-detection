import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pygame
import argparse
from PIL import Image
from model import *

ap = argparse.ArgumentParser()

ap.add_argument("--path", default="./video_test.mp4", help="path to video")
ap.add_argument("--mode", default="image", help="webcam/video")
args = ap.parse_args()

def playMusic(name_music):
    pygame.init()
    pygame.mixer.init()

    pygame.mixer.music.load(name_music)
    pygame.mixer.music.play()

transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1),
                                    # transforms.RandomRotation(degrees=(0, 10)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

transform_val = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

model = CNN()
model.load_state_dict(torch.load('best_model_2.pth', map_location=torch.device('cpu')))
# model.to('cuda')
model.eval()

mp_face_mesh = mp.solutions.face_mesh

def plot_landmark(img_base, facial_area_obj):
    all_lm = []
    img = img_base.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks = results.multi_face_landmarks[0]
    for source_idx, target_idx in facial_area_obj:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]

        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
        all_lm.append(relative_source)
        all_lm.append(relative_target)
    
    all_lm = sorted(all_lm, key = lambda a: (a[0]))
    x_min, x_max = all_lm[0][0], all_lm[-1][0]
    all_lm = sorted(all_lm, key = lambda a: (a[1]))
    y_min, y_max =  all_lm[0][1], all_lm[-1][1]
    
    img_ = img[y_min:y_max+1,x_min:x_max+1]
    return img_, [(x_min, y_min), (x_max,y_max)]

#Encoder label
label2id = {
    0: 'Close',
    1: 'Open',
}

def predict(img, model):
    img = transform_val(img)    
    img = torch.unsqueeze(img, 0).to('cpu').float()    

    with torch.no_grad():
        output = model(img)
    
    output = F.softmax(output, dim = -1)
    predicted = torch.argmax(output)
    p = label2id[predicted.item()]
    prob = torch.max(output).item()
    
    return  p, round(prob,2)


if(args.mode == "webcam"):
    cap = cv2.VideoCapture(0)
    status_l = ""
    status_r = ""
    count = 0
    check_l = 0
    check_r = 0
    count_all = 0
    checksound = 1
    checkface = 1
    count_face = 0

    while 1:
        ret, image = cap.read()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            # image = cv2.imread(path)
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
       # try:
        if(results.multi_face_landmarks):
            count_face = 0
            checkface = 1
            l_eyebrow, coor1 = plot_landmark(image, mp_face_mesh.FACEMESH_LEFT_EYE)
            _, coor3 = plot_landmark(image, mp_face_mesh.FACEMESH_TESSELATION)
            img = Image.fromarray(l_eyebrow)
            pred, prob = predict(img, model)

            if(str(pred) == 'Close'):
                check_l = 1
            else:
                check_l = 0
            if(float(prob) > 0.75):
                status_l = str(pred)

            # pred = status_l + '|' + str(prob)
            pred = status_l
            # print("Mat trai: ", pred)
            cv2.putText(image, str(pred), (coor1[0][0], coor1[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
            cv2.rectangle(image, coor1[0], coor1[1], (255,0,0), 1)
            cv2.rectangle(image, coor3[0], coor3[1], (255,0,0), 1)
        
        
            r_eyebrow, coor2 = plot_landmark(image, mp_face_mesh.FACEMESH_RIGHT_EYE)
            img = Image.fromarray(r_eyebrow)
            pred, prob = predict(img, model)

            if(str(pred) == 'Close'):
                check_r = 1
            else:
                check_r = 0
            if(float(prob)  > 0.75):
                status_r = str(pred)

            # pred = status_r + '|' + str(prob)
            pred = status_r
            # print("Mat phai: ", pred)
            cv2.putText(image, str(pred), (coor2[0][0], coor2[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
            cv2.rectangle(image, coor2[0], coor2[1], (255,0,0), 1)

            if(check_l == 1 and check_r == 1):
                count += 1
            else:
                count = 0
            if(count > 15):
                print("Canh bao ngu gat!!!")
                if(count==100 or count==200 or count==300):
                    checksound = 1
                if(checksound == 1):
                    playMusic('voice.mp3')
                    checksound = 0
                cv2.rectangle(image, (3,3), (image.shape[1]-3, image.shape[0]-3), (0,0,255), 6)
                cv2.putText(image, "Canh bao ngu gat!", (int(image.shape[1]/3)-35, image.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                if(count == 16):
                    count_all += 1
            else:
                checksound = 1
        else:
            count_face += 1
            if(checkface == 1):
                checkface = 0
                playMusic('voice2.wav')
            if(count_face>300):
                checkface = 1
                playMusic('voice2.wav')
                count_face = 1
       # except:
         #   print("Exeption!!!")
            


        ratio = image.shape[0]/image.shape[1]
        image = cv2.resize(image, (900,int(ratio*900)))
        
        blank_image = np.zeros((80, 900, 3), np.uint8)
        blank_image[:] = (204,231,232)
        cv2.putText(blank_image, "PHAN MEM THEO DOI LAI XE AN TOAN - SMART CHECK", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)
        
        text = "So lan canh bao: " + str(count_all)
        cv2.putText(blank_image, text, (250,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)
        
        if(count_all < 4):
            text = "Tinh trang: Khoe khoan"
            cv2.putText(blank_image, text, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
        elif(count_all > 3 and count_all < 7):
            text = "Tinh trang: Met moi"
            cv2.putText(blank_image, text, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,215,255), 2, cv2.LINE_AA)
        elif(count_all > 6):
            text = "Tinh trang: Rat met moi"
            cv2.putText(blank_image, text, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)

        

        combined_image = cv2.vconcat([blank_image, image])
        
        cv2.imshow('Webcam',combined_image)
        k = cv2.waitKey(20) & 0xff
        if k == ord('q'):
            break
elif(args.mode == "video"):
    cap = cv2.VideoCapture(args.path)
    status_l = ""
    status_r = ""
    count = 0
    check_l = 0
    check_r = 0
    count_all = 0
    checksound = 1
    checkface = 1
    count_face = 0

    while 1:
        ret, image = cap.read()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            # image = cv2.imread(path)
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        try:
            if(results.multi_face_landmarks):
                count_face = 0
                checkface = 1
                l_eyebrow, coor1 = plot_landmark(image, mp_face_mesh.FACEMESH_LEFT_EYE)
                _, coor3 = plot_landmark(image, mp_face_mesh.FACEMESH_TESSELATION)
                img = Image.fromarray(l_eyebrow)
                pred, prob = predict(img, model)

                if(str(pred) == 'Close'):
                    check_l = 1
                else:
                    check_l = 0
                if(float(prob) > 0.75):
                    status_l = str(pred)

                # pred = status_l + '|' + str(prob)
                pred = status_l
                # print("Mat trai: ", pred)
                cv2.putText(image, str(pred), (coor1[0][0], coor1[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
                cv2.rectangle(image, coor1[0], coor1[1], (255,0,0), 1)
                cv2.rectangle(image, coor3[0], coor3[1], (255,0,0), 1)
            
            
                r_eyebrow, coor2 = plot_landmark(image, mp_face_mesh.FACEMESH_RIGHT_EYE)
                img = Image.fromarray(r_eyebrow)
                pred, prob = predict(img, model)

                if(str(pred) == 'Close'):
                    check_r = 1
                else:
                    check_r = 0
                if(float(prob)  > 0.75):
                    status_r = str(pred)

                # pred = status_r + '|' + str(prob)
                pred = status_r
                # print("Mat phai: ", pred)
                cv2.putText(image, str(pred), (coor2[0][0], coor2[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
                cv2.rectangle(image, coor2[0], coor2[1], (255,0,0), 1)

                if(check_l == 1 and check_r == 1):
                    count += 1
                else:
                    count = 0
                if(count > 15):
                    print("Canh bao ngu gat!!!")
                    if(count==100 or count==200 or count==300):
                        checksound = 1
                    if(checksound == 1):
                        playMusic('voice.mp3')
                        checksound = 0
                    cv2.rectangle(image, (3,3), (image.shape[1]-3, image.shape[0]-3), (0,0,255), 6)
                    cv2.putText(image, "Canh bao ngu gat!", (int(image.shape[1]/3)-35, image.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                    if(count == 16):
                        count_all += 1
                else:
                    checksound = 1
            else:
                count_face += 1
                if(checkface == 1):
                    checkface = 0
                    playMusic('voice2.wav')
                if(count_face>300):
                    checkface = 1
                    playMusic('voice2.wav')
                    count_face = 1
        except:
            print("Exeption!!!")
            


        ratio = image.shape[0]/image.shape[1]
        image = cv2.resize(image, (900,int(ratio*900)))
        
        blank_image = np.zeros((80, 900, 3), np.uint8)
        blank_image[:] = (204,231,232)
        cv2.putText(blank_image, "PHAN MEM THEO DOI LAI XE AN TOAN - SMART CHECK", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)
        
        text = "So lan canh bao: " + str(count_all)
        cv2.putText(blank_image, text, (250,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)
        
        if(count_all < 4):
            text = "Tinh trang: Khoe khoan"
            cv2.putText(blank_image, text, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
        elif(count_all > 3 and count_all < 7):
            text = "Tinh trang: Met moi"
            cv2.putText(blank_image, text, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,215,255), 2, cv2.LINE_AA)
        elif(count_all > 6):
            text = "Tinh trang: Rat met moi"
            cv2.putText(blank_image, text, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)

        

        combined_image = cv2.vconcat([blank_image, image])
        
        cv2.imshow('Video',combined_image)
        k = cv2.waitKey(20) & 0xff
        if k == ord('q'):
            break
else:
    print("Error!!!!")
    print("Please use: python infer.py --model webcam/video --path ./video.mp4")

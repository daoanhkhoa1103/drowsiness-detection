import cv2
import mediapipe as mp
import random
import string
from glob import glob

mp_face_mesh = mp.solutions.face_mesh

def plot_landmark(img_base, facial_area_name, facial_area_obj):
    all_lm = []
    name = facial_area_name.split("_", 1)[-1]
    if("LIPS" in name):
        name = './lips/' + name + '_' + str(''.join(random.choices(string.ascii_uppercase + string.digits, k=7))) + '.jpg'
    elif("EYE" in name):
        name = './eyebrow/' + name + '_' + str(''.join(random.choices(string.ascii_uppercase + string.digits, k=7))) + '.jpg'
    print(name)

    img = img_base.copy()
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
    # cv2.imwrite(name, img_)

    

def plot_landmark_nose(img_base):
    all_lm = []
    name = './nose/NOSE_' + str(''.join(random.choices(string.ascii_uppercase + string.digits, k=7))) + '.jpg'

    img = img_base.copy()
    landmarks = results.multi_face_landmarks[0]
    for source_idx, target_idx in mp_face_mesh.FACEMESH_RIGHT_EYE:
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


    all_lm = []
    
    img = img_base.copy()
    landmarks = results.multi_face_landmarks[0]
    for source_idx, target_idx in mp_face_mesh.FACEMESH_LEFT_EYE:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]

        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
        all_lm.append(relative_source)
        all_lm.append(relative_target)
        
    all_lm = sorted(all_lm, key = lambda a: (a[0]))
    x_min_2, x_max_2 = all_lm[0][0], all_lm[-1][0]
    
    all_lm = []
    
    img = img_base.copy()
    landmarks = results.multi_face_landmarks[0]
    for source_idx, target_idx in mp_face_mesh.FACEMESH_LIPS:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]

        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
        all_lm.append(relative_source)
        all_lm.append(relative_target)
        
    all_lm = sorted(all_lm, key = lambda a: (a[0]))
    x_min_3, x_max_3 = all_lm[0][0], all_lm[-1][0]
    all_lm = sorted(all_lm, key = lambda a: (a[1]))
    y_min_3, y_max_3 =  all_lm[0][1], all_lm[-1][1]
    y_choose = y_min_3 - (y_max_3 - y_min_3)/4
    # cv2.rectangle(img, (x_max,y_max), (x_min_2,int(y_choose)), (255, 0, 0), 1)
    # cv2.imwrite("2.jpg", img)

    img_ = img[y_max:int(y_choose)+1,x_max:x_min_2+1]
    cv2.imwrite(name, img_)

# dir = glob('./trainset/*')
# # count = 0
# for ann in dir:
#     # print(ann)
#     path = ann
#     with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
#         image = cv2.imread(path)
#         results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     if(results.multi_face_landmarks):
#         plot_landmark(image,'FACEMESH_LEFT_EYE',mp_face_mesh.FACEMESH_LEFT_EYE)
#         plot_landmark(image,'FACEMESH_RIGHT_EYE',mp_face_mesh.FACEMESH_RIGHT_EYE)
from PIL import Image

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
print("Starting...")
cap = cv2.VideoCapture(0)
print("Camera is ready to use1111111.")
mp_face_mesh = mp.solutions.face_mesh
print("Camera is ready to use.")

count = 0
while 1:
    ret, image = cap.read()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        # image = cv2.imread(path)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if(count%5==0):
        if(results.multi_face_landmarks):
            l_eyebrow, coor1 = plot_landmark(image, mp_face_mesh.FACEMESH_LEFT_EYE)
            r_eyebrow, coor2 = plot_landmark(image, mp_face_mesh.FACEMESH_RIGHT_EYE)
            imgr = Image.fromarray(l_eyebrow)
            imgl = Image.fromarray(l_eyebrow)
            name = "./new_data/" + str(count) + "_l" + ".jpg"
            print(name)
            imgl.save(name)
            name = "./new_data/" + str(count) + "_r" + ".jpg"

            imgr.save(name)

    count+=1

    if(count==100):
        break
    cv2.putText(image, "Dang tien hanh thu thap du lieu ...", (40,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow('Webcam',image)
    k = cv2.waitKey(20) & 0xff
    if k == ord('q'):
        break

print("Releasing resources...")
if cap.isOpened():
    cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
pygame.mixer.quit()  # Stop any audio playback (if pygame was used)
print("Resources released successfully.")
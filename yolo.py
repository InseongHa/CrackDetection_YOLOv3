# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
import queue
import math
import cv2
from skimage import data
from skimage.color import rgb2gray
from skimage.data import page
from skimage.filters import (threshold_sauvola)
from skimage.morphology import skeletonize
from skimage.util import invert
from scipy import ndimage as ndi
from skimage import feature

class YOLO(object):
    _defaults = {
        "model_path": 'tiny_yolo_crack_20200623.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "classes_path": 'classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def crop_image(self, image, orig_image, saving_bounding_boxes): 
	# saving_bounding_boxes = [left, top, right, bottom]의 1차원 배열형태
        if (saving_bounding_boxes[0] < 0):
            saving_bounding_boxes[0] = 0
        left = int(saving_bounding_boxes[1])
        if (saving_bounding_boxes[1] < 0):
            saving_bounding_boxes[1] = 0
        top = int(saving_bounding_boxes[1])
        right = int(saving_bounding_boxes[2])
        bottom = int(saving_bounding_boxes[3])
        print((left, top), (right, bottom))
        ###여기서 어떻게 image를 고쳐야할까 >> 이 코드말고 bounding box만 똑 crop해서 저장하는 코드 없을까?
        #cropped_frame = image[left:right, top:bottom, :]
        #cropped_frame = image.crop(saving_bounding_boxes) 이러면 총체적 난국이 된다.
        cropped_frame = orig_image[left:right, top:bottom, :]
        cropped_frame = cropped_frame.astype('uint8')
        	#이 이미지만 따로 저장하는 코드를 만들어서 테스트해보자
        	#cv2.imwrite('Saved_Images/cropped_frame',) : 이 부분은 video읽는 부분에서 while문으로 돌리면서 순차적으로 저장해야 할듯?
        return cropped_frame

    # crop된 frame image를 전처리하는 단계
    # <1> Binarization

    def binarization_skeletonize_(self, cropped_frame):
        window_size_Pw = 71
        
        img = cropped_frame
        img_gray = rgb2gray(img)
        #=============================== binarization: 균열인 부분과 균열이 아닌 부분 분리
        thresh_sauvola_Pw = threshold_sauvola(img_gray, window_size=window_size_Pw, k=0.42)
        binary_sauvola_Pw = img_gray > thresh_sauvola_Pw
        binary_sauvola_Pw_bw = img_gray > thresh_sauvola_Pw
        binary_sauvola_Pw_bw.dtype = 'uint8'
        binary_sauvola_Pw_bw *= 255

        sauvola_frames_Pw_bw = binary_sauvola_Pw_bw
        sauvola_frames_Pw = binary_sauvola_Pw
        #=============================== skeletonization
        img_Pw = invert(sauvola_frames_Pw)
        skeleton_Pw = skeletonize(img_Pw)
        skeleton_Pw.dtype = 'uint8'
        skeleton_Pw *= 255
        
        skeleton_frames_Pw = skeleton_Pw
        #=============================== edge detection
        edges_Pw = feature.canny(sauvola_frames_Pw, 0.09)
        edges_Pw.dtype = 'uint8'
        edges_Pw *= 255

        edges_frames_Pw = edges_Pw
        #=============================== 균열 폭 계산
# 1) BFS를 통해 skeleton을 찾습니다.
# 2) 찾은 skeletion pixel에서 5 pixel의 거리만큼 떨어져있는 인접 skeletion pixel들을 기준으로 방향을 설정합니다.
# 3) 균열의 진행 방향에 수직인 직선을 긋습니다.
# 4) 균열의 폭은 이 수직선과 Edge가 만나는 거리상에 위치한 pixel 단위로 산출합니다.
# 5) 이를 실제 mm 단위로 변환한 후, 위험군을 분류합니다.
        dx_dir_right = [-5, -5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5]
        dy_dir_right = [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 4, 3, 2, 1]
        dx_dir_left = [5, 5, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -5]
        dy_dir_left = [0, -1, -2, -3, -4, -5, -5, -5, -5, -5, -4, -3, -2, -1]
        dx_bfs = [-1, -1, 0, 1, 1, 1, 0, -1]
        dy_bfs = [0, 1, 1, 1, 0, -1, -1, -1]

        # Searching the skeleton through BFS.
        start = [0, 0]
        next = []
        q = queue.Queue()
        q.put(start)

        len_x = skeleton_frames_Pw.shape[0]
        len_y = skeleton_frames_Pw.shape[1]

        visit = np.zeros((len_x, len_y))
        
        while (q.empty() ==0):
            next = q.get()
            x = next[0]
            y = next[1]
            right_x = right_y = left_x = left_y = -1
            
            # Estimating the direction of the crack from skeleton
            if (skeleton_frames_Pw[x][y] == 255):
                for i in range(0, len(dx_dir_right)):
                    right_x = x + dx_dir_right[i]
                    right_y = y + dy_dir_right[i]
                    if (right_x<0 or right_y<0 or right_x>=len_x or right_y>=len_y):
                        right_x = right_y = -1
                        continue;
                    if (skeleton_frames_Pw[right_x][right_y] == 255): break;
                    if (i==13): right_x = right_y = -1
                
                if (right_x == -1):
                    right_x = x
                    right_y = y
                
                for i in range(0, len(dx_dir_left)):
                    left_x = x + dx_dir_left[i]
                    left_y = y + dy_dir_left[i]
                    if (left_x<0 or left_y<0 or left_x>=len_x or left_y>=len_y):
                        left_x = left_y = -1
                        continue;
                    if (skeleton_frames_Pw[left_x][left_y] == 255): break;
                    if (i==13): left_x = left_y = -1

                if (left_x == -1):
                    left_x = x
                    left_y = y

                # Set the direction of the crack as angle(theta) by using acos formula
                base = right_y - left_y
                height = right_x - left_x
                hypotenuse = math.sqrt(base*base + height*height)

                if (base==0 and height!=0): theta = 90.0
                elif (base==0 and height==0): continue
                else: theta = math.degrees(math.acos((base * base + hypotenuse * hypotenuse - height * height)/(2.0 * base * hypotenuse)))

                theta += 90
                dist = 0

                # Calculate the distance if the perpendicular line meets the edge of the crack.
                for i in range(0,2):
                    pix_x = x
                    pix_y = y
                    if (theta>360): theta -= 360
                    elif (theta<0): theta += 360

                    if (theta == 0.0 or theta == 360.0):
                        while (1):
                            pix_y += 1
                            if (pix_y>=len_y):
                                pix_x = x
                                pix_y = y
                                break;
                            if (edges_frames_Pw[pix_x][pix_y] == 255): break;

                    elif (theta == 90.0):
                        while (1):
                            pix_x -= 1
                            if(pix_x<0):
                                pix_x = x
                                pix_y = y
                                break;
                            if (edges_frames_Pw[pix_x][pix_y] == 255): break;
                    
                    elif (theta == 180.0):
                        while (1):
                            pix_y -= 1
                            if (pix_y<0):
                                pix_x = x
                                pix_y = y
                                break;
                            if (edges_frames_Pw[pix_x][pix_y] == 255): break;

                    elif (theta == 270.0):
                        while (1):
                            pix_x += 1
                            if (pix_x>=len_x):
                                pix_x = x
                                pix_y = y
                                break;
                            if (edges_frames_Pw[pix_x][pix_y] ==255): break;
                    
                    else:
                        a = 1
                        radian = math.radians(theta)
                        while (1):
                            pix_x = x - round(a*math.sin(radian))
                            pix_y = y - round(a*math.cos(radian))
                            if (pix_x<0 or pix_y<0 or pix_x>=len_x or pix_y>=len_y):
                                pix_x = x
                                pix_y = y
                                break;
                            if (edges_frames_Pw[pix_x][pix_y] == 255): break;

                            if (theta>0 and theta<90):
                                if (pix_y+1<len_y and edges_frames_Pw[pix_x][pix_y+1] == 255):
                                    pix_y+=1
                                    break;
                                if (pix_x-1>=0 and edges_frames_Pw[pix_x-1][pix_y] == 255):
                                    pix_x-=1
                                    break;

                            elif (theta>90 and theta<180):
                                if (pix_y-1>=0 and edges_frames_Pw[pix_x][pix_y-1] == 255):
                                    pix_y-=1
                                    break;

                            elif (theta>180 and theta<270):
                                if (pix_y-1>=0 and edges_frames_Pw[pix_x][pix_y-1]==255): 
                                    pix_y-=1
                                    break;
                                if (pix_x+1<len_x and edges_frames_Pw[pix_x+1][pix_y]==255): 
                                    pix_x+=1
                                    break;         

                            elif (theta>270 and theta<360): 
                                if (pix_y+1<len_y and edges_frames_Pw[pix_x][pix_y+1]==255): 
                                    pix_y+=1
                                    break;
                                if (pix_x+1<len_x and edges_frames_Pw[pix_x+1][pix_y]==255): 
                                    pix_x+=1
                                    break;
                            a+=1
                    
                    dist += math.sqrt((y-pix_y)**2 + (x-pix_x)**2)
                    theta += 180

                # The list which saves the width of the crack.
                crack_width = dist

            for i in range(0,8):
                next_x = x + dx_bfs[i]
                next_y = y + dy_bfs[i]

                if (next_x < 0 or next_y < 0 or next_x >= len_x or next_y >= len_y): continue;
                if (visit[next_x][next_y] == 0):
                    q.put([next_x, next_y])
                    visit[next_x][next_y] = 1
        
        # Convert into real width.
        # print(crack_width)
        if (crack_width == 0):
            real_width = 0
        elif (crack_width < 10):
            real_width = round(crack_width*0.92, 2)
        else: #이 부분 좀 다른데 어떻게 처리할지
            real_width = round(crack_width*0.92, 2)

        print('균열 폭: ', real_width)
        return real_width

    def detect_image(self, image, orig_image):
        start = timer()
        
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        real_width_list = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
            saving_bounding_boxes = [left, top, right, bottom] # (left, top, right, bottom) 형식으로 저장하지 않아도 괜찮나?
            cropped_image = self.crop_image(image, orig_image, saving_bounding_boxes)
            real_width = self.binarization_skeletonize_(cropped_image)
            real_width_list.append(real_width)

        end = timer()
        print(end - start)
        return image, real_width_list

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    file_width = open('width.txt', 'w')
    width = []
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    #video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_FourCC  = cv2.VideoWriter_fourcc(*"mp4v")
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        # Save those information into text files.
        
        return_value, frame = vid.read()
        if not return_value: break
        orig_image = frame
        image = Image.fromarray(frame)
        image, width = yolo.detect_image(image, orig_image)
        result = np.asarray(image)
        if(len(width) > 0):
            file_width.write(str(width) + 'mm' + '\n') # 제대로 계산하는건지 별도의 video를 구해서 test 해봐야 할 것
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    file_width.close()
    yolo.close_session()


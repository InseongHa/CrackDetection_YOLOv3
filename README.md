# Webcam 및 mobile을 이용한 실시간 균열 탐지

## ◆ 개요
* 공사 현장에서는 사람의 눈으로 직접 구조물 또는 건축물의 균열을 탐지하고 판단합니다. 이와 같은 육안조사에 의존하는 방식은 부정확할 수 있으며, 조사자의 주관적인 판단에 따라 결과에 차이가 있을 수 있습니다. 딥러닝과 영상처리 기술을 활용하여 이러한 한계를 극복할 수 있습니다.

</br>

### ▶ 기능 및 목적
* 입력된 이미지 및 동영상으로부터 균열을 탐지합니다.
* 탐지한 균열의 폭을 측정하고 실제 크기로 변환하여 결과를 저장합니다.

</br>

### ▶ 사용기술
* 균열 탐지를 위한 알고리즘으로는 **YOLOv3 Tiny version**(*<https://github.com/qqwweee/keras-yolo3>*)를 사용하였습니다.
* 기존의 YOLOv3는 Darknet 프레임워크에서 수행해야 하지만, 범용성을 위해 Keras로 변환하는 작업을 거친 버전을 선택했습니다.
    + tiny yolo는 yolo에 비해 정확도가 떨어지지만 크기가 작고 빠르다는 장점을 가집니다. 모바일 환경에서 수행하는 것이 목적이었기에 tiny yolo를 사용했습니다.
* 입력된 이미지 및 동영상에서 균열을 탐지한 후, 해당 균열에 대한 폭을 계산하는 방식으로는 Binarization, Skeletonization, Edge Detection 과정을 수행했습니다.

</br>

### ▶ 주요 환경
* Python 3.7
* GoogleColaboratory
* Keras 2.1.5
* Tensorflow 1.6.0
* Image set: Training 과정에는 유타 주립 대학교(Utah)와 중동 공과대학교(METU)에서 수집한 균열 이미지 데이터 셋 및 google dataset search에서 사용할 수 있는 데이터셋들을 사용했습니다.

</br>
</br>

## ◆ 과정
* 전체적인 처리과정은 다음과 같습니다.
![img](https://user-images.githubusercontent.com/59737066/89514324-ae166100-d810-11ea-87d5-d4045869651f.png)

#### 1. Darknet 프레임워크 형식으로 구성된 YOLOv3 모델을 Keras 환경에서 구동할 수 있도록 convert 합니다.
#### 2. Training시킬 이미지들에 대해 labelling을 진행합니다. (*<https://github.com/tzutalin/labelimg>*)
* 2-1. 5,000장의 특정재질 균열 이미지에 대해 labeling을 진행했습니다.
#### 3. Transfer Learning 과정을 통해 기존의 model을 목적에 맞게끔 re-training 시킵니다.
* 3-1. "train.py"를 이용해 re-training할 필요 없이 model을 만들 수 있습니다.
#### 4. 성능 확인을 위해 이미지 한 장에 대해 inference 과정을 수행합니다.
> Convesion_and_TransferLearning.ipynb
> "convert.py"와 "train.py"를 결합한 내용입니다.

</br>

#### 5. 움직이는 동영상에서 제대로 수행하는 지 확인합니다.
* 5-1. 이 과정에서는 webcam을 우선해서 테스트했기 때문에 GoogleColaboratory를 이용하지 않고 local환경에서 anaconda를 이용했습니다.
> "tiny_yolo_video.py"에서
```
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help = "Video input path"
        # webcam을 사용하고 싶으면 type=int로 수정
    )
```
에 대해 수정사항이 필요합니다.
* 5-2. 다음과 같은 명령어를 통해 수행합니다.
```
positional arguments:
  --input        Video input path
  --output       Video output path
```

</br>

#### 6. Binarization → Skeletonization 의 과정을 순서대로 수행합니다.
* 6-1. 전처리과정에 대한 code를 "yolo.py"에 추가해 동영상을 입력하면 균열을 탐지하는 동시에 폭을 측정하고 추출할 수 있도록 통합했습니다.
```
    def crop_image(self, image, orig_image, saving_bounding_boxes): 
    ...
    def binarization_skeletonize_(self, cropped_frame):
    ...
```
</br>

## 참조
* Joseph Redmon, Ali Farhadi. "*YOLOv3: An Incremental Imrprovement*". Computer Vision and Pattern Recognition, 2018. (*<https://arxiv.org/abs/1804.02767>*)
* Ross Girshick, et al. "*You Only Look Once: Real-Time Detection*". Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition(CVPR), 2016, PP.779-788. (*<https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html>*)

</br>

## 설치

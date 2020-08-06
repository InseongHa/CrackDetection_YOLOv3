Webcam 및 mobile을 이용한 실시간 균열 탐지
==========================================
개요
----
* 공사 현장에서는 사람의 눈으로 직접 구조물 또는 건축물의 균열을 탐지하고 판단합니다. 이와 같은 육안조사에 의존하는 방식은 부정확할 수 있으며, 조사자의 주관적인 판단에 따라 결과에 차이가 있을 수 있습니다. 딥러닝과 영상처리 기술을 활용하여 이러한 한계를 극복할 수 있습니다.

사용기술 및 과정요약
--------------------
* 균열 탐지를 위한 알고리즘으로는 **YOLOv3**(*<https://github.com/qqwweee/keras-yolo3>*)를 사용하였습니다.
* 기존의 YOLOv3는 Darknet 프레임워크에서 수행해야 하지만, 범용성을 위해 Keras로 변환하는 작업을 거친 버전을 선택했습니다.
* 입력된 이미지 및 동영상에서 균열을 탐지한 후, 해당 균열에 대한 폭을 계산하는 방식으로는 Binarization, Skeletonization, Edge Detection 과정을 수행했습니다.


기능 및 목적
------------
* 입력된 이미지 및 동영상으로부터 균열을 탐지합니다.
* 탐지한 균열의 폭을 측정하고 실제 크기로 변환하여 결과를 저장합니다.

과정
----
1. 

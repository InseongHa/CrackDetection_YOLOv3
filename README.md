# YOLOv3-CrackDetection

# Binarization_Skeletonization_EdgeDetection_2.ipynb 상 에로사항
  **20200706**
> model.predict([input_1, input_2, input_3]) 의 3 numpyt array size를 지정해야 한다.

> input_1은 4D, input_2는 5D, input_3모름

> input_2는 (13, 13, 3, 6)의 shape을 가져야 한다.

> 13 = 378(frame수) / 32(yolo.py참고), 3은 그거고, 6은 모름

>> 이용하려던 프레임 추출 코드는 아무것도 입력돼있지 않은 깨끗한 video에서 frame단위로 추출할 때, bounding box를 기록하는 코드였다.
>> yolo model을 이용해 video에 이미 bounding box를 찍어냈으므로 위 프레임 추출 코드를 이용할 필요가 없었다.

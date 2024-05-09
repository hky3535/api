# api
整合 grounding_dino yolo_worldu yolov4 yolov5 yolov5u yolov8u 目标识别功能，并形成API接口供HTTP协议调用，多种不同框架共用同一套请求体

## webUI 使用web界面操作
http://0.0.0.0:60000/

## 使用python调用api方式操作
wget http://0.0.0.0:60000/usage > test.py

## 使用Docker部署
docker run -itd --name api --privileged --gpus all -p 60000:60000 hky3535/ai_vision:api_4.2

## 安装部署方法
pip安装 
.requirements.txt
./common/requirements.txt
./common/engines/bot_sort/requirements.txt
./common/engines/byte_track/requirements.txt
./common/engines/iou_track/requirements.txt
./common/engines/grounding_dino/requirements.txt
./common/engines/yolo_worldu/requirements.txt
./common/engines/yolov4/requirements.txt
./common/engines/yolov5/requirements.txt
./common/engines/yolov5u/requirements.txt
./common/engines/yolov8u/requirements.txt


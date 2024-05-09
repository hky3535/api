import importlib
import os
packages = {"cv2": "opencv-python", "numpy": "numpy", "requests": "requests"}
for package_name, package_install in packages.items():
    try:
        globals()[package_name] = __import__(package_name)
    except ImportError:
        os.system(f"python3 -m pip install {package_install} -i https://pypi.tuna.tsinghua.edu.cn/simple")
        globals()[package_name] = __import__(package_name)
import base64
from pathlib import Path
import random

def cv_b64(frame_cv):
    frame_b64 = str(base64.b64encode(cv2.imencode(".jpg", frame_cv)[1]))[2:-1]
    return frame_b64

def b64_cv(frame_b64):
    frame_cv = cv2.imdecode(numpy.frombuffer(base64.b64decode(frame_b64), numpy.uint8), cv2.IMREAD_COLOR)
    return frame_cv

def b_b64(file_b):
    file_b64 = base64.b64encode(file_b).decode("utf-8")
    return file_b64

def b64_b(file_b64):
    file_b = base64.b64decode(file_b64)
    return file_b

base_url = str() # API调用地址

def post(function, data):
    try:
        response = requests.post(url=f"{base_url}/{function}", json=data, timeout=120)
        if response.status_code != 200: 
            return False, f"请求失败！请求码：{response.status_code}；返回内容：{response.content.decode()}"
        response = response.json() # 解析返回内容 {"ret": True, "response": results}
        return response["ret"], response["response"]
    except Exception as e:
        return False, f"请求崩溃！崩溃原因：{str(e)}"

def status():
    ret, response = post(function="status", data={})
    if not ret: return False, f"获取状态失败，失败原因：{response}"
    return True, response

def load(data):
    ret, response = post(function="load", data=data)
    if not ret: return False, f"模型加载失败，失败原因：{response}"
    return True, "模型加载成功"

def infer(data):
    ret, response = post(function="infer", data=data)
    if not ret: return False, f"图像推理失败，失败原因：{response}"
    return True, response

# 总父类
class Base:
    def __init__(self, weight_path=None, config_path=None, device=None, half=None):
        self.engine_name = ''.join(['_' + _.lower() if _.isupper() else _ for _ in self.__class__.__name__]).lstrip('_')
        self.weight_path = weight_path
        self.config_path = config_path
        self.device = device
        self.half = half
        # 获取到权重文件/配置文件的文件名
        self.weight_name = Path(weight_path).name if weight_path is not None else None
        self.config_name = Path(config_path).name if config_path is not None else None
        # 将任务名设置为权重文件名
        self.engine_task = self.weight_name
    
    def load(self):
        arguments = dict()
        if self.weight_path is not None: arguments["weight"] = {"name": self.weight_name, "b64": b_b64(open(self.weight_path, "rb").read())}
        if self.config_path is not None: arguments["config"] = {"name": self.config_name, "b64": b_b64(open(self.config_path, "rb").read())}
        if self.device is not None: arguments["device"] = self.device
        if self.half is not None: arguments["half"] = self.half
        data = {
            "engine_name": self.engine_name, 
            "engine_task": self.engine_task, 
            "arguments": arguments
        }
        ret, response = load(data=data)
        return ret, response

    def infer(self, frame=None, results=None, conf=None, iou=None, classes=None):
        arguments = dict()
        if frame is not None: arguments["frame"] = {"b64": cv_b64(frame)}
        if results is not None: arguments["results"] = results
        if conf is not None: arguments["conf"] = conf
        if iou is not None: arguments["iou"] = iou
        if classes is not None: arguments["classes"] = classes
        data = {
            "engine_name": self.engine_name, 
            "engine_task": self.engine_task, 
            "arguments": arguments
        }
        ret, response = infer(data=data)
        return ret, response
        
# 初始化 获取所有API提供的功能并创建继承类 
def init(url="http://0.0.0.0:60000"):
    # 设置API调用地址（全局）
    global base_url
    base_url = url
    # 获取到所有API中提供的功能名称
    ret, response = status()
    if not ret: return ret, response
    # 将所有功能名（下划线）称变化为类名（大驼峰），并继承Base类完成继承
    for engine_name in response:
        # 生成类名（大驼峰）
        engine_class_name = ''.join([_.capitalize() for _ in engine_name.split('_')])
        # 创建并继承类
        globals()[engine_class_name] = type(engine_class_name, (Base, ), {"__init__": lambda self, *args, **kwargs: super(self.__class__, self).__init__(*args, **kwargs)})
    return ret, response

# 绘制结果
colors = dict()
def draw(frame, result):
    x0, y0, x1, y1, conf, clas = result
    # 生成类别对应颜色
    if clas not in colors: colors[clas] = [random.randint(0, 255) for _ in range(3)]
    color = colors[clas]
    # 生成图像尺寸对应粗细
    thickness = int(min(frame.shape[:2]) * 0.005)
    # 绘制目标识别边缘框
    frame = cv2.rectangle(frame, (x0, y0), (x1, y1), color=color, thickness=thickness)
    # 绘制目标识别标签
    label = f"{clas} {(conf/100):.2f}"
    size = 0.5
    margin = 3
    frame = cv2.putText(frame, label, (x0+margin, y0+int(size*20)+thickness+margin), cv2.FONT_HERSHEY_SIMPLEX, size, color=color , thickness=thickness)

    return frame

if __name__ == "__main__":
    """
    参数说明：
                   | weight_path | config_path | device | half     frame | conf | iou |   classes   
    ---------------+-------------+-------------+--------+------- --------+------+-----+-------------
    GroundingDino  |      Y      |      Y      |   Y    |            Y   |  Y   |  Y  | Y list(str) 
    ---------------+-------------+-------------+--------+------- --------+------+-----+-------------
    YoloWorldu     |      Y      |             |   Y    |  Y         Y   |  Y   |  Y  | Y list(str) 
    ---------------+-------------+-------------+--------+------- --------+------+-----+-------------
    Yolov4         |      Y      |      Y      |   Y    |            Y   |  Y   |  Y  | Y list(int) 
    ---------------+-------------+-------------+--------+------- --------+------+-----+-------------
    Yolov5         |      Y      |             |   Y    |  Y         Y   |  Y   |  Y  | Y list(int) 
    ---------------+-------------+-------------+--------+------- --------+------+-----+-------------
    Yolov5u        |      Y      |             |   Y    |  Y         Y   |  Y   |  Y  | Y list(int) 
    ---------------+-------------+-------------+--------+------- --------+------+-----+-------------
    Yolov8u        |      Y      |             |   Y    |  Y         Y   |  Y   |  Y  | Y list(int) 
    """

    ret, response = init(url="http://0.0.0.0:60000")
    print("初始化结果", ret, response)

    engine = Yolov8u(weight_path="/hky3535/test/yolov8s.pt", device="gpu", half=False)

    ret, response = engine.load()
    print("加载结果", ret, response)

    frame = cv2.imread("/hky3535/test/source/000000000110.jpg")
    ret, response = engine.infer(frame=frame, conf=0.25, iou=0.7, classes=False)
    print("推理结果", ret, response)
    for result in response: frame = draw(frame, result)
    cv2.imwrite("/hky3535/test/source/_000000000110.jpg", frame)
    

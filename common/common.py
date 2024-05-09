"""hekaiyue 何恺悦 2024-01-14"""
from .engines import engines

import cv2
import numpy
import base64

import tempfile
import shutil
import atexit


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


class Common:
    def __init__(self):
        # 静态框架 {engine_name: engine_handle} 驱动器名称 驱动器句柄
        self.engines = dict()
        self.engines.update(engines)
        # 动态加载 {engine_name: {engine_task: engine_loaded}} 驱动器名称 驱动器任务 驱动器加载
        self.loaded = {engine_name: {} for engine_name in self.engines}

        # 全局 临时目录 启动时创建 退出时删除
        self.temp_path = False
        self.at_enter_exit()

    def at_enter_exit(self):
        # 启动时创建
        self.temp_path = tempfile.mkdtemp()
        # 退出时删除
        def clean_temp():
            if self.temp_path is not False:
                shutil.rmtree(self.temp_path)
        atexit.register(clean_temp)

    def download(self, file):
        # 判断是否可以解析为字典
        if not isinstance(file, dict): return file
        # 判断是否为需要解析的结构
        if "name" not in file or "b64" not in file: return file
        # 如果字典结构为 {name: str, b64: str}
        file_name, file_b64 = file["name"], file["b64"]
        file_b = b64_b(file_b64)
        file_path = f"{self.temp_path}/{file_name}"
        open(file_path, "wb").write(file_b)
        return file_path

    def view(self, frame):
        # 判断是否可以解析为字典
        if not isinstance(frame, dict): return frame
        # 判断是否为需要解析的结构
        if "b64" not in frame: return frame
        # 如果字典结构为 {b64: str}
        frame_b64 = frame["b64"]
        frame_cv = b64_cv(frame_b64=frame_b64)
        return frame_cv

    def status(self):
        return {_: list(__.keys()) for _, __ in self.loaded.items()}

    def load(self, data):
        # 加载固定参数
        engine_name = data["engine_name"]
        engine_task = data["engine_task"]
        arguments = data["arguments"]
        # 检查重复加载
        if engine_task in self.status()[engine_name]: return
        # 遍历加载参数 --> 判断是否需要下载 -True-> 下载并返回路径
        for argument in arguments:
            arguments[argument] = self.download(file=arguments[argument])
        # 加载（非保障调用）
        self.loaded[engine_name][engine_task] = self.engines[engine_name](**arguments)

    def infer(self, data):
        # 加载固定参数
        engine_name = data["engine_name"]
        engine_task = data["engine_task"]
        arguments = data["arguments"]
        # 检查是否加载
        if engine_task not in self.status()[engine_name]: return
        # 遍历推理参数 --> 判断是否需要查看/读取 -True-> 查看并返回图片/读取并返回数组
        for argument in arguments:
            arguments[argument] = self.view(frame=arguments[argument])
        # 推理（非保障调用）（返回值为list()）
        return self.loaded[engine_name][engine_task].infer(**arguments)


"""hekaiyue 何恺悦 2024-01-23"""
class Engine:
    def __init__(self, iou_thresh=0.5, buffer_size=5):
        self.iou_thresh = iou_thresh # 判断为同一目标的阈值
        self.buffer_size = buffer_size # 缓存大小
        self.buffer = list() # 缓存
        self.id = 0 # 追踪id

    def iou(self, box0, box1):
        # 计算交集的左上角和右下角坐标
        x0 = max(box0[0], box1[0])
        y0 = max(box0[1], box1[1])
        x1 = min(box0[2], box1[2])
        y1 = min(box0[3], box1[3])
        # 计算交集的宽度和高度
        intersection_width = max(0, x1 - x0)
        intersection_height = max(0, y1 - y0)
        # 计算交集面积
        intersection_area = intersection_width * intersection_height
        # 计算并集面积
        box0_area = (box0[2] - box0[0]) * (box0[3] - box0[1])
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        union_area = box0_area + box1_area - intersection_area
        if union_area <= 0: return 0
        # 计算交并比
        iou = intersection_area / union_area
        return iou

    def infer(self, results):
        # [x0, y0, x1, y1, ...] --> [x0, y0, x1, y1, id, ...]
        now_ids = list()
        new_ids = list()
        buffer = [_ for __ in self.buffer for _ in __] # 把缓存的三维列表展开成一维列表 [[[x0, y0, x1, y1, ...], ...], ...] 
        for current_target in results: # 遍历当前帧中所有目标
            current_id = False
            for history_target in buffer: # 遍历缓存帧中所有目标
                if current_target[-1] == history_target[-1] and self.iou(current_target, history_target) > self.iou_thresh: # 如果 当前帧中的该目标 匹配到了 缓存帧中出现过的该目标
                    current_id = history_target[4] # 则 当前帧中的该目标 继承 缓存帧中出现过的该目标 的 追踪id
                    if current_id in now_ids: current_id = False # 如果已经被匹配掉了则不允许重复匹配
                    break
            if current_id is False: # 如果 当前帧中的该目标 未匹配到 缓存帧中出现过的任何目标
                self.id += 1
                current_id = self.id # 申请并赋予一个新id
                now_ids.append(current_id)
                current_target.insert(4, current_id)
                new_ids.append(current_id) # 判定为新目标
            else: # 如果 当前帧中的该目标 匹配到 缓存帧中出现过的某个目标
                now_ids.append(current_id)
                current_target.insert(4, current_id) # 直接赋予继承到的 追踪id

        end_ids = list()
        self.buffer.append(results) # 将处理完成的当前帧推入缓存中
        if len(self.buffer) > self.buffer_size: # 如果缓存超长
            abort_buffer = self.buffer[0] # 获取头帧
            self.buffer = self.buffer[1:] # 抛弃头帧

            history_ids = [_[4] for __ in self.buffer for _ in __] # 查找并展平剩余buffer中所有帧中所有目标的id
            for abort_target in abort_buffer: # 遍历抛弃帧中所有被抛弃的目标
                abort_id = abort_target[4]
                if abort_id not in history_ids: # 如果被抛弃的目标的id没有再被继承的可能（连续buffer_size帧没有被继承过）
                    end_ids.append(abort_id) # 判定为消失目标
                    break
        
        return [results, now_ids, new_ids, end_ids]


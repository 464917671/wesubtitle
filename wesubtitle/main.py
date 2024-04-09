# -*- encoding: utf-8 -*-
import json
import os
import sys

print('\n'.join(f'获得 python 环境变量： {i}' for i in sys.path))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前 python 文件的目录
print('获得当前 python 执行文件绝对路径:', os.path.realpath(__file__))  # 获得当前python执行文件绝对路径

import argparse
import copy
import datetime

import cv2
from paddleocr import PaddleOCR
from skimage.metrics import structural_similarity
import srt


# python3 -m pip uninstall paddlepaddle PaddleOCR
# python3 -m pip install paddlepaddle PaddleOCR opencv-python srt -i https://pypi.tuna.tsinghua.edu.cn/simple
# python3 ~/Documents/Gitee/myproject/PycharmProjects/we_subtitle/we_subtitle/main.py ~/Downloads/罗翔大熊猫.mp4

class Move2Srt:
    def __init__(self):
        self.args = None
        self.output_srt = None

        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch")

        self.cap = None
        self.w = None
        self.h = None
        self.count = None
        self.fps = None

        self.cur = 0
        self.detected = False
        self.box = None
        self.content = ''
        self.start = 0
        self.ref_gray_image = None
        self.subs = []

        self.progress_file = None

    def box2int(self, box):
        # 将框中的坐标转换为整数
        for i in range(len(box)):
            for j in range(len(box[i])):
                box[i][j] = int(box[i][j])
        return box

    def detect_subtitle_area(self, ocr_results, h, w):
        '''
        Args:
            w(int): 视频的宽度
            h(int): 视频的高度
        '''
        ocr_results = ocr_results[0]  # 0，第一张图片的结果

        if ocr_results is None or not ocr_results:
            print("返回 False, None, None")
            return False, None, None

        # 合并水平文本区域
        idx = 0
        candidates = []
        while idx < len(ocr_results):
            boxes, text = ocr_results[idx]
            # 假设字幕在视频底部
            if boxes[0][1] < h * 0.75:
                idx += 1
                continue
            idx += 1
            con_boxes = copy.deepcopy(boxes)
            con_text = text[0]
            while idx < len(ocr_results):
                n_boxes, n_text = ocr_results[idx]
                if abs(n_boxes[0][1] - boxes[0][1]) < h * 0.01 and \
                        abs(n_boxes[3][1] - boxes[3][1]) < h * 0.01:
                    con_boxes[1] = n_boxes[1]
                    con_boxes[2] = n_boxes[2]
                    con_text = con_text + ' ' + n_text[0]
                    idx += 1
                else:
                    break
            candidates.append((con_boxes, con_text))
        # TODO(Binbin Zhang): 目前只支持水平居中的字幕
        if len(candidates) > 0:
            sub_boxes, subtitle = candidates[-1]
            # 偏移量小于10%
            if (sub_boxes[0][0] + sub_boxes[1][0]) / w > 0.90:
                return True, self.box2int(sub_boxes), subtitle
        return False, None, None

    def get_args(self):
        parser = argparse.ArgumentParser(description='we subtitle')
        parser.add_argument('-s',
                            '--subsampling',
                            type=int,
                            default=3,
                            help='采样率，用于加速')
        parser.add_argument('-t',
                            '--similarity_thresh',
                            type=float,
                            default=0.8,
                            help='相似度阈值')

        parser.add_argument('input_video', help='输入视频文件')

        self.args = parser.parse_args()
        # 根据输入视频名称自动设置输出SRT文件
        self.output_srt = os.path.splitext(self.args.input_video)[0] + '.srt'
        self.args.output_srt = self.output_srt

        self.progress_file = os.path.splitext(self.args.input_video)[0] + '_progress.json'

        return self.args, self.output_srt

    def subtitle_to_dict(self, sub):
        return {
            'index': sub.index,
            'start': sub.start.total_seconds(),
            'end': sub.end.total_seconds(),
            'content': sub.content
        }

    def save_progress(self, progress_file, cur_frame, subtitles):
        # 保存进度到文件
        progress = {
            'current_frame': cur_frame,
            'subs': [self.subtitle_to_dict(sub) for sub in subtitles]
        }
        with open(progress_file, 'w', encoding='utf8') as pf:
            json.dump(progress, pf, ensure_ascii=False)
        print(f'进度已保存到文件: {progress_file} {cur_frame}')

    def resume_progress(self, progress_file):
        # 初始化 视频参数
        self.cap = cv2.VideoCapture(self.args.input_video)

        # 从进度文件加载
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf8') as pf:
                progress = json.load(pf)
                cur_frame = progress.get('current_frame', 0)
                subtitles = [
                    srt.Subtitle(
                        index=sub_data['index'],
                        start=datetime.timedelta(seconds=sub_data['start']),
                        end=datetime.timedelta(seconds=sub_data['end']),
                        content=sub_data['content']
                    )
                    for sub_data in progress.get('subs', [])
                ]
                print(f"读取处理进度：{cur_frame} {subtitles[-1]}")
                self.subs = subtitles
                self.cur = cur_frame  # 确保cur设置为加载的进度

            # 初始化 视频参数
            if self.cur > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur)

        # 初始化 视频参数
        self.w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print('视频信息 宽度: {}, 高度: {}, 总帧数: {}, 帧率: {}'.format(
            self.w, self.h, self.count, self.fps))
        return 0, []

    # 将实时保存SRT部分的代码改为以下形式
    def _save_srt_file(self, args, subs):
        srt_content = srt.compose(subs)
        if not srt_content.strip():
            print(f'警告: 没有要写入文件的字幕文本.')
        else:
            with open(args.output_srt, 'w', encoding='utf8') as fout:
                fout.write(srt_content)
                print(f'实时保存 SRT :{subs[-1]}')
                print(f'实时保存 SRT 完成. 字幕数量: {len(subs)}')

    # 在 _add_subs 函数中增加保存字幕 和 进度信息
    def _add_subs(self, args, cur, end, start, fps, content, subs, progress_file):
        print('新增字幕 {} {} {}'.format(start / fps, cur / fps, content))
        new_sub = srt.Subtitle(
            index=len(subs) + 1,  # 设置序号
            start=datetime.timedelta(seconds=start / fps),
            end=datetime.timedelta(seconds=end / fps),
            content=content.strip(),
        )
        subs.append(new_sub)
        print(f'添加新字幕: {new_sub}')  # 打印新增的字幕

        self.save_progress(progress_file, cur, subs)  # 在这里保存当前进度
        self._save_srt_file(args, subs)  # 调用保存 SRT 文件函数


# 主函数
def main():
    move2srt = Move2Srt()
    # 初始化媒体文件及 SRT 路劲
    args, output_srt = move2srt.get_args()

    # 进度文件
    progress_file = move2srt.progress_file
    # 载入视频进度
    move2srt.resume_progress(progress_file)

    ocr = move2srt.ocr

    cap = move2srt.cap
    w = move2srt.w
    h = move2srt.h
    count = move2srt.count
    fps = move2srt.fps

    cur = move2srt.cur
    detected = move2srt.detected
    box = move2srt.box
    content = move2srt.content
    start = move2srt.start
    ref_gray_image = move2srt.ref_gray_image
    subs = move2srt.subs

    print(f"使用视频进度：{cap.get(cv2.CAP_PROP_POS_FRAMES)}")
    try:
        while cap.isOpened():

            if cur < start:
                cur += 1
                cap.read()
                continue
            # 读取媒体文件 ret 是否返回帧  frame 返回 视频 信息 为 列表
            ret, frame = cap.read()
            # 如果返回 False 则 把识别的字母 追加到 subs
            if not ret:
                if detected:
                    move2srt._add_subs(args, cur, cur, start, fps, content, subs, progress_file)
                break
            cur += 1
            # 如果不到视频结尾 则继续
            if cur % args.subsampling != 0:
                continue

            # 每 n 帧保存字幕进度
            if cur % 10 == 0:
                move2srt.save_progress(progress_file, cur, subs)

            if detected:
                # 计算与参考字幕区域的相似度，如果结果大于阈值，则为相同字幕，否则字幕区域发生变化
                hyp_gray_image = frame[box[1][1]:box[2][1], box[0][0]:box[1][0], :]
                hyp_gray_image = cv2.cvtColor(hyp_gray_image, cv2.COLOR_BGR2GRAY)
                similarity = structural_similarity(hyp_gray_image, ref_gray_image)
                if similarity > args.similarity_thresh:  # 相同字幕
                    continue
                else:
                    # 记录当前字幕 args, cur - args.subsampling, start, fps, content, subs, progress_file
                    move2srt._add_subs(args, cur, cur - args.subsampling, start, fps, content, subs, progress_file)
                    detected = False
            else:
                # 识别字幕区域
                ocr_results = ocr.ocr(frame)
                # 打印识别结果
                # print(ocr_results)
                detected, box, content = move2srt.detect_subtitle_area(ocr_results, h, w)
                if detected:
                    start = cur
                    ref_gray_image = frame[box[1][1]:box[2][1],
                                     box[0][0]:box[1][0], :]
                    ref_gray_image = cv2.cvtColor(ref_gray_image,
                                                  cv2.COLOR_BGR2GRAY)

        # 释放视频捕获对象
        cap.release()

        # 保存最终的SRT文件
        print(f'{datetime.datetime.now()} 保存完整字幕...')
        with open(args.output_srt, 'w', encoding='utf8') as fout:
            fout.write(srt.compose(subs))

        # 成功完成后清除进度文件
        print(f'{datetime.datetime.now()} 清除进度文件...')
        if os.path.exists(progress_file):
            os.remove(progress_file)

    except KeyboardInterrupt:
        # 保存当前进度
        move2srt.save_progress(progress_file, cur, subs)
        print("程序被手动终止")
    except Exception as e:
        # 保存当前进度
        move2srt.save_progress(progress_file, cur, subs)
        print(f"程序出现异常: {str(e)}")


# 加载外部执行参数和进度恢复逻辑
if __name__ == '__main__':
    main()

import cv2
import numpy as np
from filter import Filter

class DashboardDetect:
    def __init__(self, show_image=False):
        self.result = None
        self.show_image = show_image
        self.filter = Filter(10)

    def clear_filter(self):
        self.filter.clear()

    def detect(self, frame, mark_center, number, color):
        # 二值化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 2)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, None, iterations=2)

        if self.show_image:
            cv2.imshow('binary', binary)

        # 开闭运算、反色
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        bin_invert = cv2.bitwise_not(binary)

        if self.show_image:
            cv2.imshow('bin_invert', bin_invert)

        # 以图片中心为圆心，最小边长的一半为半径画圆，圆 70% 以外的部分置白色
        h, w = frame.shape[:2]
        r = min(h, w) // 2 - 0.3 * min(h, w)
        for i in range(h):
            for j in range(w):
                if (i - h // 2) ** 2 + (j - w // 2) ** 2 > r ** 2:
                    bin_invert[i, j] = 0

        if self.show_image:
            cv2.imshow('bin_invert_fill', bin_invert)

        contours, _ = cv2.findContours(bin_invert, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        pointer_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        # 寻找轮廓上离 图像 中心最远的点
        max_distance = 0
        max_distance_point = (w // 2, h // 2)
        for point in pointer_contour:
            point = point[0]
            distance = (point[0] - w // 2) ** 2 + (point[1] - h // 2) ** 2
            if distance > max_distance:
                max_distance = distance
                max_distance_point = (point[0], point[1])
        cv2.line(frame, max_distance_point, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), (0, 255, 0), 2)

        cv2.line(frame, mark_center, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), (0, 255, 0), 2)

        # 判断 3 点夹角
        # 定义点A、B、C的坐标
        A = np.array(max_distance_point)
        B = np.array([frame.shape[1]/2, frame.shape[0]/2])
        C = np.array(mark_center)

        # 计算向量BA和BC
        BA = A - B
        BC = C - B

        # 计算向量的点积
        dot_product = np.dot(BA, BC)

        # 计算向量的模长
        norm_BA = np.linalg.norm(BA)
        norm_BC = np.linalg.norm(BC)

        # 计算夹角的余弦值
        cos_theta = dot_product / (norm_BA * norm_BC)

        # 计算夹角（以弧度为单位），并确保余弦值在合法范围内
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        # 计算向量的叉积来确定旋转方向
        cross_product = np.cross(BA, BC)

        # 根据叉积的符号调整角度的正负
        # 如果叉积为负，说明 BC 到 BA 是顺时针方向，角度不变
        # 如果叉积为正，说明 BC 到 BA 是逆时针方向，需要从 360 度中减去角度
        if cross_product > 0:
            theta_deg = 360 - np.degrees(theta_rad)
        else:
            theta_deg = np.degrees(theta_rad)

        if 30 <= theta_deg <= 130:
            self.result = "low"
        elif 130 <= theta_deg <= 230:
            self.result = "normal"
        elif 230 <= theta_deg <= 340:
            self.result = "high"
        cv2.putText(frame, self.result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.show_image:
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                exit(0)

        return self.filter.filter((self.result, number, color))

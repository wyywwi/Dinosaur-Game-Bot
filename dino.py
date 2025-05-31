import cv2
import numpy as np
import pyautogui
import time
import keyboard
import mss
from collections import deque
import statistics

class DinosaurGameBot:
    def __init__(self):
        # 显示程序说明
        self.print_instructions()
        
        # 初始化变量
        self.dino_pos = None
        self.game_region = None
        self.detection_region = None
        self.calibration_complete = False
        self.game_started = False
        self.obstacles = []
        self.last_jump_time = 0
        self.start_time = time.time()
        
        # 真实速度测量系统
        self.speed_measurements = deque(maxlen=10)  # 保存最近10次速度测量
        self.current_speed = 0.0
        self.last_speed_measurement = 0
        self.reference_obstacles = deque(maxlen=5)  # 用于速度测量的参考障碍物
        self.speed_calibrated = False
        
        # 跳跃时机
        self.optimal_lead_time = 0.32  # 初始提前时间，会被动态调整
        self.last_action_time = 0
        
        # 恐龙状态跟踪
        self.dino_state = "running"  # running, jumping, ducking
        self.dino_rect = None  # 存储检测到的恐龙框
        self.dino_height_threshold = 40  # 恐龙跳跃时的高度变化阈值
        
        # 游戏结束检测系统
        self.prev_obstacles = deque(maxlen=5)  # 保存最近5帧的障碍物位置
        self.game_over_counter = 0  # 游戏结束计数器
        self.last_restart_time = 0  # 上次重启时间
        
        # 游戏参数（基于实际测量调整）
        self.MIN_SPEED_SAMPLES = 3  # 最少需要的速度样本数
        self.SPEED_MEASUREMENT_INTERVAL = 0.05  # 速度测量间隔
        self.JUMP_COOLDOWN = 0.3  # 跳跃冷却时间
        self.GAME_OVER_THRESHOLD = 5  # 连续多少帧障碍物不动判定为游戏结束
        
        # 图像处理参数
        self.OBSTACLE_THRESHOLD = 100
        self.MIN_OBSTACLE_WIDTH = 8
        self.MIN_OBSTACLE_HEIGHT = 12
        self.DINO_THRESHOLD = 100  # 恐龙检测阈值
        
        # 检测区域参数
        self.DETECTION_WIDTH = 400
        self.DETECTION_HEIGHT = 80
        self.DETECTION_X_OFFSET = 60
        self.DETECTION_Y_OFFSET = -30
        
        # 恐龙检测区域
        self.DINO_DETECTION_WIDTH = 60
        self.DINO_DETECTION_HEIGHT = 80
    
    def print_instructions(self):
        """打印程序使用说明"""
        print("=" * 60)
        print("Google Dinosaur Game 脚本")
        print("=" * 60)
        print("使用说明:")
        print("1. 打开Chrome浏览器，断开网络连接并访问任意网页")
        print("2. 出现小恐龙游戏后，运行此程序")
        print("3. 按 'C' 键进入校准模式")
        print("4. 在校准模式下，点击恐龙头顶位置")
        print("5. 校准完成后，按 'S' 键开始自动游戏")
        print("6. 程序会自动测量游戏速度并优化跳跃时机")
        print("7. 按 'Q' 键退出程序")
        print("8. 按 'R' 键重新校准")
        print("=" * 60)
    
    def calibrate(self):
        """用户校准模式：让用户点击恐龙头顶位置"""
        print("进入校准模式...")
        print("请将鼠标移动到恐龙头顶位置，然后按空格键确认")
        
        calibration_window = "校准窗口 (按空格确认位置)"
        # 创建全屏窗口
        cv2.namedWindow(calibration_window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(calibration_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        screenshot = pyautogui.screenshot()
        screen_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        cv2.imshow(calibration_window, screen_img)
        cv2.waitKey(1)
        
        selected_pos = None
        while selected_pos is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                mouse_x, mouse_y = pyautogui.position()
                selected_pos = (mouse_x, mouse_y)
                print(f"已选择位置: {selected_pos}")
            
            temp_img = screen_img.copy()
            if pyautogui.position() != (0, 0):
                mx, my = pyautogui.position()
                cv2.circle(temp_img, (mx, my), 10, (0, 0, 255), 2)
                cv2.putText(temp_img, f"({mx}, {my})", (mx+15, my-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(calibration_window, temp_img)
            
            if key == 27:
                break
        
        cv2.destroyAllWindows()
        
        if selected_pos:
            self.dino_pos = selected_pos
            self.game_region = {
                'top': max(0, self.dino_pos[1] - 100),
                'left': max(0, self.dino_pos[0] - 200),
                'width': 800,
                'height': 200
            }
            
            region_x = self.dino_pos[0] + self.DETECTION_X_OFFSET
            region_y = self.dino_pos[1] + self.DETECTION_Y_OFFSET
            
            self.detection_region = {
                'x': region_x - self.game_region['left'],
                'y': region_y - self.game_region['top'],
                'width': self.DETECTION_WIDTH,
                'height': self.DETECTION_HEIGHT
            }
            
            print(f"恐龙位置: {self.dino_pos}")
            print(f"游戏区域: {self.game_region}")
            print(f"检测区域: {self.detection_region}")
            
            self.show_calibration_result(screen_img)
            self.calibration_complete = True
            
            # 重置速度测量系统
            self.speed_measurements.clear()
            self.reference_obstacles.clear()
            self.speed_calibrated = False
            self.current_speed = 0.0
            
            return True
        
        print("校准取消")
        return False
    
    def show_calibration_result(self, screen_img):
        """显示校准结果（全屏显示）"""
        result_img = screen_img.copy()
        
        cv2.circle(result_img, self.dino_pos, 10, (0, 0, 255), -1)
        cv2.putText(result_img, "Dino Head", 
                   (self.dino_pos[0] + 15, self.dino_pos[1] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        rx, ry, rw, rh = (
            self.game_region['left'], 
            self.game_region['top'], 
            self.game_region['width'], 
            self.game_region['height']
        )
        cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
        cv2.putText(result_img, "Game Area", 
                   (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        detect_x = self.dino_pos[0] + self.DETECTION_X_OFFSET
        detect_y = self.dino_pos[1] + self.DETECTION_Y_OFFSET
        cv2.rectangle(result_img, 
                      (detect_x, detect_y),
                      (detect_x + self.DETECTION_WIDTH, 
                       detect_y + self.DETECTION_HEIGHT), 
                      (0, 255, 0), 2)
        cv2.putText(result_img, "Detection Area", 
                   (detect_x, detect_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 创建全屏窗口
        cv2.namedWindow("校准结果 (按ESC关闭)", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("校准结果 (按ESC关闭)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("校准结果 (按ESC关闭)", result_img)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if cv2.getWindowProperty("校准结果 (按ESC关闭)", cv2.WND_PROP_VISIBLE) < 1:
                break
            
        cv2.destroyAllWindows()
    
    def start_game(self):
        """开始游戏"""
        if not self.game_started:
            print("游戏开始！")
            print("正在测量游戏速度，请等待...")
            self.game_started = True
            self.start_time = time.time()
            self.last_speed_measurement = time.time()
            pyautogui.press('space')
            
            # 重置游戏结束检测
            self.game_over_counter = 0
            self.prev_obstacles.clear()
    
    def restart_game(self):
        """重新开始游戏"""
        print("检测到游戏结束，重新开始游戏...")
        # 按两次空格键重启游戏
        pyautogui.press('space')
        time.sleep(0.2)
        pyautogui.press('space')
        
        # 重置游戏状态
        self.game_started = False
        self.obstacles = []
        self.last_jump_time = 0
        self.optimal_lead_time = 0.32
        self.start_time = time.time()
        
        # 重置速度测量系统
        self.speed_measurements.clear()
        self.reference_obstacles.clear()
        self.speed_calibrated = False
        self.current_speed = 0.0
        
        # 重置游戏结束检测
        self.game_over_counter = 0
        self.prev_obstacles.clear()
        self.last_restart_time = time.time()
        
        # 重新开始游戏
        time.sleep(0.5)
        self.start_game()
    
    def check_game_over(self, current_obstacles):
        """检测游戏是否结束"""
        current_time = time.time()
        
        # 避免刚重启后立即检测
        if current_time - self.last_restart_time < 2.0:
            return False
        
        # 如果没有障碍物，不进行检测
        if not current_obstacles:
            return False
        
        # 保存当前障碍物位置
        self.prev_obstacles.append(current_obstacles)
        
        # 需要至少2帧才能比较
        if len(self.prev_obstacles) < 2:
            return False
        
        # 比较最近两帧的障碍物位置
        prev_frame = self.prev_obstacles[-2]
        curr_frame = self.prev_obstacles[-1]
        
        # 检查是否有移动的障碍物
        moving_obstacles = False
        
        for curr_obs in curr_frame:
            for prev_obs in prev_frame:
                # 如果找到相同类型的障碍物且位置有变化
                if (curr_obs['type'] == prev_obs['type'] and
                    abs(curr_obs['x'] - prev_obs['x']) > 2 and
                    abs(curr_obs['y'] - prev_obs['y']) < 5):
                    moving_obstacles = True
                    break
            if moving_obstacles:
                break
        
        # 如果没有检测到移动的障碍物，增加计数器
        if not moving_obstacles:
            self.game_over_counter += 1
            if self.game_over_counter >= self.GAME_OVER_THRESHOLD:
                return True
        else:
            self.game_over_counter = 0
            
        return False
    
    def capture_screen(self):
        """捕获游戏区域屏幕"""
        with mss.mss() as sct:
            monitor = {
                "top": self.game_region['top'],
                "left": self.game_region['left'],
                "width": self.game_region['width'],
                "height": self.game_region['height']
            }
            screen = np.array(sct.grab(monitor))
            return cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)
    
    def detect_obstacles(self, screen):
        """检测障碍物"""
        det_x = self.detection_region['x']
        det_y = self.detection_region['y']
        det_w = self.detection_region['width']
        det_h = self.detection_region['height']
        
        detection_area = screen[
            det_y:det_y + det_h,
            det_x:det_x + det_w
        ]
        
        _, thresh = cv2.threshold(detection_area, self.OBSTACLE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        current_time = time.time()
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= self.MIN_OBSTACLE_WIDTH and h >= self.MIN_OBSTACLE_HEIGHT:
                abs_x = x + det_x
                abs_y = y + det_y
                
                obstacle = {
                    'x': abs_x,
                    'y': abs_y,
                    'width': w,
                    'height': h,
                    'type': self.classify_obstacle(y, h, w, self.detection_region['height']),
                    'timestamp': current_time,
                    'id': f"{abs_x}_{abs_y}_{current_time}"  # 唯一标识符
                }
                
                obstacles.append(obstacle)
        
        return obstacles
    
    def classify_obstacle(self, y, height, width, region_height):
        """根据尺寸和位置分类障碍物"""
        # 计算障碍物底部位置（在检测区域中的相对位置）
        bottom_position = y + height
        
        # 地面障碍物（仙人掌）特征：底部接近检测区域底部
        ground_threshold = region_height - 15  # 距离底部15像素内视为地面障碍物
        
        if bottom_position > ground_threshold:
            return "cactus"
        else:
            # 飞鸟分类：根据在检测区域中的高度位置
            if y < region_height * 0.2:  # 在检测区域上半部分
                return "bird_high"
            else:  # 在检测区域下半部分
                return "bird_low"

    def detect_dinosaur(self, screen):
        """直接检测恐龙框并确定状态"""
        # 计算恐龙检测区域
        dino_x = self.dino_pos[0] - self.game_region['left']
        dino_y = self.dino_pos[1] - self.game_region['top']
        
        # 定义检测区域：以定位点为中心
        start_x = max(0, dino_x - self.DINO_DETECTION_WIDTH // 2)
        start_y = max(0, dino_y - self.DINO_DETECTION_HEIGHT // 2)
        end_x = min(screen.shape[1], dino_x + self.DINO_DETECTION_WIDTH // 2)
        end_y = min(screen.shape[0], dino_y + self.DINO_DETECTION_HEIGHT // 2)
        
        dino_roi = screen[start_y:end_y, start_x:end_x]
        
        if dino_roi.size == 0:
            return "unknown", None
        
        # 二值化处理
        _, thresh = cv2.threshold(dino_roi, self.DINO_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果没有找到轮廓，返回未知状态
        if not contours:
            return "unknown", None
        
        # 找到最大的轮廓（应该是恐龙）
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 转换为游戏区域绝对坐标
        abs_x = x + start_x
        abs_y = y + start_y
        self.dino_rect = (abs_x, abs_y, w, h)
        
        # 计算恐龙框顶部与定位点的垂直距离
        top_distance = abs_y - dino_y
        
        # 根据距离判断状态
        if top_distance < -15:  # 恐龙框顶部远高于定位点
            return "jumping", self.dino_rect
        elif top_distance > 15:  # 恐龙框顶部低于定位点
            return "ducking", self.dino_rect
        else:  # 大致齐平
            return "running", self.dino_rect
    
    def measure_speed(self, current_obstacles):
        """测量游戏速度"""
        current_time = time.time()
        
        # 检查是否到了测量时间
        if current_time - self.last_speed_measurement < self.SPEED_MEASUREMENT_INTERVAL:
            return
        
        # 尝试匹配之前的障碍物来计算速度
        for current_obs in current_obstacles:
            for ref_obs in self.reference_obstacles:
                # 通过位置和尺寸匹配障碍物（允许一定误差）
                if (abs(current_obs['y'] - ref_obs['y']) < 5 and 
                    abs(current_obs['width'] - ref_obs['width']) < 3 and
                    abs(current_obs['height'] - ref_obs['height']) < 3 and
                    current_obs['x'] < ref_obs['x']):  # 障碍物应该向左移动
                    
                    # 计算速度
                    distance_moved = ref_obs['x'] - current_obs['x']  # 像素
                    time_elapsed = current_time - ref_obs['timestamp']  # 秒
                    
                    if time_elapsed > 0 and distance_moved > 0:
                        speed = distance_moved / time_elapsed  # 像素/秒
                        
                        # 过滤异常值（速度应该在合理范围内）
                        if 400 < speed < 600:
                            self.speed_measurements.append(speed)
                            print(f"🎯 测量到速度: {speed:.1f} 像素/秒 (距离:{distance_moved:.1f}px, 时间:{time_elapsed:.3f}s)")
                            
                            # 更新当前速度（使用最近几次测量的平均值）
                            if len(self.speed_measurements) >= self.MIN_SPEED_SAMPLES:
                                self.current_speed = statistics.median(self.speed_measurements)
                                if not self.speed_calibrated:
                                    self.speed_calibrated = True
                                    print(f"✅ 速度校准完成！当前速度: {self.current_speed:.1f} 像素/秒")
                            
                            break
        
        # 更新参考障碍物列表
        self.reference_obstacles.clear()
        for obs in current_obstacles:
            self.reference_obstacles.append(obs.copy())
        
        self.last_speed_measurement = current_time
    
    def calculate_collision_time(self, obstacle):
        """计算障碍物到达恐龙的时间"""
        if not self.speed_calibrated or self.current_speed <= 0:
            return float('inf')  # 如果速度未校准，返回无穷大
        
        dino_x = self.dino_pos[0] - self.game_region['left']
        distance_to_dino = obstacle['x'] - dino_x
        
        if distance_to_dino <= 0:
            return 0  # 障碍物已经到达或超过恐龙
        
        collision_time = distance_to_dino / self.current_speed
        return collision_time
    
    def should_jump(self, obstacle):
        """判断是否应该跳跃"""
        collision_time = self.calculate_collision_time(obstacle)
        
        # 根据障碍物类型和当前优化的提前时间判断
        if obstacle['type'] in ["cactus", "bird_low"]:
            return collision_time <= self.optimal_lead_time
        elif obstacle['type'] == "bird_high":
            # 对于高飞鸟，我们选择下蹲而不是跳跃
            return False
        
        return False
    
    def should_duck(self, obstacle):
        """判断是否应该下蹲"""
        collision_time = self.calculate_collision_time(obstacle)
        
        if obstacle['type'] == "bird_high":
            return collision_time <= self.optimal_lead_time
        
        return False
    
    def jump(self):
        """让恐龙跳跃"""
        if self.dino_state == "running":
            pyautogui.press('space')
            self.last_jump_time = time.time()
            self.last_action_time = time.time()
            print(f"🦘 跳跃！(速度: {self.current_speed:.1f} px/s, 提前时间: {self.optimal_lead_time:.2f}s)")
            return True
        return False
    
    def duck(self):
        """让恐龙下蹲"""
        if self.dino_state == "running":
            pyautogui.keyDown('down')
            time.sleep(0.15)
            pyautogui.keyUp('down')
            self.last_jump_time = time.time()
            self.last_action_time = time.time()
            print(f"🦆 下蹲！(速度: {self.current_speed:.1f} px/s, 提前时间: {self.optimal_lead_time:.2f}s)")
            return True
        return False
    
    def optimize_timing(self):
        """优化跳跃时机（简单的自适应调整）"""
        # 这里可以根据游戏反馈来调整optimal_lead_time
        # 例如：如果经常撞到障碍物，增加提前时间
        # 如果经常过早跳跃，减少提前时间
        
        # 简单实现：根据游戏时间动态调整
        game_time = time.time() - self.start_time
        if game_time > 10:  # 游戏10秒后，速度更快，需要更多提前时间
            self.optimal_lead_time = min(0.6, self.optimal_lead_time + 0.001)
    
    def take_action(self):
        """根据障碍物情况采取行动"""
        if not self.obstacles or not self.speed_calibrated:
            return
        
        # 找到最近的障碍物
        closest_obstacle = min(self.obstacles, key=lambda o: o['x'])
        collision_time = self.calculate_collision_time(closest_obstacle)
        
        # 显示详细信息
        dino_x = self.dino_pos[0] - self.game_region['left']
        distance = closest_obstacle['x'] - dino_x
        
        if collision_time < 0.5:  # 只显示即将到达的障碍物信息
            print(f"📍 最近障碍物: {closest_obstacle['type']}, "
                  f"距离: {distance:.0f}px, "
                  f"预计到达: {collision_time:.2f}s, "
                  f"当前速度: {self.current_speed:.1f}px/s")
        
        # 根据计算结果采取行动
        if self.should_jump(closest_obstacle):
            self.jump()
        elif self.should_duck(closest_obstacle):
            self.duck()
    
    def visualize(self, screen, dino_state):
        """可视化检测结果"""
        # 转换为彩色图像用于显示
        if len(screen.shape) == 2:
            screen = cv2.cvtColor(screen, cv2.COLOR_GRAY2BGR)
        
        # 绘制恐龙位置
        dino_x = self.dino_pos[0] - self.game_region['left']
        dino_y = self.dino_pos[1] - self.game_region['top']
        
        # 绘制原始定位点
        cv2.circle(screen, (dino_x, dino_y), 5, (0, 0, 255), -1)
        cv2.putText(screen, "Dino Point", 
                   (dino_x + 10, dino_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制恐龙框（如果检测到）
        if self.dino_rect:
            x, y, w, h = self.dino_rect
            cv2.rectangle(screen, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # 绘制状态指示线
            cv2.line(screen, (x, dino_y), (x+w, dino_y), (0, 0, 255), 1)  # 定位点水平线
            cv2.line(screen, (x, y), (x, dino_y), (255, 0, 0), 1)  # 顶部到定位点的垂直线
            
            # 显示距离信息
            distance = y - dino_y
            cv2.putText(screen, f"Top Dist: {distance}", 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 显示恐龙状态
        state_colors = {
            "running": (100, 200, 100),
            "jumping": (200, 100, 100),
            "ducking": (100, 100, 200),
            "unknown": (200, 200, 200)
        }
        color = state_colors.get(dino_state, (200, 200, 200))
        cv2.putText(screen, f"Dino State: {dino_state}", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制检测区域
        det_x = self.detection_region['x']
        det_y = self.detection_region['y']
        det_w = self.detection_region['width']
        det_h = self.detection_region['height']
        cv2.rectangle(screen, 
                      (det_x, det_y),
                      (det_x + det_w, det_y + det_h), 
                      (100, 100, 100), 1)
        
        # 绘制障碍物和碰撞时间
        for obstacle in self.obstacles:
            color = (0, 0, 255)  # 红色 - 仙人掌
            if obstacle['type'] == "bird_low":
                color = (0, 255, 0)  # 绿色 - 低飞鸟
            elif obstacle['type'] == "bird_high":
                color = (255, 0, 0)  # 蓝色 - 高飞鸟
            
            cv2.rectangle(screen, 
                          (obstacle['x'], obstacle['y']),
                          (obstacle['x'] + obstacle['width'], 
                           obstacle['y'] + obstacle['height']), 
                          color, 2)
            
            # 显示障碍物类型和碰撞时间
            collision_time = self.calculate_collision_time(obstacle)
            if collision_time < 5.0:  # 只显示即将到达的障碍物时间
                cv2.putText(screen, f"{obstacle['type'][:4]}", 
                           (obstacle['x'], obstacle['y'] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(screen, f"{collision_time:.1f}s", 
                           (obstacle['x'], obstacle['y'] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 显示状态信息（使用黑色文字）
        y_offset = 40
        status_color = (0, 0, 0)  # 黑色文字
        
        # 速度信息
        if self.speed_calibrated:
            cv2.putText(screen, f"Speed: {self.current_speed:.1f} px/s", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        else:
            cv2.putText(screen, f"Measuring speed... ({len(self.speed_measurements)}/{self.MIN_SPEED_SAMPLES})", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # 红色提醒
        
        y_offset += 20
        cv2.putText(screen, f"Lead time: {self.optimal_lead_time:.2f}s", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        y_offset += 20
        elapsed_time = time.time() - self.start_time
        cv2.putText(screen, f"Game time: {elapsed_time:.1f}s", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        y_offset += 20
        cv2.putText(screen, f"Samples: {len(self.speed_measurements)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # 显示游戏结束状态
        if self.game_over_counter > 0:
            cv2.putText(screen, f"Game Over Detected: {self.game_over_counter}/{self.GAME_OVER_THRESHOLD}", 
                       (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow('Dinosaur Game - Real Speed Measurement', screen)
        cv2.waitKey(1)
    
    def run(self):
        """运行游戏机器人"""
        print("准备运行...")
        
        while True:
            # 退出检测
            if keyboard.is_pressed('q'):
                print("退出游戏...")
                cv2.destroyAllWindows()
                break
            
            # 校准检测
            if keyboard.is_pressed('c') or keyboard.is_pressed('r'):
                self.calibration_complete = False
                self.game_started = False
                self.calibrate()
            
            # 开始游戏检测
            if keyboard.is_pressed('s') and self.calibration_complete and not self.game_started:
                self.start_game()
            
            if self.game_started:
                # 捕获屏幕
                screen = self.capture_screen()
                
                # 检测恐龙状态
                self.dino_state, self.dino_rect = self.detect_dinosaur(screen)
                
                # 检测障碍物
                self.obstacles = self.detect_obstacles(screen)
                
                # 检测游戏是否结束
                if self.check_game_over(self.obstacles):
                    self.restart_game()
                    continue  # 跳过本帧剩余处理
                
                # 测量真实速度
                self.measure_speed(self.obstacles)
                
                # 优化时机
                # self.optimize_timing()
                
                # 采取行动（基于真实速度计算）
                self.take_action()
                
                # 可视化结果
                self.visualize(screen, self.dino_state)
            else:
                # 显示等待状态
                if self.calibration_complete:
                    blank_screen = np.zeros((200, 800, 3), dtype=np.uint8)
                    cv2.putText(blank_screen, "Calibration Complete!", 
                               (250, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(blank_screen, "Press 'S' to start game", 
                               (250, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow('Dinosaur Game - Real Speed Measurement', blank_screen)
                else:
                    blank_screen = np.zeros((200, 800, 3), dtype=np.uint8)
                    cv2.putText(blank_screen, "Press 'C' to calibrate", 
                               (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow('Dinosaur Game - Real Speed Measurement', blank_screen)
                
                cv2.waitKey(1)
            
            # 控制循环速度
            time.sleep(0.02)

# 启动游戏机器人
if __name__ == "__main__":
    print("🚀 启动机器人...")
    bot = DinosaurGameBot()
    bot.run()
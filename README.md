# Dinosaur-Game-Bot

A Python script for Google Chrome dinosaur game, using cv2

## Features

- Real-time obstacle detection and classification
- Dynamic speed measurement for precise timing
- Automatic game restart on game over
- Visual feedback with detection overlay
- Adaptive jump timing optimization

## Requirements

```bash
pip install opencv-python numpy pyautogui keyboard mss
```

## Usage

1. Open Chrome browser and navigate to any page while offline to access the dinosaur game
2. Run the script:
   ```bash
   python dinosaur_bot.py
   ```
3. Press `C` to enter calibration mode
4. Click on the dinosaur's head position when prompted
5. Press `S` to start automatic gameplay
6. Press `Q` to quit

## Controls

- `C` - Calibrate dinosaur position
- `S` - Start automatic gameplay
- `R` - Recalibrate
- `Q` - Quit

---

## 环境要求

```bash
pip install opencv-python numpy pyautogui keyboard mss
```

## 使用方法

1. 打开Chrome浏览器，断网后访问任意页面进入小恐龙游戏
2. 运行脚本：
   ```bash
   python dinosaur_bot.py
   ```
3. 按 `C` 进入校准模式
4. 根据提示点击恐龙头部位置
5. 按 `S` 开始自动游戏
6. 按 `Q` 退出程序

## 控制按键

- `C` - 校准恐龙位置
- `S` - 开始自动游戏
- `R` - 重新校准
- `Q` - 退出程序
import cv2
import numpy as np
import math
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import deque
from ultralytics.engine.results import Keypoints
import os
import torch

class PostureGaitAnalyzer:
    def __init__(self, video_path, model_path=None):
        """
        Инициализация анализатора осанки и походки.
        
        Args:
            video_path (str): Путь к видео файлу
            model_path (str, optional): Путь к предобученной модели YOLOv8 pose. По умолчанию используется 'yolov8n-pose.pt'
        """
        self.video_path = video_path
        self.model_path = model_path or 'yolo11n-pose.pt'
        
        # Загрузка модели YOLO
        self.model = YOLO(self.model_path)
        
        # Инициализация видеозахвата
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Индексы ключевых точек YOLOv8
        self.keypoints_mapping = {
            'nose': 0,
            'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
        
        # Буферы для хранения координат ключевых точек для анализа походки
        self.left_ankle_pos = deque(maxlen=30)
        self.right_ankle_pos = deque(maxlen=30)
        self.left_wrist_pos = deque(maxlen=30)
        self.right_wrist_pos = deque(maxlen=30)
        
        # Счетчики шагов
        self.left_steps = 0
        self.right_steps = 0
        self.prev_left_dir = 0
        self.prev_right_dir = 0
        
        # Для определения завершения шага
        self.step_phase = "none"
        self.step_metrics = []  # Метрики между двумя шагами
        self.step_averages = []  # Усредненные метрики за два шага
        
        # Флаги для отслеживания фазы шага
        self.left_step_start = False
        self.right_step_start = False
        
        # Буферы для метрик
        self.metrics_history = {
            'leg_length_diff': [],
            'shoulder_height_diff': [],
            'pelvic_tilt': [],
            'shoulder_tilt': [],
            'knee_valgus_varus_left': [],
            'knee_valgus_varus_right': [],
            'step_width': [],
            'step_asymmetry': [],
            'pelvic_shoulder_rotation': [],
            'center_of_gravity_deviation': [],
            'arm_movement_symmetry': []
        }
        
        # Для хранения метрик между шагами
        self.current_step_metrics = {metric: [] for metric in self.metrics_history.keys()}
    
    def calculate_distance(self, point1, point2):
        """Расчет евклидового расстояния между двумя точками"""
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def calculate_angle(self, point1, point2):
        """Расчет угла между двумя точками относительно горизонтали"""
        return math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))
    
    def detect_keypoints(self, frame):
        """Обнаружение ключевых точек на кадре"""
        results = self.model(frame, verbose=False)[0]
        
        if not hasattr(results, 'keypoints') or results.keypoints is None or len(results.keypoints) == 0:
            return None
        
        # Convert the Keypoints object to a numpy array with the right format
        # This adapts to the new API structure
        keypoints_data = results.keypoints[0].cpu().numpy()  # Get first person's keypoints
        
        return keypoints_data
    
    def analyze_frame(self, keypoints, frame, frame_idx):
        """Анализ осанки и походки на основе ключевых точек"""
        metrics = {}

        if isinstance(keypoints, Keypoints):
            keypoints = keypoints.data

        # Проверяем, является ли keypoints объектом Keypoints
        if isinstance(keypoints, torch.Tensor):  
            keypoints = keypoints.data.cpu().numpy()  # Преобразуем в numpy массив

        # Если keypoints уже является numpy массивом — ничего не делаем
        elif isinstance(keypoints, np.ndarray):
            pass  # Оставляем без изменений

        else:
            print(f"Ошибка: keypoints имеет неверный тип ({type(keypoints)})")
            return None, frame
        
        if keypoints.ndim == 3 and keypoints.shape[0] == 1:
            keypoints = np.squeeze(keypoints)  # (1, 17, 3) → (17, 3)

        #  keypoints имеет хотя бы 3 столбца (x, y, confidence)
        if keypoints.ndim != 2 or keypoints.shape[1] < 3:
            print(f"Ошибка: keypoints имеет неверную форму {keypoints.shape}")
            return None, frame
    
        # Проверяем, что keypoints имеет правильную форму (2D массив с >=3 колонками)
        if keypoints.ndim != 2 or keypoints.shape[1] < 3:
            print(f"Ошибка: keypoints имеет неверную форму {keypoints.shape}")
            return None, frame

        # Проверяем уверенность предсказанных точек
        kp_coords = {}
        for name, idx in self.keypoints_mapping.items():
            if idx < len(keypoints):  # Проверяем, что индекс в пределах массива
                x, y, conf = keypoints[idx]
                if conf > 0.5:  #  Используем только уверенные точки
                    kp_coords[name] = (int(x), int(y))
                else:
                    kp_coords[name] = None  # Пропускаем неуверенные точки

        # Проверяем, есть ли вообще какие-то хорошие точки
        if all(v is None for v in kp_coords.values()):
            print("Все точки имеют низкую уверенность, пропускаем кадр")
            return None, frame

        # Получение координат ключевых точек
        kp_coords = {}
        for name, idx in self.keypoints_mapping.items():
            if idx < len(keypoints):  # Проверка, что индекс в пределах массива
                x, y, conf = keypoints[idx]
                kp_coords[name] = (int(x), int(y)) if conf > 0.5 else None

        # Рисуем ключевые точки
        for name, pos in kp_coords.items():
            if pos:
                cv2.circle(frame, pos, 3, (0, 255, 0), -1)
        
        # Отрисовка соединений между ключевыми точками
        connections = [
            ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_elbow'),
            ('right_shoulder', 'right_elbow'), ('left_elbow', 'left_wrist'),
            ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
            ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')
        ]
        
        for conn in connections:
            if kp_coords[conn[0]] and kp_coords[conn[1]]:
                cv2.line(frame, kp_coords[conn[0]], kp_coords[conn[1]], (0, 255, 255), 2)
        
        # 1. Разница в длине ног
        if all(kp_coords[k] for k in ['left_hip', 'left_ankle', 'right_hip', 'right_ankle']):
            left_leg_length = self.calculate_distance(kp_coords['left_hip'], kp_coords['left_ankle'])
            right_leg_length = self.calculate_distance(kp_coords['right_hip'], kp_coords['right_ankle'])
            leg_length_diff = abs(left_leg_length - right_leg_length)
            metrics['leg_length_diff'] = leg_length_diff
            self.metrics_history['leg_length_diff'].append(leg_length_diff)
            self.current_step_metrics['leg_length_diff'].append(leg_length_diff)
            
            # Отображение на кадре
            cv2.putText(frame, f"Leg Length Diff: {leg_length_diff:.2f}px", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 2. Высота плеч (асимметрия)
        if kp_coords['left_shoulder'] and kp_coords['right_shoulder']:
            shoulder_height_diff = abs(kp_coords['left_shoulder'][1] - kp_coords['right_shoulder'][1])
            metrics['shoulder_height_diff'] = shoulder_height_diff
            self.metrics_history['shoulder_height_diff'].append(shoulder_height_diff)
            self.current_step_metrics['shoulder_height_diff'].append(shoulder_height_diff)
            
            cv2.putText(frame, f"Shoulder Height Diff: {shoulder_height_diff:.2f}px", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 3. Наклон таза
        if kp_coords['left_hip'] and kp_coords['right_hip']:
            pelvic_tilt = self.calculate_angle(kp_coords['left_hip'], kp_coords['right_hip'])
            metrics['pelvic_tilt'] = pelvic_tilt
            self.metrics_history['pelvic_tilt'].append(pelvic_tilt)
            self.current_step_metrics['pelvic_tilt'].append(pelvic_tilt)
            
            cv2.putText(frame, f"Pelvic Tilt: {pelvic_tilt:.2f} deg", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 4. Наклон плеч
        if kp_coords['left_shoulder'] and kp_coords['right_shoulder']:
            shoulder_tilt = self.calculate_angle(kp_coords['left_shoulder'], kp_coords['right_shoulder'])
            metrics['shoulder_tilt'] = shoulder_tilt
            self.metrics_history['shoulder_tilt'].append(shoulder_tilt)
            self.current_step_metrics['shoulder_tilt'].append(shoulder_tilt)
            
            cv2.putText(frame, f"Shoulder Tilt: {shoulder_tilt:.2f} deg", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 5. Вальгус/варус коленей
        if all(kp_coords[k] for k in ['left_hip', 'left_knee', 'left_ankle']):
            # Для левого колена
            hip_knee_vector = (kp_coords['left_knee'][0] - kp_coords['left_hip'][0], 
                              kp_coords['left_knee'][1] - kp_coords['left_hip'][1])
            knee_ankle_vector = (kp_coords['left_ankle'][0] - kp_coords['left_knee'][0], 
                                kp_coords['left_ankle'][1] - kp_coords['left_knee'][1])
            
            # Нормализация векторов
            hip_knee_norm = math.sqrt(hip_knee_vector[0]**2 + hip_knee_vector[1]**2)
            knee_ankle_norm = math.sqrt(knee_ankle_vector[0]**2 + knee_ankle_vector[1]**2)
            
            if hip_knee_norm > 0 and knee_ankle_norm > 0:
                dot_product = hip_knee_vector[0] * knee_ankle_vector[0] + hip_knee_vector[1] * knee_ankle_vector[1]
                angle = math.degrees(math.acos(max(-1, min(1, dot_product / (hip_knee_norm * knee_ankle_norm)))))
                knee_valgus_varus_left = 180 - angle
                metrics['knee_valgus_varus_left'] = knee_valgus_varus_left
                self.metrics_history['knee_valgus_varus_left'].append(knee_valgus_varus_left)
                self.current_step_metrics['knee_valgus_varus_left'].append(knee_valgus_varus_left)
                
                cv2.putText(frame, f"Left Knee V: {knee_valgus_varus_left:.2f} deg", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if all(kp_coords[k] for k in ['right_hip', 'right_knee', 'right_ankle']):
            # Для правого колена
            hip_knee_vector = (kp_coords['right_knee'][0] - kp_coords['right_hip'][0], 
                              kp_coords['right_knee'][1] - kp_coords['right_hip'][1])
            knee_ankle_vector = (kp_coords['right_ankle'][0] - kp_coords['right_knee'][0], 
                                kp_coords['right_ankle'][1] - kp_coords['right_knee'][1])
            
            hip_knee_norm = math.sqrt(hip_knee_vector[0]**2 + hip_knee_vector[1]**2)
            knee_ankle_norm = math.sqrt(knee_ankle_vector[0]**2 + knee_ankle_vector[1]**2)
            
            if hip_knee_norm > 0 and knee_ankle_norm > 0:
                dot_product = hip_knee_vector[0] * knee_ankle_vector[0] + hip_knee_vector[1] * knee_ankle_vector[1]
                angle = math.degrees(math.acos(max(-1, min(1, dot_product / (hip_knee_norm * knee_ankle_norm)))))
                knee_valgus_varus_right = 180 - angle
                metrics['knee_valgus_varus_right'] = knee_valgus_varus_right
                self.metrics_history['knee_valgus_varus_right'].append(knee_valgus_varus_right)
                self.current_step_metrics['knee_valgus_varus_right'].append(knee_valgus_varus_right)
                
                cv2.putText(frame, f"Right Knee V: {knee_valgus_varus_right:.2f} deg", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Сохранение позиций лодыжек для анализа походки
        if kp_coords['left_ankle']:
            self.left_ankle_pos.append((kp_coords['left_ankle'], frame_idx))
        
        if kp_coords['right_ankle']:
            self.right_ankle_pos.append((kp_coords['right_ankle'], frame_idx))
        
        # Сохранение позиций запястий для анализа симметрии движения рук
        if kp_coords['left_wrist']:
            self.left_wrist_pos.append(kp_coords['left_wrist'])
        
        if kp_coords['right_wrist']:
            self.right_wrist_pos.append(kp_coords['right_wrist'])
        
        # 6. Ширина шага
        if kp_coords['left_ankle'] and kp_coords['right_ankle']:
            step_width = abs(kp_coords['left_ankle'][0] - kp_coords['right_ankle'][0])
            metrics['step_width'] = step_width
            self.metrics_history['step_width'].append(step_width)
            self.current_step_metrics['step_width'].append(step_width)
            
            cv2.putText(frame, f"Step Width: {step_width:.2f}px", (10, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 7. Асимметрия шага
        if len(self.left_ankle_pos) > 5 and len(self.right_ankle_pos) > 5:
            left_stride = self.calculate_stride([pos for pos, _ in self.left_ankle_pos])
            right_stride = self.calculate_stride([pos for pos, _ in self.right_ankle_pos])
            
            if left_stride and right_stride:
                step_asymmetry = abs(left_stride - right_stride)
                metrics['step_asymmetry'] = step_asymmetry
                self.metrics_history['step_asymmetry'].append(step_asymmetry)
                self.current_step_metrics['step_asymmetry'].append(step_asymmetry)
                
                cv2.putText(frame, f"Step Asymmetry: {step_asymmetry:.2f}px", (10, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 8. Ротация таза и плеч
        if all(kp_coords[k] for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            shoulder_angle = self.calculate_angle(kp_coords['left_shoulder'], kp_coords['right_shoulder'])
            pelvic_angle = self.calculate_angle(kp_coords['left_hip'], kp_coords['right_hip'])
            rotation_diff = abs(shoulder_angle - pelvic_angle)
            metrics['pelvic_shoulder_rotation'] = rotation_diff
            self.metrics_history['pelvic_shoulder_rotation'].append(rotation_diff)
            self.current_step_metrics['pelvic_shoulder_rotation'].append(rotation_diff)
            
            cv2.putText(frame, f"P-S Rotation: {rotation_diff:.2f} deg", (10, 270),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 9. Отклонение центра тяжести
        if all(kp_coords[k] for k in ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']):
            # Примерное вычисление центра тяжести как среднего между плечами и бедрами
            center_x = (kp_coords['left_shoulder'][0] + kp_coords['right_shoulder'][0] + 
                       kp_coords['left_hip'][0] + kp_coords['right_hip'][0]) / 4
            
            # Центральная линия (ось) тела
            mid_shoulder_x = (kp_coords['left_shoulder'][0] + kp_coords['right_shoulder'][0]) / 2
            mid_hip_x = (kp_coords['left_hip'][0] + kp_coords['right_hip'][0]) / 2
            
            # Отклонение от оси
            deviation = abs(center_x - (mid_shoulder_x + mid_hip_x) / 2)
            metrics['center_of_gravity_deviation'] = deviation
            self.metrics_history['center_of_gravity_deviation'].append(deviation)
            self.current_step_metrics['center_of_gravity_deviation'].append(deviation)
            
            cv2.putText(frame, f"COG Deviation: {deviation:.2f}px", (10, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Отрисовка центра тяжести и оси
            cog_pos = (int(center_x), int((kp_coords['left_hip'][1] + kp_coords['right_hip'][1]) / 2))
            cv2.circle(frame, cog_pos, 5, (255, 0, 0), -1)
            
            # Вертикальная ось тела
            mid_shoulder = (int(mid_shoulder_x), int((kp_coords['left_shoulder'][1] + kp_coords['right_shoulder'][1]) / 2))
            mid_hip = (int(mid_hip_x), int((kp_coords['left_hip'][1] + kp_coords['right_hip'][1]) / 2))
            cv2.line(frame, mid_shoulder, mid_hip, (255, 255, 0), 2)
            
            # Продолжение оси вниз
            bottom_y = self.frame_height
            axis_dir = (mid_hip[0] - mid_shoulder[0], mid_hip[1] - mid_shoulder[1])
            if axis_dir[1] != 0:
                t = (bottom_y - mid_hip[1]) / axis_dir[1]
                bottom_x = int(mid_hip[0] + t * axis_dir[0])
                cv2.line(frame, mid_hip, (bottom_x, bottom_y), (255, 255, 0), 2)
        
        # 10. Симметрия движения рук
        if len(self.left_wrist_pos) > 5 and len(self.right_wrist_pos) > 5:
            left_arm_movement = self.calculate_vertical_movement(self.left_wrist_pos)
            right_arm_movement = self.calculate_vertical_movement(self.right_wrist_pos)
            
            if left_arm_movement is not None and right_arm_movement is not None:
                arm_asymmetry = abs(left_arm_movement - right_arm_movement)
                metrics['arm_movement_symmetry'] = arm_asymmetry
                self.metrics_history['arm_movement_symmetry'].append(arm_asymmetry)
                self.current_step_metrics['arm_movement_symmetry'].append(arm_asymmetry)
                
                cv2.putText(frame, f"Arm Asym: {arm_asymmetry:.2f}px", (10, 330),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 11. Подсчет шагов и обработка фаз шага
        prev_left_steps = self.left_steps
        prev_right_steps = self.right_steps
        
        self.detect_and_process_steps(frame_idx)
        
        total_steps = self.left_steps + self.right_steps
        cv2.putText(frame, f"Steps: {total_steps}", (10, 360),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Если обнаружен новый шаг, выполняем усреднение метрик и сохранение
        if self.left_steps > prev_left_steps or self.right_steps > prev_right_steps:
            self.process_step_completed()
        
        # Отображение последних метрик шага, если они есть
        if self.step_averages:
            last_avg = self.step_averages[-1]
            cv2.putText(frame, f"Avg Step Metrics: L-Leg={last_avg.get('leg_length_diff', 0):.2f}", (10, 390),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return metrics, frame
    
    def detect_and_process_steps(self, frame_idx):
        """Обнаружение шагов и обработка фаз шага"""
        if len(self.left_ankle_pos) < 3 or len(self.right_ankle_pos) < 3:
            return
        
        # Для левой ноги
        left_positions = [(pos, idx) for pos, idx in self.left_ankle_pos]
        if len(left_positions) >= 3:
            # Определяем направление движения
            current_dir = 1 if left_positions[-1][0][0] > left_positions[-2][0][0] else -1
            
            # Если направление изменилось, считаем это шагом
            if current_dir != self.prev_left_dir and self.prev_left_dir != 0:
                self.left_steps += 1
                self.left_step_start = True
                # Сохраняем индекс кадра начала шага
                self.step_start_idx = frame_idx
            
            self.prev_left_dir = current_dir
        
        # Для правой ноги
        right_positions = [(pos, idx) for pos, idx in self.right_ankle_pos]
        if len(right_positions) >= 3:
            current_dir = 1 if right_positions[-1][0][0] > right_positions[-2][0][0] else -1
            
            if current_dir != self.prev_right_dir and self.prev_right_dir != 0:
                self.right_steps += 1
                self.right_step_start = True
                # Сохраняем индекс кадра начала шага
                self.step_start_idx = frame_idx
            
            self.prev_right_dir = current_dir
    
    def process_step_completed(self):
        """Обработка завершения шага и усреднение метрик"""
        # Усредняем метрики текущего шага
        step_avg = {}
        
        for metric, values in self.current_step_metrics.items():
            if values:
                step_avg[metric] = float(np.mean(values))
        
        if step_avg:
            # Добавляем усредненные метрики для этого шага
            self.step_averages.append(step_avg)
            
            # Сбрасываем метрики текущего шага
            self.current_step_metrics = {metric: [] for metric in self.metrics_history.keys()}
    
    def calculate_stride(self, pos_history):
        """Расчет длины шага на основе истории позиций"""
        if len(pos_history) < 5:
            return None
        
        # Находим максимальное горизонтальное расстояние между позициями
        max_dist = 0
        
        for i in range(len(pos_history)):
            for j in range(i+1, len(pos_history)):
                dist = abs(pos_history[i][0] - pos_history[j][0])
                max_dist = max(max_dist, dist)
        
        return max_dist
    
    def calculate_vertical_movement(self, pos_history):
        """Расчет вертикального движения на основе истории позиций"""
        if len(pos_history) < 5:
            return None
        
        y_values = [pos[1] for pos in pos_history]
        return max(y_values) - min(y_values)
    
    def analyze_video(self, output_path=None, json_path=None):
        """Анализ видео и сохранение результатов"""
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        
        all_metrics = []
        frame_idx = 0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            keypoints = self.detect_keypoints(frame)
            metrics, processed_frame = self.analyze_frame(keypoints, frame, frame_idx)
            frame_idx += 1
            
            if metrics:
                # Добавляем индекс кадра и шаги к метрикам
                metrics['frame_idx'] = frame_idx
                metrics['left_steps'] = self.left_steps
                metrics['right_steps'] = self.right_steps
                all_metrics.append(metrics)
            
            if output_path:
                out.write(processed_frame)
            
            # Отображение кадра (для интерактивного анализа)
            cv2.imshow('Posture and Gait Analysis', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Генерируем отчет
        report = self.generate_report(all_metrics)
        
        # Сохраняем результаты в JSON
        if json_path:
            self.save_to_json(json_path, report)
        
        return report
    
    def save_to_json(self, json_path, report):
        """Сохранение результатов анализа в JSON-файл"""
        # Создаем структуру данных для JSON
        json_data = {
            'video_info': {
                'path': self.video_path,
                'width': self.frame_width,
                'height': self.frame_height,
                'fps': self.fps
            },
            'overall_metrics': report,
            'steps': {
                'total_count': self.left_steps + self.right_steps,
                'left_steps': self.left_steps,
                'right_steps': self.right_steps
            },
            'step_averages': self.step_averages
        }
        
        # Преобразуем numpy типы в стандартные Python типы
        json_data_serializable = self.convert_to_serializable(json_data)
        
        # Сохраняем в файл
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data_serializable, f, indent=4)
        
        print(f"Результаты анализа сохранены в {json_path}")
    
    def convert_to_serializable(self, obj):
        """Преобразование объекта в сериализуемый формат для JSON"""
        if isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def generate_report(self, all_metrics):
        """Генерация отчета на основе собранных метрик"""
        report = {}
        
        # Обработка каждой метрики
        for metric_name in self.metrics_history:
            values = self.metrics_history[metric_name]
            if values:
                report[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        # Добавляем информацию о шагах
        report['step_count'] = {
            'total': self.left_steps + self.right_steps,
            'left': self.left_steps,
            'right': self.right_steps
        }
        
        # Анализ симметрии походки
        if self.left_steps > 0 and self.right_steps > 0:
            step_symmetry = min(self.left_steps, self.right_steps) / max(self.left_steps, self.right_steps)
            report['gait_symmetry'] = float(step_symmetry)
        else:
            report['gait_symmetry'] = 0.0
        
        return report
    
    def visualize_metrics(self, output_dir=None):
        """Визуализация метрик в виде графиков"""
        if not output_dir:
            output_dir = '.'
        
        # Создаем подкаталог, если он не существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Для каждой метрики создаем график
        for metric_name, values in self.metrics_history.items():
            if not values:
                continue
            
            plt.figure(figsize=(12, 6))
            plt.plot(values, label=metric_name)
            plt.title(f'{metric_name} over time')
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend()
            
            # Добавляем скользящее среднее для сглаживания
            window_size = min(30, len(values) // 2) if len(values) > 10 else 1
            if window_size > 1:
                smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
                plt.plot(range(window_size-1, window_size-1+len(smoothed)), smoothed, 'r--', label='Moving average')
                plt.legend()
            
            plt.savefig(f'{output_dir}/{metric_name}.png')
            plt.close()
        
        # Создаем общий график для сравнения ключевых метрик
        key_metrics = ['shoulder_tilt', 'pelvic_tilt', 'step_asymmetry']
        plt.figure(figsize=(12, 8))
        
        for metric in key_metrics:
            if metric in self.metrics_history and self.metrics_history[metric]:
                # Нормализуем значения для сравнимости
                values = self.metrics_history[metric]
                if max(values) - min(values) > 0:
                    normalized = [(x - min(values)) / (max(values) - min(values)) for x in values]
                    plt.plot(normalized, label=metric)
        
        plt.title('Comparative analysis of key metrics')
        plt.xlabel('Frame')
        plt.ylabel('Normalized value')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{output_dir}/comparative_metrics.png')
        plt.close()
        
        # График изменения метрик между шагами
        if self.step_averages:
            metrics_to_plot = ['leg_length_diff', 'shoulder_height_diff', 'pelvic_tilt', 'step_width']
            
            plt.figure(figsize=(14, 8))
            
            for metric in metrics_to_plot:
                values = [step_avg.get(metric, 0) for step_avg in self.step_averages]
                if values:
                    plt.plot(values, marker='o', label=metric)
            
            plt.title('Metrics change between steps')
            plt.xlabel('Step number')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'{output_dir}/metrics_between_steps.png')
            plt.close()
        
        print(f"Графики сохранены в директории {output_dir}")
    
    def generate_summary_report(self, output_path=None):
        """Генерация текстового отчета с рекомендациями"""
        report = self.generate_report(None)  # Генерируем отчет на основе собранных метрик
        
        summary = []
        summary.append("=== ОТЧЕТ ПО АНАЛИЗУ ОСАНКИ И ПОХОДКИ ===\n")
        summary.append(f"Видео: {self.video_path}")
        summary.append(f"Разрешение: {self.frame_width}x{self.frame_height}, FPS: {self.fps}\n")
        
        # Общая статистика
        summary.append("== ОБЩАЯ СТАТИСТИКА ==")
        summary.append(f"Всего шагов: {report['step_count']['total']}")
        summary.append(f"Левых шагов: {report['step_count']['left']}")
        summary.append(f"Правых шагов: {report['step_count']['right']}")
        summary.append(f"Симметрия походки: {report['gait_symmetry']:.2f} (1.0 - идеальная симметрия)\n")
        
        # Анализ ключевых метрик
        summary.append("== АНАЛИЗ КЛЮЧЕВЫХ МЕТРИК ==")
        
        metrics_descriptions = {
            'leg_length_diff': 'Разница в длине ног (пикселей)',
            'shoulder_height_diff': 'Разница в высоте плеч (пикселей)',
            'pelvic_tilt': 'Наклон таза (градусы)',
            'shoulder_tilt': 'Наклон плеч (градусы)',
            'knee_valgus_varus_left': 'Вальгус/варус левого колена (градусы)',
            'knee_valgus_varus_right': 'Вальгус/варус правого колена (градусы)',
            'step_width': 'Ширина шага (пикселей)',
            'step_asymmetry': 'Асимметрия шага (пикселей)',
            'pelvic_shoulder_rotation': 'Ротация таза и плеч (градусы)',
            'center_of_gravity_deviation': 'Отклонение центра тяжести (пикселей)',
            'arm_movement_symmetry': 'Симметрия движения рук (пикселей)'
        }
        
        for metric, desc in metrics_descriptions.items():
            if metric in report:
                summary.append(f"{desc}:")
                summary.append(f"  Среднее: {report[metric]['mean']:.2f}")
                summary.append(f"  Стандартное отклонение: {report[metric]['std']:.2f}")
                summary.append(f"  Минимум: {report[metric]['min']:.2f}")
                summary.append(f"  Максимум: {report[metric]['max']:.2f}")
                summary.append(f"  Медиана: {report[metric]['median']:.2f}\n")
        
        # Рекомендации
        summary.append("== РЕКОМЕНДАЦИИ ==")
        
        # Логика для рекомендаций на основе метрик
        recommendations = []
        
        # Анализ осанки
        if 'shoulder_height_diff' in report and report['shoulder_height_diff']['mean'] > 10:
            recommendations.append("- Наблюдается асимметрия плеч. Рекомендуются упражнения для укрепления мышц спины и растяжки.")
        
        if 'pelvic_tilt' in report and abs(report['pelvic_tilt']['mean']) > 5:
            if report['pelvic_tilt']['mean'] > 0:
                recommendations.append("- Присутствует передний наклон таза. Рекомендуются упражнения для укрепления брюшных мышц и растяжки сгибателей бедра.")
            else:
                recommendations.append("- Присутствует задний наклон таза. Рекомендуются упражнения для укрепления мышц нижней части спины и растяжки подколенных сухожилий.")
        
        if 'shoulder_tilt' in report and abs(report['shoulder_tilt']['mean']) > 5:
            recommendations.append("- Наблюдается наклон плеч. Рекомендуются упражнения для улучшения осанки и баланса мышц.")
        
        # Анализ походки
        if 'step_asymmetry' in report and report['step_asymmetry']['mean'] > 20:
            recommendations.append("- Значительная асимметрия шага. Рекомендуется консультация специалиста и упражнения для улучшения равномерности походки.")
        
        if 'gait_symmetry' in report and report['gait_symmetry'] < 0.8:
            recommendations.append("- Неравномерное распределение шагов между левой и правой ногой. Рекомендуются тренировки баланса и равномерности ходьбы.")
        
        if 'knee_valgus_varus_left' in report and report['knee_valgus_varus_left']['mean'] > 10:
            recommendations.append("- Наблюдается вальгус/варус левого колена. Рекомендуются укрепляющие упражнения для мышц бедра и растяжки.")
        
        if 'knee_valgus_varus_right' in report and report['knee_valgus_varus_right']['mean'] > 10:
            recommendations.append("- Наблюдается вальгус/варус правого колена. Рекомендуются укрепляющие упражнения для мышц бедра и растяжки.")
        
        if 'center_of_gravity_deviation' in report and report['center_of_gravity_deviation']['mean'] > 15:
            recommendations.append("- Значительное отклонение центра тяжести. Рекомендуются упражнения для улучшения баланса и равномерной нагрузки на ноги.")
        
        if 'arm_movement_symmetry' in report and report['arm_movement_symmetry']['mean'] > 20:
            recommendations.append("- Асимметрия движения рук при ходьбе. Рекомендуются упражнения для улучшения координации движений верхних конечностей.")
        
        # Если серьезных проблем не обнаружено
        if not recommendations:
            recommendations.append("- Серьезных отклонений не обнаружено. Для поддержания здоровой осанки и походки рекомендуется регулярная физическая активность и упражнения на баланс.")
        
        # Добавляем рекомендации в отчет
        summary.extend(recommendations)
        
        # Формируем итоговый текст отчета
        summary_text = '\n'.join(summary)
        
        # Сохраняем отчет в файл, если указан путь
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            print(f"Отчет сохранен в {output_path}")
        
        return summary_text

def main():
    """Пример использования анализатора осанки и походки"""
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description='Анализатор осанки и походки')
    parser.add_argument('video', type=str, help='Путь к видео файлу для анализа')
    parser.add_argument('--output', type=str, default=None, help='Путь для сохранения обработанного видео')
    parser.add_argument('--json', type=str, default=None, help='Путь для сохранения результатов в JSON')
    parser.add_argument('--report', type=str, default=None, help='Путь для сохранения текстового отчета')
    parser.add_argument('--visualize', type=str, default=None, help='Директория для сохранения графиков')
    parser.add_argument('--model', type=str, default=None, help='Путь к модели YOLOv8 (опционально)')
    
    args = parser.parse_args()
    
    # Проверка наличия файла видео
    if not os.path.exists(args.video):
        print(f"Ошибка: файл видео {args.video} не найден")
        return
    
    # Создание и запуск анализатора
    analyzer = PostureGaitAnalyzer(args.video, args.model)
    
    # Анализ видео
    print(f"Анализ видео: {args.video}")
    report = analyzer.analyze_video(args.output, args.json)
    
    # Визуализация метрик
    if args.visualize:
        print("Генерация графиков...")
        analyzer.visualize_metrics(args.visualize)
    
    # Генерация и сохранение текстового отчета
    if args.report:
        print("Генерация текстового отчета...")
        analyzer.generate_summary_report(args.report)
    
    print("Анализ завершен!")

if __name__ == "__main__":
    main()
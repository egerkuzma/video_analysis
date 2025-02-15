import websocket
import ssl
import threading
import time
import logging
import json
import base64
import signal
import sys
import cv2
import numpy as np
import subprocess
import torch
from ultralytics import YOLO
import warnings

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('websocket_debug.log'),
        logging.StreamHandler()
    ]
)

# Глобальные переменные
ffmpeg_proc = None
ws_app = None
init_data_received = False
model = None
people_count = 0
tracked_ids = {}  # Изменяем на словарь для хранения состояний
ZONE_COOLDOWN = 300  # Количество кадров ожидания перед повторным подсчетом


# Параметры видео
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3
# Добавляем коэффициент масштабирования для отображения
SCALE_FACTOR = 1  # уменьшаем размер в 2 раза

# Добавляем параметры зоны детекции после параметров видео
DETECTION_ZONE = {
    'x': 350,
    'y': 50,
    'width': 150,
    'height': 150
}

# Добавляем новые константы после существующих
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 5  # секунды между попытками реконнекта

def init_model():
    """
    Инициализация YOLOv8 с трекером
    """
    global model
    try:
        # Загружаем YOLOv8n (можно использовать 's', 'm', 'l', 'x' для других размеров)
        model = YOLO('yolov8x.pt')
        logging.info("YOLOv8 успешно инициализирован")
    except Exception as e:
        logging.error(f"Ошибка при инициализации YOLOv8: {e}")
        sys.exit(1)

def calculate_distance(pos1, pos2):
    """Вычисляет евклидово расстояние между двумя точками"""
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

def process_frame(frame):
    """
    Обработка полученного кадра
    """
    global model, ws_app, people_count, tracked_ids
    try:
        # Создаем маску для верхней трети кадра
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[50:200, 300:500] = 255
        
        # Применяем маску к кадру
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Запускаем детекцию и трекинг с отключенным выводом
        results = model.track(masked_frame, persist=True, classes=[0], conf=0.2, verbose=False)

        # Рисуем линию разделения
        cv2.line(frame, (0, FRAME_HEIGHT//3), (FRAME_WIDTH, FRAME_HEIGHT//3), (0, 0, 255), 2)
        
        # Уменьшаем счетчики cooldown для всех треков
        for track_id in list(tracked_ids.keys()):
            tracked_ids[track_id]['cooldown'] = max(0, tracked_ids[track_id]['cooldown'] - 1)
            # Удаляем треки, которые не появлялись долгое время
            if tracked_ids[track_id]['frames_missing'] > 300:  # 2 секунды при 30 FPS
                del tracked_ids[track_id]
            else:
                tracked_ids[track_id]['frames_missing'] += 1

        # Рисуем зону детекции
        zone_color = (255, 0, 0)  # Синий цвет для зоны
        cv2.rectangle(frame, 
                     (DETECTION_ZONE['x'], DETECTION_ZONE['y']), 
                     (DETECTION_ZONE['x'] + DETECTION_ZONE['width'], 
                      DETECTION_ZONE['y'] + DETECTION_ZONE['height']), 
                     zone_color, 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box[:4])
                track_id = int(track_id)
                
                # Пропускаем уже подсчитанные треки
                if track_id in tracked_ids and tracked_ids[track_id]['counted']:
                    continue
                
                # Проверяем, находится ли центр трека в зоне детекции
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                in_zone = (DETECTION_ZONE['x'] <= center_x <= DETECTION_ZONE['x'] + DETECTION_ZONE['width'] and
                          DETECTION_ZONE['y'] <= center_y <= DETECTION_ZONE['y'] + DETECTION_ZONE['height'])
                
                # Инициализируем трек, если он новый
                if track_id not in tracked_ids:
                    tracked_ids[track_id] = {
                        'counted': False,
                        'in_zone': False,
                        'cooldown': 0,
                        'frames_missing': 0
                    }
                
                # Обновляем состояние трека
                track_info = tracked_ids[track_id]
                track_info['frames_missing'] = 0
                
                if in_zone:
                    if not track_info['in_zone'] and track_info['cooldown'] == 0:
                        if not track_info['counted']:
                            people_count += 1
                            track_info['counted'] = True
                            continue  # Прекращаем отслеживание после подсчёта
                        track_info['cooldown'] = ZONE_COOLDOWN
                    color = (0, 255, 0)  # Зеленый цвет для треков в зоне
                else:
                    color = (255, 255, 0)  # Желтый цвет для треков вне зоны
                
                track_info['in_zone'] = in_zone
                
                # Рисуем бокс и ID только для неподсчитанных треков
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Добавляем индикатор состояния подсчета
                status = "Counted" if track_info['counted'] else "Not counted"
                cv2.putText(frame, status, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Отображаем счетчик
        cv2.putText(frame, f"Total unique people: {people_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        display_width = int(FRAME_WIDTH * SCALE_FACTOR)
        display_height = int(FRAME_HEIGHT * SCALE_FACTOR)
        display_frame = cv2.resize(frame, (display_width, display_height))
        
        cv2.imshow('Video Analysis', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if ws_app:
                ws_app.close()
    except Exception as e:
        logging.error(f"Error in process_frame: {e}")

def ffmpeg_decoder():
    """
    Потоковая функция для чтения кадров, декодированных FFmpeg.
    """
    global ffmpeg_proc
    while True:
        try:
            raw_frame = ffmpeg_proc.stdout.read(FRAME_SIZE)
            if len(raw_frame) < FRAME_SIZE:
                continue  # Убираем sleep
            frame = np.frombuffer(raw_frame, np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3)).copy()
            process_frame(frame)
        except Exception as e:
            logging.error(f"Error in ffmpeg_decoder: {e}")
            break

def write_init_data(data):
    """
    Обработка и передача инициализационных данных (в формате base64) в FFmpeg.
    Эти данные (например, заголовок фрагментированного MP4) должны идти перед медиаданными.
    """
    global init_data_received, ffmpeg_proc
    if not init_data_received:
        init_bytes = base64.b64decode(data)
        try:
            ffmpeg_proc.stdin.write(init_bytes)
            ffmpeg_proc.stdin.flush()
            init_data_received = True
            logging.info("Initialization data written to ffmpeg.")
            print("Video processing started...")
        except Exception as e:
            logging.error(f"Error writing init data: {e}")

def on_message(ws, message):
    """
    Обработка сообщений из WebSocket.
    Если сообщение текстовое и содержит инициализационные данные – передаём их в FFmpeg.
    Если сообщение бинарное – сразу пишем в stdin FFmpeg.
    """
    global ffmpeg_proc
    try:
        if isinstance(message, str):
            json_data = json.loads(message)
            if json_data.get('type') == 'stream-init' and 'data' in json_data:
                write_init_data(json_data['data'])
        elif isinstance(message, bytes):
            if ffmpeg_proc and ffmpeg_proc.stdin:
                try:
                    ffmpeg_proc.stdin.write(message)
                    ffmpeg_proc.stdin.flush()
                except Exception as e:
                    logging.error(f"Error writing video chunk to ffmpeg: {e}")
    except Exception as e:
        logging.error(f"Error in on_message: {e}")

def on_error(ws, error):
    logging.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    """
    Обработчик закрытия соединения WebSocket
    """
    logging.info(f"WebSocket closed: Code: {close_status_code}, Message: {close_msg}")
    
    # Пытаемся переподключиться
    if not reconnect():
        logging.error("Не удалось восстановить соединение")
        cv2.destroyAllWindows()
        if ffmpeg_proc:
            ffmpeg_proc.terminate()
        sys.exit(1)

def on_open(ws):
    logging.info("WebSocket connection established.")
    print("Press 'q' in the video window or Ctrl+C in the console to quit.")

def signal_handler(signum, frame):
    logging.info("Termination signal received. Closing...")
    if ws_app:
        ws_app.close()
    if ffmpeg_proc:
        ffmpeg_proc.terminate()
    cv2.destroyAllWindows()
    sys.exit(0)

def connect_websocket():
    """
    Создает и возвращает WebSocket соединение
    """
    global ws_app
    ws_url = (
        "wss://rs2125.extcam.com/ws-fmp4/live?server=100-QBFfvYTO6A5EZ33a6T3esI&camera=0&access_token=public&streams=video&vcodec=h264&acodec=aac&acodec=mp3&acodec=pcma&acodec=pcmu&acodec=none&duration=0&q=2&d=rs2125.extcam.com&public=1&owner_id=100001687685&u=100001310398&ts=1739624899.387309&token=7d53f0f32ae34cdccffc45ad9b6ddcb0"
    )
    
    ws_app = websocket.WebSocketApp(ws_url,
                                  on_open=on_open,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close)
    return ws_app

def reconnect():
    """
    Пытается переподключиться к WebSocket
    """
    global ws_app, ffmpeg_proc
    attempts = 0
    
    while attempts < MAX_RECONNECT_ATTEMPTS:
        logging.info(f"Попытка переподключения {attempts + 1}/{MAX_RECONNECT_ATTEMPTS}")
        try:
            # Закрываем старые процессы
            if ffmpeg_proc:
                ffmpeg_proc.terminate()
            if ws_app:
                ws_app.close()
            
            # Пересоздаем FFmpeg процесс
            ffmpeg_cmd = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-i', 'pipe:0',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                'pipe:1'
            ]
            ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            
            # Создаем новое WebSocket соединение
            ws_app = connect_websocket()
            
            # Запускаем WebSocket в отдельном потоке
            websocket_thread = threading.Thread(target=ws_app.run_forever, 
                                             kwargs={'sslopt': {"cert_reqs": ssl.CERT_NONE}},
                                             daemon=True)
            websocket_thread.start()
            
            logging.info("Успешное переподключение!")
            return True
            
        except Exception as e:
            logging.error(f"Ошибка при переподключении: {e}")
            attempts += 1
            time.sleep(RECONNECT_DELAY)
    
    logging.error("Превышено максимальное количество попыток переподключения")
    return False

if __name__ == "__main__":
    # Обработка сигналов завершения
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logging.info("Script started")
    
    # Выводим информацию об устройстве
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nИспользуется устройство: {device}\n")
    
    # Инициализируем YOLOv8
    init_model()

    # Запускаем FFmpeg-процесс для декодирования видеопотока
    ffmpeg_cmd = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', 'pipe:0',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        'pipe:1'
    ]
    
    try:
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    except Exception as e:
        logging.error(f"Error starting ffmpeg: {e}")
        sys.exit(1)
    
    # Запускаем поток для чтения и обработки кадров из FFmpeg
    decoder_thread = threading.Thread(target=ffmpeg_decoder, daemon=True)
    decoder_thread.start()
    
    try:
        # Создаем и запускаем WebSocket соединение
        ws_app = connect_websocket()
        ws_app.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if not reconnect():
            logging.error("Не удалось восстановить соединение")
    finally:
        if ffmpeg_proc:
            ffmpeg_proc.terminate()
        cv2.destroyAllWindows()
        print("Stream ended.")

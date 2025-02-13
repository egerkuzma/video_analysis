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
from deepface import DeepFace
import subprocess

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

# Параметры видео (укажите реальные значения для вашего потока)
FRAME_WIDTH = 1280  # измените, если требуется
FRAME_HEIGHT = 720  # измените, если требуется
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3  # для формата bgr24

# Параметры ROI (области интереса)
LEFT_ROI_X = 240
LEFT_ROI_Y = 130
LEFT_ROI_WIDTH = 250
LEFT_ROI_HEIGHT = 150

def process_frame(frame):
    """
    Обработка полученного кадра:
      - Рисование ROI
      - Анализ эмоций с помощью DeepFace
      - Отображение результата
    """
    try:
        # Если вдруг передали read-only массив, делаем копию
        frame = frame.copy()

        height, width = frame.shape[:2]
        print(f"Frame size: {width}x{height}")
        
        # Рисуем прямоугольник вокруг ROI
        cv2.rectangle(frame, 
                      (LEFT_ROI_X, LEFT_ROI_Y), 
                      (LEFT_ROI_X + LEFT_ROI_WIDTH, LEFT_ROI_Y + LEFT_ROI_HEIGHT), 
                      (255, 0, 0), 3)
        
        # Вырезаем область для анализа
        left_roi = frame[LEFT_ROI_Y:LEFT_ROI_Y+LEFT_ROI_HEIGHT, 
                         LEFT_ROI_X:LEFT_ROI_X+LEFT_ROI_WIDTH]
        
        try:
            print("Starting emotion analysis...")
            result = DeepFace.analyze(left_roi, 
                                      actions=['emotion'],
                                      enforce_detection=False,
                                      silent=True,
                                      detector_backend='opencv')
            print("Analysis completed")
            if result:
                emotion = result[0]['dominant_emotion']
                emotion_score = result[0]['emotion'][emotion]
                print(f"Detected emotion: {emotion} ({emotion_score:.2f})")
                emotion_ru = {
                    'angry': 'Zlost',
                    'disgust': 'Otvrashchenie',
                    'fear': 'Strach',
                    'happy': 'Radost',
                    'sad': 'Grust',
                    'surprise': 'Udivlenie',
                    'neutral': 'Neitralno'
                }
                text = f"{emotion_ru.get(emotion, emotion).upper()} ({emotion_score:.2f})"
                cv2.putText(frame, 
                            text, 
                            (LEFT_ROI_X + 10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 255, 255), 
                            2)
            else:
                cv2.putText(frame, 
                            "NO EMOTIONS DETECTED", 
                            (LEFT_ROI_X + 10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.5, 
                            (0, 0, 255), 
                            4)
        except Exception as e:
            print(f"Error in emotion analysis: {e}")
            cv2.putText(frame, 
                        "ANALYSIS ERROR", 
                        (LEFT_ROI_X + 10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, 
                        (0, 0, 255), 
                        4)
        
        cv2.imshow('Video Analysis', frame)
        # Если нажата клавиша 'q', завершаем работу
        if cv2.waitKey(int(1000 / 25)) & 0xFF == ord('q'):
            if ws_app:
                ws_app.close()
    except Exception as e:
        logging.error(f"Error in process_frame: {e}")

def ffmpeg_decoder():
    """
    Потоковая функция для чтения кадров, декодированных FFmpeg.
    Каждый кадр читается из stdout и передается в process_frame.
    """
    global ffmpeg_proc
    while True:
        try:
            # Читаем ровно один кадр (FRAME_SIZE байт)
            raw_frame = ffmpeg_proc.stdout.read(FRAME_SIZE)
            if len(raw_frame) < FRAME_SIZE:
                # Если данных меньше, ждём поступления следующих
                time.sleep(0.01)
                continue
            # Преобразуем считанные байты в изображение numpy и делаем копию, чтобы оно было записываемым
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
    logging.info(f"WebSocket closed: Code: {close_status_code}, Message: {close_msg}")
    cv2.destroyAllWindows()
    if ffmpeg_proc:
        ffmpeg_proc.terminate()
    print("Stream closed.")

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

if __name__ == "__main__":
    # Обработка сигналов завершения
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logging.info("Script started")

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
    
    # Настраиваем WebSocket-соединение
    ws_url = (
        "wss://rs2073.extcam.com/ws-fmp4/live"
    )
    
    ws_app = websocket.WebSocketApp(ws_url,
                                    on_open=on_open,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close)
    
    websocket.enableTrace(False)
    
    try:
        ws_app.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        if ffmpeg_proc:
            ffmpeg_proc.terminate()
        cv2.destroyAllWindows()
        print("Stream ended.")

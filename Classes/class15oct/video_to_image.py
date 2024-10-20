import cv2
import os 

VIDEO_PATH = 'video.MOV'
OUTPUT_PATH = 'dataset/groot'

def video_to_image(video_path):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video")
        exit()
    
    i = 0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate) // 2
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("No se puede recibir el fotograma. Saliendo...")
            break
            
        if i % frame_interval == 0: # Save every 2 seconds
            cv2.imwrite(f'{OUTPUT_PATH}/frame_{i}.jpg', frame) # Save frame as JPEG file
        
        i += 1
        
    print(f'Frames guardados en {OUTPUT_PATH}')
    cap.release()
    cv2.destroyAllWindows()


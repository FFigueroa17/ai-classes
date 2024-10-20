import cv2
import os 

VIDEO_PATH = 'videos/bag.MOV'
OUTPUT_PATH = 'dataset/bag'

def video_to_image(video_path):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video")
        exit()
    
    i = 0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * 0.1)
    saved_images = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret or saved_images >= 1000:
            print("No se puede recibir el fotograma o se alcanzaron las 1000 imágenes. Saliendo...")
            break
            
        if i % frame_interval == 0: # Save every 0.1 seconds
            cv2.imwrite(f'{OUTPUT_PATH}/frame_{i}.jpg', frame) # Save frame as JPEG file
            saved_images += 1
        
        i += 1
        
    print(f'Frames guardados en {OUTPUT_PATH}')
    cap.release()
    cv2.destroyAllWindows()

video_to_image(VIDEO_PATH)
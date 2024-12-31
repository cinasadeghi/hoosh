import tkinter as tk
import time
import cv2
from PIL import Image, ImageTk
import requests
from deepface import DeepFace

def get_weather():
    city = "Shiraz"
    latitude = 29.6103
    longitude = 52.5311
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    response = requests.get(url)
    weather_data = response.json()
    if "current_weather" in weather_data:
        temperature = weather_data["current_weather"]["temperature"]
        weather = weather_data["current_weather"]["weathercode"]
        return f"{city}: {temperature}°C, {weather}"
    else:
        return "Weather data not available"

def update():
    current_time = time.strftime("%H:%M:%S")
    time_label.config(text=current_time)
    weather_label.config(text=get_weather())
    root.after(60000, update)  # Update every 60 seconds

def update_video():
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        try:
            analysis = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']
            emotion_label.config(text=f"Emotion: {dominant_emotion.capitalize()}")
        except Exception as e:
            emotion_label.config(text="Emotion: Not detected")
            print(f"Error detecting emotion: {e}")

    video_label.after(10, update_video)

root = tk.Tk()
root.title("Smart Mirror")
root.configure(bg='black')

video_label = tk.Label(root)
video_label.pack(fill=tk.BOTH, expand=True)

time_label = tk.Label(root, font=('Helvetica', 48), fg='white', bg='black')
time_label.pack(pady=20)

weather_label = tk.Label(root, font=('Helvetica', 24), fg='white', bg='black')
weather_label.pack(pady=20)

emotion_label = tk.Label(root, font=('Helvetica', 24), fg='white', bg='black')
emotion_label.pack(pady=20)

cap = cv2.VideoCapture(0)

update()
update_video()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
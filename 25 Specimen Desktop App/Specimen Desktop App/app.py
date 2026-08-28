import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from tkinter import ttk
from threading import Thread
from PIL import Image, ImageTk

class MotionDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Motion Detection App")
        master.configure(background='#f3f4f6')

        # Apply theme for ttk
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 10), padding=5)
        style.configure('TLabel', font=('Helvetica', 10), background='#f3f4f6')
        style.configure('TFrame', background='#f3f4f6')

        # Set bold font for output text
        self.bold_font = ('Helvetica', 10, 'bold')

        self.create_widgets()

    def create_widgets(self):
        # Frame for controls
        control_frame = ttk.Frame(self.master)
        control_frame.pack(padx=10, pady=10, fill='x', expand=False)

        self.file_label = ttk.Label(control_frame, text="Selected Video: None")
        self.file_label.pack(side='left', padx=5, pady=5)

        self.browse_button = ttk.Button(control_frame, text="Browse", command=self.browse_video)
        self.browse_button.pack(side='left', padx=5, pady=5)

        self.run_button = ttk.Button(control_frame, text="Run", command=self.run_motion_detection)
        self.run_button.pack(side='left', padx=5, pady=5)

        # Frame for video canvas
        video_frame = ttk.Frame(self.master, relief=tk.RAISED, borderwidth=1)
        video_frame.pack(padx=10, pady=10, fill='both', expand=True)

        self.canvas = tk.Canvas(video_frame, bg='black', height=300)
        self.canvas.pack(fill='both', expand=True)

        # Frame for output text
        output_frame = ttk.Frame(self.master)
        output_frame.pack(padx=10, pady=10, fill='both', expand=True)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=self.bold_font, height=8)
        self.output_text.pack(fill='both', expand=True)

        # Status label
        self.status_label = ttk.Label(self.master, text="Status: Ready")
        self.status_label.pack(padx=10, pady=5)

    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if file_path:
            self.file_label.config(text=f"Selected Video: {file_path}")
            self.video_path = file_path

    def run_motion_detection(self):
        if hasattr(self, 'video_path'):
            self.status_label.config(text="Running Motion Detection...")
            self.run_button.config(state="disabled")
            motion_thread = Thread(target=self.run_motion_detection_thread)
            motion_thread.start()
        else:
            messagebox.showerror("Error", "Please select a video file first.")
            self.status_label.config(text="Status: Ready")

    def run_motion_detection_thread(self):
        try:
            calculate_movement(self.video_path, self)
            self.status_label.config(text="Motion Detection Completed.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Status: Error")
        finally:
            self.run_button.config(state="normal")

    def update_output_text(self, text):
        self.output_text.delete(1.0, tk.END)  # Clear existing text
        self.output_text.insert(tk.END, text)  # Insert new text
        self.output_text.see(tk.END)

def calculate_movement(video_path, app):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Could not read the first frame.")

    total_movement = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        diff = cv2.absdiff(prev_frame, frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, threshold_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

        movement = np.sum(threshold_diff) / 255
        total_movement += movement
        frame_count += 1

        current_time = frame_count / fps
        time_remaining = total_duration - current_time
        remaining_min, remaining_sec = divmod(time_remaining, 60)

        output_text = f"Processing Frame {frame_count}/{total_frames}. "
        output_text += f"Time Remaining: {int(remaining_min)} min {int(remaining_sec)} sec. "
        output_text += f"Average Movement: {total_movement/frame_count:.2f}"
        app.update_output_text(output_text)

        # Convert the frame to RGB format for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas with the new frame
        app.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        app.canvas.image = img_tk

        # Update the Tkinter window to display the new frame
        app.master.update()

        prev_frame = frame

    average_movement = total_movement / frame_count
    app.update_output_text(f"Final Average Movement: {average_movement:.2f}")

    cap.release()
    cv2.destroyAllWindows()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = MotionDetectionApp(root)
    root.mainloop()

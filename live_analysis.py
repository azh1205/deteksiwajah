
#live_analysis.py
from deepface import DeepFace
import cv2
import numpy as np
from datetime import datetime
import os

# ----------------- Configuration -----------------
DB_PATH = "face_database"
RECOGNITION_MODEL = "VGG-Face"
DETECTOR_BACKEND = "mtcnn"

# UI Colors (BGR format)
COLORS = {
    'primary': (255, 107, 53),      # Orange
    'success': (46, 204, 113),      # Green
    'danger': (231, 76, 60),        # Red
    'dark': (44, 62, 80),           # Dark blue
    'light': (236, 240, 241),       # Light gray
    'text': (255, 255, 255),        # White
    'bg_overlay': (30, 30, 30),     # Dark overlay
}

# Emotion emoji mapping
EMOTION_EMOJIS = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'surprise': '😲',
    'fear': '😨',
    'disgust': '🤢',
    'neutral': '😐'
}

class FaceRecognitionUI:
    def __init__(self, db_path, model_name, detector_backend):
        self.db_path = db_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.cap = cv2.VideoCapture(0)
        self.frame_count = 0
        self.process_every_n_frames = 3  # Process every 3rd frame for better performance
        self.last_results = []
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    def draw_rounded_rectangle(self, img, pt1, pt2, color, thickness=2, radius=15):
        """Draw a rounded rectangle"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw main rectangles
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw corners
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        
    def draw_info_panel(self, frame, face_data, x, y, w, h):
        """Draw an attractive info panel for detected face"""
        panel_width = 280
        panel_height = 180
        panel_x = x + w + 15
        panel_y = y
        
        # Adjust panel position if it goes off screen
        if panel_x + panel_width > frame.shape[1]:
            panel_x = x - panel_width - 15
        if panel_y + panel_height > frame.shape[0]:
            panel_y = frame.shape[0] - panel_height - 10
            
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     COLORS['bg_overlay'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Draw border
        self.draw_rounded_rectangle(frame, (panel_x, panel_y),
                                   (panel_x + panel_width, panel_y + panel_height),
                                   COLORS['primary'], 2, 10)
        
        # Extract info
        identity = face_data.get('identity', 'Unknown')
        if identity != 'Unknown':
            identity = os.path.basename(identity).split('.')[0]
        
        emotion = face_data.get('dominant_emotion', 'neutral')
        confidence = face_data.get('distance', 0)
        
        # Draw identity section
        text_y = panel_y + 35
        cv2.putText(frame, "IDENTITY", (panel_x + 15, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
        
        text_y += 25
        identity_color = COLORS['success'] if identity != 'Unknown' else COLORS['danger']
        cv2.putText(frame, identity[:20], (panel_x + 15, text_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, identity_color, 2)
        
        # Draw emotion section
        text_y += 35
        cv2.putText(frame, "EMOTION", (panel_x + 15, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
        
        text_y += 25
        emoji = EMOTION_EMOJIS.get(emotion, '😐')
        emotion_text = f"{emotion.upper()}"
        cv2.putText(frame, emotion_text, (panel_x + 15, text_y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, COLORS['primary'], 2)
        
        # Draw confidence section
        if identity != 'Unknown':
            text_y += 35
            cv2.putText(frame, "CONFIDENCE", (panel_x + 15, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
            
            # Confidence bar
            text_y += 20
            bar_width = 250
            bar_height = 15
            conf_percentage = max(0, min(100, 100 - (confidence * 10)))
            
            # Background bar
            cv2.rectangle(frame, (panel_x + 15, text_y),
                         (panel_x + 15 + bar_width, text_y + bar_height),
                         (60, 60, 60), -1)
            
            # Confidence bar
            conf_bar_width = int(bar_width * (conf_percentage / 100))
            bar_color = COLORS['success'] if conf_percentage > 60 else COLORS['danger']
            cv2.rectangle(frame, (panel_x + 15, text_y),
                         (panel_x + 15 + conf_bar_width, text_y + bar_height),
                         bar_color, -1)
            
            # Percentage text
            cv2.putText(frame, f"{int(conf_percentage)}%",
                       (panel_x + 15 + bar_width + 10, text_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
    
    def draw_header(self, frame):
        """Draw header bar"""
        header_height = 60
        
        # Draw header background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], header_height),
                     COLORS['bg_overlay'], -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Title
        cv2.putText(frame, "Face Recognition System", (20, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, COLORS['primary'], 2)
        
        # Time
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_str, (frame.shape[1] - 150, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)
        
        # FPS counter
        fps_text = f"Faces: {len(self.last_results)}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 350, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['light'], 1)
    
    def draw_footer(self, frame):
        """Draw footer with instructions"""
        footer_height = 40
        footer_y = frame.shape[0] - footer_height
        
        # Draw footer background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, footer_y), (frame.shape[1], frame.shape[0]),
                     COLORS['bg_overlay'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Instructions
        text = "Press 'Q' to quit | 'S' to save snapshot | 'R' to reload database"
        cv2.putText(frame, text, (20, footer_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['light'], 1)
    
    def process_frame(self, frame):
        """Process frame with DeepFace"""
        try:
            results = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=self.detector_backend,
                silent=True
            )
            
            # Try to find matches in database
            try:
                matches = DeepFace.find(
                    frame,
                    db_path=self.db_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    silent=True
                )
                
                # Process results
                if isinstance(results, list):
                    for i, result in enumerate(results):
                        if i < len(matches) and len(matches[i]) > 0:
                            result['identity'] = matches[i].iloc[0]['identity']
                            result['distance'] = matches[i].iloc[0]['distance']
                        else:
                            result['identity'] = 'Unknown'
                            result['distance'] = 0
                else:
                    if len(matches[0]) > 0:
                        results['identity'] = matches[0].iloc[0]['identity']
                        results['distance'] = matches[0].iloc[0]['distance']
                    else:
                        results['identity'] = 'Unknown'
                        results['distance'] = 0
                    results = [results]
            except Exception as e:
                # Database lookup failed, mark as Unknown
                if isinstance(results, list):
                    for result in results:
                        result['identity'] = 'Unknown'
                        result['distance'] = 0
                else:
                    results['identity'] = 'Unknown'
                    results['distance'] = 0
                    results = [results]
            
            return results
        except Exception as e:
            print(f"Frame processing error: {e}")
            return []
    
    def run(self):
        """Main loop"""
        print(f"\n{'='*60}")
        print(f"🚀 Face Recognition System Started")
        print(f"{'='*60}")
        print(f"📁 Database: {self.db_path}")
        print(f"🤖 Model: {self.model_name}")
        print(f"🔍 Detector: {self.detector_backend}")
        print(f"{'='*60}\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ Failed to read frame from camera")
                    break
                
                # Process frame periodically
                if self.frame_count % self.process_every_n_frames == 0:
                    self.last_results = self.process_frame(frame)
                
                # Draw UI elements
                self.draw_header(frame)
                
                # Draw face boxes and info panels
                for face in self.last_results:
                    region = face.get('region', {})
                    x = region.get('x', 0)
                    y = region.get('y', 0)
                    w = region.get('w', 100)
                    h = region.get('h', 100)
                    
                    # Draw face box with rounded corners
                    identity = face.get('identity', 'Unknown')
                    box_color = COLORS['success'] if identity != 'Unknown' else COLORS['primary']
                    self.draw_rounded_rectangle(frame, (x, y), (x + w, y + h), box_color, 3, 15)
                    
                    # Draw corner accents
                    corner_len = 20
                    cv2.line(frame, (x, y), (x + corner_len, y), box_color, 5)
                    cv2.line(frame, (x, y), (x, y + corner_len), box_color, 5)
                    cv2.line(frame, (x + w, y), (x + w - corner_len, y), box_color, 5)
                    cv2.line(frame, (x + w, y), (x + w, y + corner_len), box_color, 5)
                    cv2.line(frame, (x, y + h), (x + corner_len, y + h), box_color, 5)
                    cv2.line(frame, (x, y + h), (x, y + h - corner_len), box_color, 5)
                    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), box_color, 5)
                    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), box_color, 5)
                    
                    # Draw info panel
                    self.draw_info_panel(frame, face, x, y, w, h)
                
                self.draw_footer(frame)
                
                # Show frame
                cv2.imshow('Face Recognition System', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"snapshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Snapshot saved: {filename}")
                elif key == ord('r') or key == ord('R'):
                    print("🔄 Reloading database...")
                
                self.frame_count += 1
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            # Cleanup - always executed
            self.cap.release()
            cv2.destroyAllWindows()
            print("\n✅ System stopped successfully")

# ----------------- Run the application -----------------
if __name__ == "__main__":
    app = FaceRecognitionUI(DB_PATH, RECOGNITION_MODEL, DETECTOR_BACKEND)
    app.run()
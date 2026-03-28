import customtkinter as ctk
from deepface import DeepFace
import os
import tkinter as tk
from PIL import Image, ImageTk
from deepface.modules import detection
import numpy as np
import threading
from datetime import datetime, timedelta
import cv2
import matplotlib.pyplot as plt
import csv
import tkinter.messagebox as tkmsg
import deepface.modules.modeling as df_mod
from queue import Queue


# ----------  GLOBAL MODEL CACHE  ----------
# key   : "VGG-Face", "Facenet", ...
# value : dict returned by DeepFace.build_model (already initialised)
MODEL_CACHE = {}
def get_model(name: str):
    if name not in MODEL_CACHE:
        MODEL_CACHE[name] = DeepFace.build_model(name)
    return MODEL_CACHE[name]

def _inject_model(name: str):
    """Put our pre-loaded model into DeepFace’s private cache."""
    if name not in df_mod.cached_models.get("facial_recognition", {}):
        df_mod.cached_models.setdefault("facial_recognition", {})[name] = get_model(name)

# warm-up and inject
for m in ["VGG-Face", "Facenet", "ArcFace", "OpenFace", "DeepID", "SFace"]:
    try:
        _inject_model(m)
    except Exception as e:
        print("warm-up skip", m, e)

ctk.set_appearance_mode("Dark") 
ctk.set_default_color_theme("green") 


# =====================  SERVICE LAYER  =====================
# =====================  UPDATED CAMERA SERVICE  =====================
class CameraService:
    """Handles webcam life-cycle and frame grabbing."""
    def __init__(self, src=0, width=640, height=480):
        self.cap = None
        self.src = src
        self.width = width
        self.height = height

    @staticmethod
    def get_available_cameras(max_test=5):
        """Detect available cameras by trying to open them."""
        available = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to verify it actually works
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
        return available

    def start(self):
        try:
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return True
        except Exception as e:
            print(f"Camera start error: {e}")
            self.stop()
            return False

    def read(self):
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
        return ret, frame

    def stop(self):
        if self.cap:
            self.cap.release()
            self.cap = None


class FaceRecognitionService:
    """All DeepFace operations (analyze, verify, find)."""
    def __init__(self, db_path="face_database", model_name="VGG-Face",
                 detector_backend="retinaface"):
        self.db_path = db_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        # pre-inject model into DeepFace cache
        _inject_model(model_name)

    # ---------------  static helpers  ---------------
    @staticmethod
    def verify_two_images(path_a, path_b, model_name="VGG-Face", **kwargs):
        _inject_model(model_name)
        return DeepFace.verify(
            img1_path=path_a, 
            img2_path=path_b,
            model_name=model_name,
            detector_backend="retinaface",
            # Pass any extra keyword arguments (like threshold) here:
            **kwargs 
        )

    @staticmethod
    def extract_aligned_face(img_path, detector="retinaface"):
        face_data = detection.extract_faces(img_path,
                                              detector_backend=detector,
                                              enforce_detection=True,
                                              align=True)[0]
        face = face_data["face"]
        rgb = face[:, :, ::-1]
        return np.absolute(rgb * 255).astype("uint8")

    # ---------------  live camera  ---------------
    def analyze_frame(self, frame_bgr, gamma_lut=None):
        if gamma_lut is not None:
            frame_bgr = cv2.LUT(frame_bgr, gamma_lut)

        # Skip emotion analysis for speed
        try:
            matches = DeepFace.find(frame_bgr,
                            db_path=self.db_path,
                            model_name=self.model_name,
                            detector_backend=self.detector_backend,
                            enforce_detection=False,
                            silent=True)
        except Exception:
            matches = []

        results = []
        if matches:
            for idx, match_df in enumerate(matches):
                result = {}
                if len(match_df) > 0:
                    folder = os.path.dirname(match_df.iloc[0]["identity"])
                    result["identity"] = folder
                    result["distance"] = match_df.iloc[0]["distance"]
                    result["name"] = os.path.basename(folder)
                    result["region"] = {
                        "x": int(match_df.iloc[0].get("source_x", 0)),
                        "y": int(match_df.iloc[0].get("source_y", 0)),
                        "w": int(match_df.iloc[0].get("source_w", 100)),
                        "h": int(match_df.iloc[0].get("source_h", 100))
                    }
                else:
                    result["identity"] = "Unknown"
                    result["name"] = "Unknown"
                    result["distance"] = 0
                    result["region"] = {"x": 0, "y": 0, "w": 100, "h": 100}
                results.append(result)
        
        return results


class ForensicsService:
    """Digital-forensics helpers (hash, exif, ela, gps, strings, thumbnail)."""
    @staticmethod
    def file_hash(file_path):
        import hashlib
        with open(file_path, "rb") as f:
            data = f.read()
        return {"MD5": hashlib.md5(data).hexdigest(),
                "SHA-1": hashlib.sha1(data).hexdigest(),
                "SHA-256": hashlib.sha256(data).hexdigest()}

    @staticmethod
    def exif_data(file_path):
        from PIL import Image
        from PIL.ExifTags import TAGS
        try:
            img = Image.open(file_path)
            exif = img.getexif()
            if exif:
                return {TAGS.get(k, k): v for k, v in exif.items()}
            return {}
        except (AttributeError, Exception):
            return {}

    @staticmethod
    def gps_info(file_path):
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
        exif = ForensicsService.exif_data(file_path)
        gps = exif.get("GPSInfo", {})
        return {GPSTAGS.get(k, k): v for k, v in gps.items()} if gps else {}

    @staticmethod
    def ela_image(file_path, quality=95):
        import cv2
        img = cv2.imread(file_path)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        resaved = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(img, resaved)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        cv2.normalize(diff, diff, 0, 255, cv2.NORM_MINMAX)
        # return both PIL image and numeric score
        from PIL import Image
        #score = float(np.mean(diff[diff >= np.percentile(diff, 95)]))
        mask = diff >= np.percentile(diff, 95)
        if np.any(mask):
            score = float(np.mean(diff[mask]))
        else:
            score = 0.0
        return Image.fromarray(diff, mode="L"), score

    @staticmethod
    def strings(file_path, min_len=4):
        import re
        with open(file_path, "rb") as f:
            data = f.read()
        found = re.findall(b"[\x20-\x7E]{%d,}" % min_len, data)
        return [s.decode("ascii", errors="ignore") for s in found]

    @staticmethod
    def has_thumbnail(file_path):
        from PIL import Image
        try:
            img = Image.open(file_path)
            exif = img.getexif()
            return 0x501B in exif  # Thumbnail tag
        except (AttributeError, Exception):
            return False


# ---------- 1st tab: your original face-verification UI ----------
# ---------- 1st tab: Face Verification UI ----------
class FaceVerificationTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.models = ["VGG-Face", "Facenet", "ArcFace", "OpenFace", "DeepID", "Dlib", "SFace"]
        
        # Thread-safe queue for cropped face updates
        self._pending_cropped_updates = []

        # ----------  UI build ----------
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Top Control Panel ---
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        top_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(top_frame, text="Image A:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=(0, 10), pady=10, sticky="w")
        self.img1_entry = ctk.CTkEntry(top_frame, placeholder_text="Select first image...")
        self.img1_entry.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ew")
        ctk.CTkButton(top_frame, text="Browse...", width=100, command=lambda: self.browse_file(self.img1_entry, self.img_label_a, self.cropped_label_a)).grid(row=0, column=2)

        ctk.CTkLabel(top_frame, text="Image B:", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, padx=(0, 10), pady=10, sticky="w")
        self.img2_entry = ctk.CTkEntry(top_frame, placeholder_text="Select second image...")
        self.img2_entry.grid(row=1, column=1, padx=(0, 10), pady=10, sticky="ew")
        ctk.CTkButton(top_frame, text="Browse...", width=100, command=lambda: self.browse_file(self.img2_entry, self.img_label_b, self.cropped_label_b)).grid(row=1, column=2)

        # --- Main Content Area (Scrollable) ---
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        scroll.grid_columnconfigure(0, weight=1)

        # --- Image Display ---
        disp_frame = ctk.CTkFrame(scroll)
        disp_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        disp_frame.grid_columnconfigure([0, 3], weight=1)
        disp_frame.grid_columnconfigure([1, 2], weight=0)  # Cropped images are fixed size

        self.img_label_a = ctk.CTkLabel(disp_frame, text="Image A", text_color="gray50", fg_color="gray17", height=200, corner_radius=8)
        self.img_label_a.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.cropped_label_a = ctk.CTkLabel(disp_frame, text="Face A", text_color="gray50", fg_color="gray17", height=120, width=120, corner_radius=8)
        self.cropped_label_a.grid(row=0, column=1, padx=5, pady=10)

        self.cropped_label_b = ctk.CTkLabel(disp_frame, text="Face B", text_color="gray50", fg_color="gray17", height=120, width=120, corner_radius=8)
        self.cropped_label_b.grid(row=0, column=2, padx=5, pady=10)

        self.img_label_b = ctk.CTkLabel(disp_frame, text="Image B", text_color="gray50", fg_color="gray17", height=200, corner_radius=8)
        self.img_label_b.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")

        # --- Model Selection & Action ---
        action_frame = ctk.CTkFrame(scroll)
        action_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        action_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(action_frame, text="Select Model to Compare:", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 5))
        btn_box = ctk.CTkFrame(action_frame)
        btn_box.pack(pady=10, padx=10)
        for i, m in enumerate(self.models):
            ctk.CTkButton(btn_box, text=m, command=lambda m=m: self.compare_faces(m), width=90).grid(row=0, column=i, padx=3, pady=5)

        # --- Output Display ---
        out_frame = ctk.CTkFrame(scroll, fg_color="gray17")
        out_frame.grid(row=2, column=0, sticky="ew")
        out_frame.grid_columnconfigure(0, weight=1)

        self.verified_label = ctk.CTkLabel(out_frame, text="Awaiting Comparison...", font=ctk.CTkFont(size=16, weight="bold"), text_color="gray60")
        self.verified_label.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10))

        self.distance_label = ctk.CTkLabel(out_frame, text="Distance: --", text_color="gray50")
        self.distance_label.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        self.similarity_label = ctk.CTkLabel(out_frame, text="Similarity: --", text_color="gray50")
        self.similarity_label.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="w")
        
        # Start polling for cropped face updates
        self._process_cropped_updates()

    # ----------  UI update polling  ----------
    def _process_cropped_updates(self):
        """Process pending cropped face updates on main thread (thread-safe)"""
        while self._pending_cropped_updates:
            label, ctk_img, error_text = self._pending_cropped_updates.pop(0)
            if ctk_img:
                label.configure(image=ctk_img, text="")
                label.ctk_img = ctk_img  # Keep reference
            elif error_text:
                label.configure(text=error_text, image=None)
        
        # Schedule next check
        self.after(100, self._process_cropped_updates)

    # ----------  service calls  ----------
    def browse_file(self, entry_widget, image_label, cropped_label):
        """Browse for an image file and display both full and aligned face"""
        path = tk.filedialog.askopenfilename(
            title="Select an Image", 
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not path:
            return
            
        # Update entry
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, path)
        
        # Display full image preview
        try:
            img = Image.open(path)
            img.thumbnail((150, 150), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img, size=img.size)
            image_label.configure(image=ctk_img, text="")
            image_label.ctk_img = ctk_img
        except Exception as e:
            image_label.configure(text=f"Error: {e}", image=None)
        
        # Display aligned face - run extraction in background, update UI on main thread
        def extract_and_update():
            try:
                arr = FaceRecognitionService.extract_aligned_face(path)
                pil = Image.fromarray(arr).resize((100, 100), Image.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=pil, size=(100, 100))
                # Store result and schedule UI update
                self._pending_cropped_updates.append((cropped_label, ctk_img, None))
            except Exception as e:
                # Store error and schedule UI update
                self._pending_cropped_updates.append((cropped_label, None, "No Face\nDetected!"))
        
        threading.Thread(target=extract_and_update, daemon=True).start()

    def compare_faces(self, model_name):
        """Compare two faces using the specified model"""
        a = self.img1_entry.get()
        b = self.img2_entry.get()
        
        if not os.path.isfile(a) or not os.path.isfile(b):
            self.verified_label.configure(text="ERROR: Please select two valid image files.", text_color="#F44336")
            return

        def comparison_worker():
            try:
                # Apply custom threshold only for ArcFace model
                kwargs = {}
                if model_name == "ArcFace":
                    kwargs['threshold'] = 0.50
                
                res = FaceRecognitionService.verify_two_images(a, b, model_name, **kwargs)
                verified, dist, thr = res["verified"], res["distance"], res["threshold"]
                #sim = (1 - dist) * 100 # similarity can exceed 100% if dist < 0.
                sim = max(0, min(100, (1 - dist) * 100)) #
                color = "#4CAF50" if verified else "#F44336"

                # Schedule UI updates on the main thread
                self.after(0, self.update_comparison_results, verified, dist, thr, sim, color, model_name)

            except Exception as e:
                error_msg = str(e).replace("`", "'").replace("b'", "")
                if len(error_msg) > 60:
                    error_msg = error_msg[:60] + "..."
                self.after(0, self.update_comparison_error, error_msg)

        threading.Thread(target=comparison_worker, daemon=True).start()

    def update_comparison_results(self, verified, dist, thr, sim, color, model_name):
        self.verified_label.configure(text=f"Result: {'VERIFIED' if verified else 'NOT VERIFIED'}", text_color=color)
        self.distance_label.configure(text=f"Distance: {dist:.4f} (Threshold: {thr:.2f})")
        self.similarity_label.configure(text=f"Similarity Score: {sim:.2f}% (Model: {model_name})")

    def update_comparison_error(self, error_msg):
        self.verified_label.configure(text=f"ERROR: {error_msg}", text_color="#F44336")
        self.distance_label.configure(text="Distance: --")
        self.similarity_label.configure(text="Similarity: --")

# ---------- 2nd tab: Digital Forensics Tools ----------
class DigitalForensicsTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(0, weight=1)
        self.tabview = ctk.CTkTabview(self); self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        for t in ("Summary", "Hash", "EXIF", "ELA", "GPS", "Thumbnail", "Strings"):
            self.tabview.add(t)
        self.current_image_path = None
        self.init_summary_tab(); self.init_hash_tab(); self.init_exif_tab()
        self.init_ela_tab(); self.init_gps_tab(); self.init_thumbnail_tab(); self.init_strings_tab()

    # ----------  Summary tab  ----------
    def init_summary_tab(self):
        tab = self.tabview.tab("Summary")
        tab.grid_columnconfigure(0, weight=1) 
        tab.grid_rowconfigure(1, weight=1) 
 
        # --- Input Frame --- 
        inp_frame = ctk.CTkFrame(tab) 
        inp_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew") 
        inp_frame.grid_columnconfigure(1, weight=1) 
 
        ctk.CTkLabel(inp_frame, text="Image File:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=15, pady=15, sticky="w") 
        self.summary_image_entry = ctk.CTkEntry(inp_frame, placeholder_text="Select an image to run a full forensic analysis...") 
        self.summary_image_entry.grid(row=0, column=1, padx=15, pady=15, sticky="ew") 
        ctk.CTkButton(inp_frame, text="Browse...", width=100, command=self.browse_summary_image).grid(row=0, column=2, padx=15, pady=15) 
        self.analyze_all_btn = ctk.CTkButton(inp_frame, text="Run Full Analysis", height=35, command=self.run_full_summary) 
        self.analyze_all_btn.grid(row=1, column=0, columnspan=3, padx=15, pady=(0, 15), sticky="ew") 
 
        # --- Output Frame --- 
        out_frame = ctk.CTkFrame(tab) 
        out_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew") 
        out_frame.grid_columnconfigure(0, weight=1) 
        out_frame.grid_rowconfigure(0, weight=1) 
 
        self.summary_output = ctk.CTkTextbox(out_frame, wrap="word", font=ctk.CTkFont(family="monospace", size=13)) 
        self.summary_output.pack(padx=10, pady=10, fill="both", expand=True) 
        self.summary_output.insert("1.0", "Select an image and click 'Run Full Analysis' to begin.\n") 

    def browse_summary_image(self):
        file_path = tk.filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        if file_path:
            self.summary_image_entry.delete(0, tk.END); self.summary_image_entry.insert(0, file_path); self.current_image_path = file_path

    def run_full_summary(self):
        path = self.summary_image_entry.get()
        if not path or not os.path.exists(path):
            tkmsg.showerror("Error", "Please select a valid image file.")
            return

        self.analyze_all_btn.configure(state="disabled", text="Analyzing...")
        self.summary_output.delete("1.0", "end")
        self.summary_output.insert("1.0", f"Running comprehensive analysis on {os.path.basename(path)}...\n\n")

        def worker():
            try:
                # --- Analysis ---
                hashes = ForensicsService.file_hash(path)
                exif = ForensicsService.exif_data(path)
                gps = ForensicsService.gps_info(path)
                thumb = ForensicsService.has_thumbnail(path)
                
                # --- Formatting ---
                f_size = f"{os.path.getsize(path) / 1024:.2f} KB"
                f_hashes = "\n".join([f"  {k}: {v}" for k, v in hashes.items()])
                f_exif = "\n".join([f"  {k}: {v}" for k, v in exif.items()]) if exif else "  No EXIF data found."
                f_gps = "\n".join([f"  {k}: {v}" for k, v in gps.items()]) if gps else "  No GPS data found."
                f_thumb = "  EXIF thumbnail data may exist." if thumb else "  No thumbnail data found."

                # --- Display ---
                summary = (
                    f"=== FORENSIC SUMMARY ===\n"
                    f"File:    {os.path.basename(path)}\n"
                    f"Size:    {f_size}\n"
                    f"--------------------------------\n"
                    f"FILE HASHES:\n{f_hashes}\n"
                    f"--------------------------------\n"
                    f"EXIF METADATA:\n{f_exif}\n"
                    f"--------------------------------\n"
                    f"GPS INFORMATION:\n{f_gps}\n"
                    f"--------------------------------\n"
                    f"THUMBNAIL:\n{f_thumb}\n"
                    f"--------------------------------\n"
                    f"ANALYSIS COMPLETE"
                )
                self.after(0, lambda: self.summary_output.delete("1.0", "end"))
                self.after(0, lambda: self.summary_output.insert("1.0", summary))

            except Exception as e:
                self.after(0, lambda: self.summary_output.insert("end", f"\n\nERROR: {e}"))
            finally:
                self.after(0, lambda: self.analyze_all_btn.configure(state="normal", text="Run Full Analysis"))

        threading.Thread(target=worker, daemon=True).start()


    # ----------  generic single-tab builder  ----------
    def _init_simple_tab(self, tab_name, analysis_func):
        tab = self.tabview.tab(tab_name)
        tab.grid_columnconfigure(1, weight=1); tab.grid_rowconfigure(1, weight=1) # Main grid config

        # --- Input Frame ---
        inp_frame = ctk.CTkFrame(tab)
        inp_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="ew")
        inp_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(inp_frame, text="Image:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=15, pady=15, sticky="w")
        entry = ctk.CTkEntry(inp_frame, placeholder_text=f"Select image for {tab_name} analysis...")
        entry.grid(row=0, column=1, padx=15, pady=15, sticky="ew")
        setattr(self, f"{tab_name.lower()}_entry", entry)
        ctk.CTkButton(inp_frame, text="Browse...", width=100, command=lambda e=entry: self._browse_for_tab(e)).grid(row=0, column=2, padx=15)

        # --- Preview Frame ---
        preview_frame = ctk.CTkFrame(tab)
        preview_frame.grid(row=1, column=0, padx=(20, 10), pady=(0, 20), sticky="nsew")
        preview_frame.grid_propagate(False) # Prevent resizing
        preview_frame.grid_columnconfigure(0, weight=1); preview_frame.grid_rowconfigure(0, weight=1)
        preview = ctk.CTkLabel(preview_frame, text="", fg_color="gray17")
        preview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        setattr(self, f"{tab_name.lower()}_preview", preview)
        tab.grid_columnconfigure(0, weight=1)


        # --- Output Frame ---
        out_frame = ctk.CTkFrame(tab)
        out_frame.grid(row=1, column=1, padx=(10, 20), pady=(0, 20), sticky="nsew")
        out_frame.grid_columnconfigure(0, weight=1); out_frame.grid_rowconfigure(0, weight=1)
        txt = ctk.CTkTextbox(out_frame, wrap="word", font=ctk.CTkFont(family="monospace"))
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        setattr(self, f"{tab_name.lower()}_output", txt)
        tab.grid_columnconfigure(1, weight=1)

        # --- Analyze Button ---
        ctk.CTkButton(tab, text=f"Run {tab_name} Analysis", height=35, command=lambda e=entry: analysis_func(e)).grid(row=2, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="ew")


    def _browse_for_tab(self, entry):
        file_path = tk.filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        if not file_path:
            return
        entry.delete(0, tk.END); entry.insert(0, file_path)

        # Also update the preview image for the specific tab
        tab_name = ""
        if entry == self.ela_entry: tab_name = "ela"
        elif entry == self.hash_entry: tab_name = "hash"
        elif entry == self.exif_entry: tab_name = "exif"
        elif entry == self.gps_entry: tab_name = "gps"
        elif entry == self.thumbnail_entry: tab_name = "thumbnail"
        elif entry == self.strings_entry: tab_name = "strings"


        if tab_name:
            preview_label = getattr(self, f"{tab_name}_preview")
            try:
                img = Image.open(file_path)
                img.thumbnail((300, 300), Image.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=img, size=img.size)
                preview_label.configure(image=ctk_img, text="")
                preview_label.ctk_img = ctk_img
            except Exception:
                preview_label.configure(text="Preview\nError", image=None)

    # ----------  individual tabs  ----------
    def init_hash_tab(self): self._init_simple_tab("Hash", self.run_hash_analysis)
    def init_exif_tab(self): self._init_simple_tab("EXIF", self.run_exif_analysis)
    def init_ela_tab(self): self._init_simple_tab("ELA", self.run_ela_analysis)
    def init_gps_tab(self): self._init_simple_tab("GPS", self.run_gps_analysis)
    def init_thumbnail_tab(self): self._init_simple_tab("Thumbnail", self.run_thumbnail_analysis)
    def init_strings_tab(self): self._init_simple_tab("Strings", self.run_strings_analysis)

    def run_hash_analysis(self, entry):
        path = entry.get()
        if not os.path.isfile(path): self.hash_output.insert("1.0", "ERROR: Invalid file path.\n"); return
        self.hash_output.delete("1.0", "end")
        def worker():
            hashes = ForensicsService.file_hash(path)
            self.after(0, lambda: self.hash_output.delete("1.0", "end"))
            for k, v in hashes.items(): self.after(0, lambda k=k, v=v: self.hash_output.insert("end", f"{k}: {v}\n"))
        threading.Thread(target=worker, daemon=True).start()

    def run_exif_analysis(self, entry):
        path = entry.get()
        if not os.path.isfile(path): self.exif_output.insert("1.0", "ERROR: Invalid file path.\n"); return
        self.exif_output.delete("1.0", "end")
        def worker():
            data = ForensicsService.exif_data(path)
            self.after(0, lambda: self.exif_output.delete("1.0", "end"))
            if data:
                for k, v in data.items(): self.after(0, lambda k=k, v=v: self.exif_output.insert("end", f"{k}: {v}\n"))
            else: self.after(0, lambda: self.exif_output.insert("end", "No EXIF data found.\n"))
        threading.Thread(target=worker, daemon=True).start()

    def run_ela_analysis(self, entry):
        path = entry.get()
        if not os.path.isfile(path):
            self.ela_output.insert("1.0", "ERROR: file not found\n")
            return
        self.ela_output.delete("1.0", "end")
        self.ela_output.insert("1.0", "Running ELA analysis...")

        def worker():
            try:
                pil_img, score = ForensicsService.ela_image(path)
                self.after(0, self.update_ela_preview, pil_img, score)
            except Exception as e:
                self.after(0, lambda: self.ela_output.insert("end", f"\n\nELA analysis failed: {str(e)}\n"))

        threading.Thread(target=worker, daemon=True).start()

    def update_ela_preview(self, pil_img, score):
        frame_width = self.ela_preview.winfo_width()
        frame_height = self.ela_preview.winfo_height()
        
        # Resize image to fit frame while maintaining aspect ratio
        img_aspect = pil_img.width / pil_img.height
        frame_aspect = frame_width / frame_height
        
        if img_aspect > frame_aspect:
            # Image is wider than the frame, so width is the limiting factor
            new_width = frame_width
            new_height = int(new_width / img_aspect)
        else:
            # Image is taller than the frame, so height is the limiting factor
            new_height = frame_height
            new_width = int(new_height * img_aspect)
            
        resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        ctk_img = ctk.CTkImage(light_image=resized_img, size=(new_width, new_height))
        
        self.ela_preview.configure(image=ctk_img, text="")
        self.ela_preview.ctk_img = ctk_img
        
        self.ela_output.delete("1.0", "end")
        self.ela_output.insert("end", f"ELA score (higher is more suspicious): {score:.2f}\n\nBright areas indicate potential manipulation or different compression levels.")

    def run_gps_analysis(self, entry):
        path = entry.get()
        if not os.path.isfile(path): self.gps_output.insert("1.0", "ERROR: Invalid file path.\n"); return
        self.gps_output.delete("1.0", "end")
        def worker():
            gps = ForensicsService.gps_info(path)
            self.after(0, lambda: self.gps_output.delete("1.0", "end"))
            if gps:
                for k, v in gps.items(): self.after(0, lambda k=k, v=v: self.gps_output.insert("end", f"{k}: {v}\n"))
            else: self.after(0, lambda: self.gps_output.insert("end", "No GPS data found in EXIF metadata.\n"))
        threading.Thread(target=worker, daemon=True).start()

    def run_thumbnail_analysis(self, entry):
        path = entry.get()
        if not os.path.isfile(path): self.thumbnail_output.insert("1.0", "ERROR: Invalid file path.\n"); return
        self.thumbnail_output.delete("1.0", "end")
        def worker():
            has_thumb = ForensicsService.has_thumbnail(path)
            self.after(0, lambda: self.thumbnail_output.delete("1.0", "end"))
            self.after(0, lambda: self.thumbnail_output.insert("end", "Result: EXIF data is present, which may contain a thumbnail." if has_thumb else "Result: No EXIF data found, so no embedded thumbnail."))
        threading.Thread(target=worker, daemon=True).start()

    def run_strings_analysis(self, entry):
        path = entry.get()
        if not os.path.isfile(path): self.strings_output.insert("1.0", "ERROR: Invalid file path.\n"); return
        self.strings_output.delete("1.0", "end")
        self.strings_output.insert("1.0", "Searching for embedded strings...")
        def worker():
            try:
                strings = ForensicsService.strings(path)
                self.after(0, lambda: self.strings_output.delete("1.0", "end"))
                if not strings:
                    self.after(0, lambda: self.strings_output.insert("end", "No printable strings found."))
                    return
                
                self.after(0, lambda: self.strings_output.insert("end", f"Found {len(strings)} printable text strings (showing first 200):\n\n"))
                for s in strings[:200]: self.after(0, lambda s=s: self.strings_output.insert("end", f"{s}\n"))
                if len(strings) > 200: self.after(0, lambda: self.strings_output.insert("end", f"\n... and {len(strings)-200} more."))
            except Exception as e:
                self.after(0, lambda: self.strings_output.insert("end", f"\n\nError reading strings: {e}"))
        threading.Thread(target=worker, daemon=True).start()

# ---------- 3rd tab: Live Face Recognition + Analytics ----------
# Updated LiveRecognitionTab class with fixes:
class LiveRecognitionTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        # ----------  services  ----------
        self.cam_svc = CameraService()
        self.recog_svc = FaceRecognitionService(db_path="face_database",
                                                model_name="VGG-Face",
                                                detector_backend="retinaface")  # Changed to opencv for speed

        # ----------  analytics objects  ----------
        self.history = []
        self.attendance = {}  # Changed to dict to store first detection time
        self.accuracy_buffer = []
        self.MAX_ACCURACY_BUF = 30
        self.MAX_HISTORY_SIZE = 1000

        # ----------  GUI const  ----------
        self.is_running = False
        self.frame_count = 0
        self.process_every_n_frames = 5  # Increased for better performance
        self.last_results = []
        self.lut = np.array([((i / 255.0) ** (1.0 / 1.8)) * 255
                             for i in np.arange(256)], dtype="uint8")
        
        # ----------  thread safety  ----------
        self.ui_update_queue = Queue()
        self.results_lock = threading.Lock()

        # ----------  build UI  ----------
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_control_panel()
        self._build_video_panel()
        self._build_analytics_panel()
        self._reset_daily_stats()
        
        # Start UI update polling
        self._process_ui_queue()

    # ----------  UI builders  ----------
    def _build_control_panel(self):
        ctrl = ctk.CTkFrame(self)
        ctrl.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="ew")
        ctrl.grid_columnconfigure(1, weight=1)

        # --- Left Side: DB, Model, and Camera ---
        config_frame = ctk.CTkFrame(ctrl, fg_color="transparent")
        config_frame.grid(row=0, column=0, padx=(10, 20), pady=10, sticky="w")
        config_frame.grid_columnconfigure(1, weight=1)
    
        # Database
        ctk.CTkLabel(config_frame, text="Database:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=(0,10), pady=5, sticky="w")
        self.db_entry = ctk.CTkEntry(config_frame, placeholder_text="face_database", width=200)
        self.db_entry.insert(0, "face_database")
        self.db_entry.grid(row=0, column=1, pady=5, sticky="ew")
    
        # Model
        ctk.CTkLabel(config_frame, text="Model:", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, padx=(0,10), pady=5, sticky="w")
        self.model_combo = ctk.CTkComboBox(config_frame, values=["VGG-Face", "Facenet", "ArcFace", "OpenFace"], state="readonly", width=200)
        self.model_combo.set("VGG-Face")
        self.model_combo.grid(row=1, column=1, pady=5, sticky="ew")
    
        # Camera Selection
        ctk.CTkLabel(config_frame, text="Camera:", font=ctk.CTkFont(weight="bold")).grid(row=2, column=0, padx=(0,10), pady=5, sticky="w")
    
        # Detect available cameras
        available_cameras = CameraService.get_available_cameras()
        if not available_cameras:
            available_cameras = [0]  # Fallback to default
    
        camera_options = [f"Camera {i}" for i in available_cameras]
        self.camera_combo = ctk.CTkComboBox(config_frame, values=camera_options, state="readonly", width=200)
        self.camera_combo.set(camera_options[0])
        self.camera_combo.grid(row=2, column=1, pady=5, sticky="ew")
    
        # Store the mapping
        self.available_cameras = available_cameras
    
        # --- Right Side: Camera Controls ---
        btn_frame = ctk.CTkFrame(ctrl, fg_color="transparent")
        btn_frame.grid(row=0, column=2, padx=(20, 10), pady=10, sticky="e")
    
        self.start_btn = ctk.CTkButton(btn_frame, text="▶ Start Camera", width=150, height=35, command=self.start_camera, fg_color="#28a745", hover_color="#218838")
        self.start_btn.pack(pady=5, fill="x")
    
        self.stop_btn = ctk.CTkButton(btn_frame, text="⏹ Stop Camera", width=150, height=35, command=self.stop_camera, fg_color="#dc3545", hover_color="#c82333", state="disabled")
        self.stop_btn.pack(pady=5, fill="x")
    
        self.snapshot_btn = ctk.CTkButton(btn_frame, text="📷 Snapshot", width=150, height=35, command=self.take_snapshot, state="disabled")
        self.snapshot_btn.pack(pady=5, fill="x")

    def _build_video_panel(self):
        disp = ctk.CTkFrame(self)
        disp.grid(row=1, column=0, padx=(20, 10), pady=(0, 20), sticky="nsew")
        disp.grid_columnconfigure(0, weight=1)
        disp.grid_rowconfigure(0, weight=1)
        self.video_label = ctk.CTkLabel(disp, text="", fg_color="gray17")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_analytics_panel(self):
        right_panel = ctk.CTkScrollableFrame(self, fg_color="transparent")
        right_panel.grid(row=1, column=1, padx=(10, 20), pady=(0, 20), sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)
        
        info_frame = ctk.CTkFrame(right_panel)
        info_frame.pack(fill="x", expand=True, pady=(0, 15))
        ctk.CTkLabel(info_frame, text="Live Detections", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,5), padx=10)
        self.info_text = ctk.CTkTextbox(info_frame, height=150, wrap="word", font=ctk.CTkFont(family="monospace", size=12))
        self.info_text.pack(fill="x", expand=True, padx=10, pady=(0, 10))
        self.info_text.insert("1.0", "Start camera to begin live recognition.")

        stats_frame = ctk.CTkFrame(right_panel)
        stats_frame.pack(fill="x", expand=True, pady=15)
        stats_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(stats_frame, text="Real-time Stats", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, columnspan=2, pady=(10,5), padx=10)
        ctk.CTkLabel(stats_frame, text="Avg. Accuracy:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.accuracy_label = ctk.CTkLabel(stats_frame, text="-- %", font=ctk.CTkFont(weight="bold"))
        self.accuracy_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        ctk.CTkLabel(stats_frame, text="Attendance:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.attendance_label = ctk.CTkLabel(stats_frame, text="0", font=ctk.CTkFont(weight="bold"))
        self.attendance_label.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        actions_frame = ctk.CTkFrame(right_panel)
        actions_frame.pack(fill="x", expand=True, pady=15)
        actions_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(actions_frame, text="Analysis & Export", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,5))
        ctk.CTkButton(actions_frame, text="Show Attendee List", command=self.show_attendee_list).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(actions_frame, text="Detection History", command=self.plot_history).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(actions_frame, text="Export CSV", command=self.export_csv).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(actions_frame, text="Reset Daily Stats", command=self._reset_daily_stats, fg_color="gray50", hover_color="gray40").pack(fill="x", padx=10, pady=(5,10))

    # ----------  camera control  ----------
    def start_camera(self):
        if self.is_running:
            return
        
        self.recog_svc.db_path = self.db_entry.get()
        self.recog_svc.model_name = self.model_combo.get()
    
        # Get selected camera
        try:
            selected = self.camera_combo.get()
            camera_idx = int(selected.split()[-1])
            camera_src = camera_idx
        except (ValueError, IndexError):
            camera_src = 0
    
        # Update camera source
        self.cam_svc.src = camera_src
    
        if not os.path.exists(self.recog_svc.db_path):
            self.info_text.delete("1.0", "end")
            self.info_text.insert("1.0", f"ERROR: Database folder '{self.recog_svc.db_path}' not found!")
            return
    
        try:
            if not self.cam_svc.start():
                self.info_text.delete("1.0", "end")
                self.info_text.insert("1.0", "ERROR: Could not open camera!")
                return
        except Exception as e:
            self.info_text.delete("1.0", "end")
            self.info_text.insert("1.0", f"ERROR: Camera failed: {str(e)}")
            return
        
        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.snapshot_btn.configure(state="normal")
        self.model_combo.configure(state="disabled")
        self.db_entry.configure(state="disabled")
        self.camera_combo.configure(state="disabled")

        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", f"Camera started.\nCamera: {self.camera_combo.get()}\nModel: {self.recog_svc.model_name}\nDB: {self.recog_svc.db_path}\nSearching for faces...")
    
        threading.Thread(target=self.update_frame, daemon=True).start()

    def stop_camera(self):
        self.is_running = False
        self.cam_svc.stop()
        
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.snapshot_btn.configure(state="disabled")
        self.model_combo.configure(state="readonly")
        self.db_entry.configure(state="normal")
        self.camera_combo.configure(state="readonly")
        
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", "Camera stopped.")

    def take_snapshot(self):
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{ts}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.info_text.insert("end", f"\n📷 Snapshot: {filename}\n")
            self.info_text.see("end")

    # ----------  video loop  ----------
    def update_frame(self):
        try:
            while self.is_running:
                ret, frame = self.cam_svc.read()
                if not ret:
                    break
                    
                self.current_frame = frame.copy()
                
                if self.frame_count % self.process_every_n_frames == 0:
                    enhanced = cv2.LUT(frame, self.lut)
                    results = self.recog_svc.analyze_frame(enhanced, gamma_lut=None)
                    
                    with self.results_lock:
                        self.last_results = results
                    
                    for face in results:
                        name = face.get('name', 'Unknown')
                        distance = face.get('distance', 0)
                        self._record_detection(name, distance)
                    
                    info_text = self.build_info_text()
                    self.ui_update_queue.put(('info', info_text))
                
                with self.results_lock:
                    results_copy = self.last_results.copy()
                
                for face in results_copy:
                    self.draw_face_box(frame, face)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.thumbnail((640, 480), Image.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=img, size=img.size)
                
                self.ui_update_queue.put(('video', ctk_img))
                
                self.frame_count += 1
        except Exception as e:
            print(f"Frame update error: {e}")
        finally:
            self.is_running = False
            self.cam_svc.stop()

    def draw_face_box(self, frame, face_data):
        region = face_data.get('region', {})
        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 100), region.get('h', 100)
        identity = face_data.get('name', 'Unknown')
        
        color = (46, 204, 113) if identity != 'Unknown' else (255, 107, 53)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        (lw, lh), _ = cv2.getTextSize(identity, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - lh - 8), (x + lw + 6, y), color, -1)
        cv2.putText(frame, identity, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ----------  analytics  ----------
    def _record_detection(self, name, distance):
        """Record a single detection event for analytics"""
        if name == "Unknown":
            return
        
        now = datetime.now()
        
        # Record in history
        self.history.append({
            "time": now,
            "name": name,
            "distance": distance
        })
        
        if len(self.history) > self.MAX_HISTORY_SIZE:
            self.history.pop(0)
        
        # Update attendance with first detection time
        if name not in self.attendance:
            self.attendance[name] = now
        
        # Update rolling accuracy buffer
        self.accuracy_buffer.append(distance)
        if len(self.accuracy_buffer) > self.MAX_ACCURACY_BUF:
            self.accuracy_buffer.pop(0)
        
        if self.accuracy_buffer:
            avg_dist = np.mean(self.accuracy_buffer)
            accuracy = max(0, min(100, 100 - avg_dist * 10))
            self.ui_update_queue.put(('accuracy', f"{accuracy:.1f} %"))
        
        self.ui_update_queue.put(('attendance', str(len(self.attendance))))

    def build_info_text(self):
        """Build the info text display"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        if not self.last_results:
            return f"[{current_time}] No faces detected."
        
        text = f"[{current_time}] Detected {len(self.last_results)} face(s):\n"
        for face in self.last_results:
            name = face.get('name', 'Unknown')
            distance = face.get('distance', 0)
            accuracy = max(0, min(100, 100 - distance * 10))
            text += f"  - {name} (Acc: {accuracy:.1f}%)\n"
        
        return text

    def _reset_daily_stats(self):
        """Reset all daily statistics"""
        self.history = []
        self.attendance = {}
        self.accuracy_buffer = []
        self.accuracy_label.configure(text="-- %")
        self.attendance_label.configure(text="0")
        tkmsg.showinfo("Reset", "Daily statistics cleared.")

    # ----------  UI update helpers  ----------
    def _process_ui_queue(self):
        """Process UI updates from background thread"""
        try:
            while not self.ui_update_queue.empty():
                update_type, data = self.ui_update_queue.get_nowait()
                if update_type == 'video':
                    self.video_label.configure(image=data, text="")
                    self.video_label.ctk_img = data
                elif update_type == 'info':
                    self.info_text.delete("1.0", "end")
                    self.info_text.insert("1.0", data)
                    self.info_text.see("end")
                elif update_type == 'accuracy':
                    self.accuracy_label.configure(text=data)
                elif update_type == 'attendance':
                    self.attendance_label.configure(text=data)
        except:
            pass
        finally:
            if self.winfo_exists():
                self.after(50, self._process_ui_queue)

    # ----------  new attendee list display  ----------
    def show_attendee_list(self):
        if not self.attendance:
            tkmsg.showinfo("No Data", "No attendance data collected yet.")
            return
        
        # Create dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("Attendee List")
        dialog.geometry("600x500")
        dialog.transient(self.master)
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (500 // 2)
        dialog.geometry(f"600x500+{x}+{y}")
        
        # Header
        header = ctk.CTkFrame(dialog)
        header.pack(fill="x", padx=20, pady=(20, 10))
        ctk.CTkLabel(header, text="📋 Attendee List", font=ctk.CTkFont(size=20, weight="bold")).pack()
        ctk.CTkLabel(header, text=f"Total: {len(self.attendance)} people", font=ctk.CTkFont(size=12)).pack()
        
        # Table frame
        table_frame = ctk.CTkScrollableFrame(dialog)
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)
        table_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Table headers
        ctk.CTkLabel(table_frame, text="Name", font=ctk.CTkFont(weight="bold"), anchor="w").grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkLabel(table_frame, text="First Seen", font=ctk.CTkFont(weight="bold"), anchor="w").grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        # Sort by first detection time
        sorted_attendance = sorted(self.attendance.items(), key=lambda x: x[1])
        
        # Table rows
        for idx, (name, first_time) in enumerate(sorted_attendance, start=1):
            time_str = first_time.strftime("%H:%M:%S")
            
            row_frame = ctk.CTkFrame(table_frame, fg_color=("gray80", "gray20") if idx % 2 == 0 else "transparent")
            row_frame.grid(row=idx, column=0, columnspan=2, sticky="ew", pady=1)
            row_frame.grid_columnconfigure((0, 1), weight=1)
            
            ctk.CTkLabel(row_frame, text=name, anchor="w").grid(row=0, column=0, padx=10, pady=5, sticky="w")
            ctk.CTkLabel(row_frame, text=time_str, anchor="w").grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        # Close button
        ctk.CTkButton(dialog, text="Close", command=dialog.destroy, width=120).pack(pady=(10, 20))

    # ----------  plotting and export  ----------
    def plot_history(self):
        if not self.history:
            tkmsg.showinfo("No Data", "No history data collected yet.")
            return

        times = [item['time'] for item in self.history]
        names = [item['name'] for item in self.history]
        
        unique_names = sorted(list(set(names)))
        name_to_id = {name: i + 1 for i, name in enumerate(unique_names)}
        
        y_values = [name_to_id[name] for name in names]
        
        plt.figure(figsize=(12, 6))
        plt.scatter(times, y_values, c='b', alpha=0.6, marker='o')
        plt.yticks(list(name_to_id.values()), list(name_to_id.keys()))
        plt.xlabel("Time")
        plt.title("Detection History Timeline")
        plt.grid(True, axis='y', linestyle='--')
        plt.tight_layout()
        plt.show()

    def export_csv(self):
        if not self.history:
            tkmsg.showinfo("No Data", "No history data to export.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_history_{ts}.csv"
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Time', 'Name', 'Distance']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for item in self.history:
                    writer.writerow({
                        'Time': item['time'].isoformat(),
                        'Name': item['name'],
                        'Distance': f"{item['distance']:.4f}"
                    })
            tkmsg.showinfo("Export Successful", f"History exported to {filename}")
        except Exception as e:
            tkmsg.showerror("Export Error", f"Failed to export CSV: {e}")

    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        self.cam_svc.stop()
    
    def __del__(self):
        """Destructor - calls cleanup"""
        try:
            self.cleanup()
        except:
            pass

    # ---------- 4th tab: Add Face to Database ----------
class AddFaceDatabaseTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        
        # ----------  services  ----------
        self.cam_svc = CameraService()
        self.db_path = "face_database"
        
        # ----------  state  ----------
        self.is_running = False
        self.current_frame = None
        self.captured_frame = None
        self.ui_update_queue = Queue()
        self.frame_count = 0
        self.process_every_n_frames = 5  # Process face detection every 5 frames
        self.last_face_count = 0
        
        # ----------  build UI  ----------
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_control_panel()
        self._build_camera_panel()
        self._build_info_panel()
        
        # Start UI update polling
        self._process_ui_queue()
    
    # ----------  UI builders  ----------
    def _build_control_panel(self):  # For AddFaceDatabaseTab
        ctrl = ctk.CTkFrame(self)
        ctrl.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        ctrl.grid_columnconfigure((0, 1), weight=1)
    
        # Database path
        path_frame = ctk.CTkFrame(ctrl)
        path_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        path_frame.grid_columnconfigure(1, weight=1)
    
        ctk.CTkLabel(path_frame, text="Database Path:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.db_entry = ctk.CTkEntry(path_frame, placeholder_text="face_database")
        self.db_entry.insert(0, "face_database")
        self.db_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
    
        # Camera Selection
        camera_frame = ctk.CTkFrame(ctrl)
        camera_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        camera_frame.grid_columnconfigure(1, weight=1)
    
        ctk.CTkLabel(camera_frame, text="Select Camera:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
    
        # Detect available cameras
        available_cameras = CameraService.get_available_cameras()
        if not available_cameras:
            available_cameras = [0]  # Fallback to default
    
        camera_options = [f"Camera {i}" for i in available_cameras]
        self.camera_combo = ctk.CTkComboBox(camera_frame, values=camera_options, state="readonly")
        self.camera_combo.set(camera_options[0])
        self.camera_combo.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
    
        # Store the mapping
        self.available_cameras = available_cameras
    
        # Control buttons
        btn_box = ctk.CTkFrame(ctrl)
        btn_box.grid(row=2, column=0, columnspan=2, pady=10)
    
        self.start_btn = ctk.CTkButton(btn_box, text="▶ Start Camera", command=self.start_camera,
                                   fg_color="green", hover_color="darkgreen", width=150)
        self.start_btn.grid(row=0, column=0, padx=5)
    
        self.stop_btn = ctk.CTkButton(btn_box, text="⏹ Stop Camera", command=self.stop_camera,
                                  fg_color="red", hover_color="darkred", width=150, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)
    
        self.capture_btn = ctk.CTkButton(btn_box, text="📷 Capture & Save", command=self.capture_photo,
                                     fg_color="blue", hover_color="darkblue", width=150, state="disabled")
        self.capture_btn.grid(row=0, column=2, padx=5)
    
    def _build_camera_panel(self):
        disp = ctk.CTkFrame(self)
        disp.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        disp.grid_columnconfigure((0, 1), weight=1)
        disp.grid_rowconfigure(0, weight=1)
        
        # Live camera feed
        left_frame = ctk.CTkFrame(disp)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(left_frame, text="Live Camera Feed", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=10, pady=10)
        
        self.video_label = ctk.CTkLabel(left_frame, text="Camera feed will appear here\nClick 'Start Camera' to begin",
                                       fg_color=("gray75", "gray25"), height=520, font=ctk.CTkFont(size=14))
        self.video_label.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        
        # Captured photo preview
        right_frame = ctk.CTkFrame(disp)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(right_frame, text="Last Captured Photo", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=10, pady=10)
        
        self.captured_label = ctk.CTkLabel(right_frame, text="No photo captured yet\nClick 'Capture & Save' to take a photo",
                                          fg_color=("gray75", "gray25"), height=520, font=ctk.CTkFont(size=14))
        self.captured_label.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
    
    def _build_info_panel(self):
        info = ctk.CTkFrame(self)
        info.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
        info.grid_columnconfigure(0, weight=1)
        info.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(info, text="Instructions & Status", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=10, pady=10)
        
        self.info_text = ctk.CTkTextbox(info, wrap="word", height=100)
        self.info_text.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.info_text.insert("1.0", 
            "📋 Instructions:\n"
            "1. Click 'Start Camera' to begin\n"
            "2. Position your face in the camera view\n"
            "3. Click 'Capture & Save' to take a photo\n"
            "4. Enter a name when prompted\n"
            "5. Photo will be saved to the database\n\n"
            "💡 Tip: If you enter an existing name, the photo will be added to that person's folder."
        )
        self.info_text.configure(state="disabled")
    
    # ----------  camera control  ----------
    def start_camera(self):  # For AddFaceDatabaseTab
        if self.is_running:
            return
    
        self.db_path = self.db_entry.get()
    
        # Get selected camera index
        # CORRECT:
        try:
            # Get selected text (e.g., "Camera 0") and extract the number
            selected = self.camera_combo.get()  # Returns "Camera 0", "Camera 1", etc.
            camera_idx = int(selected.split()[-1])  # Extract the number
            camera_src = camera_idx
        except (ValueError, IndexError):
            camera_src = 0  # Fallback to default camera
    
        # Update camera source
        self.cam_svc.src = camera_src
    
        # Create database folder if it doesn't exist
        if not os.path.exists(self.db_path):
            try:
                os.makedirs(self.db_path)
                self._update_info_text(f"✓ Created database folder: {self.db_path}\n")
            except Exception as e:
                self._update_info_text(f"✗ Error creating database folder: {str(e)}\n")
                return
    
        try:
            if not self.cam_svc.start():
                self._update_info_text("✗ ERROR: Could not open camera!\n")
                self.cam_svc.stop()
                return
        except Exception as e:
            self._update_info_text(f"✗ ERROR: Camera initialization failed: {str(e)}\n")
            self.cam_svc.stop()
            return
    
        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.capture_btn.configure(state="normal")
        self.camera_combo.configure(state="disabled")  # Disable camera selection while running
    
        self._update_info_text(f"✓ Camera started: {self.camera_combo.get()}\n✓ Database path: {self.db_path}\n")
    
        threading.Thread(target=self.update_frame, daemon=True).start()
    
    # Update the stop_camera method for AddFaceDatabaseTab:
    def stop_camera(self):  # For AddFaceDatabaseTab
        self.is_running = False
        self.cam_svc.stop()
    
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.capture_btn.configure(state="disabled")
        self.camera_combo.configure(state="readonly")  # Re-enable camera selection
    
        self.video_label.configure(image=None, text="Camera stopped\nClick 'Start Camera' to begin again")
        self._update_info_text("✓ Camera stopped\n")
    
    def capture_photo(self):
        if not self.is_running or self.current_frame is None:
            tkmsg.showwarning("No Frame", "No camera frame available to capture!")
            return
        
        # Validate that a face is detected
        try:
            faces = DeepFace.extract_faces(
                self.current_frame,
                detector_backend="retinaface",
                enforce_detection=True,  # Require face detection
                align=False
            )
            
            if not faces or len(faces) == 0:
                tkmsg.showwarning("No Face Detected", 
                    "No face detected in the current frame!\n\n"
                    "Please position your face clearly in the camera view.")
                return
            
            if len(faces) > 1:
                response = tkmsg.askyesno("Multiple Faces", 
                    f"Detected {len(faces)} faces in the frame.\n\n"
                    "Do you want to continue anyway?")
                if not response:
                    return
            
            # Store the captured frame
            self.captured_frame = self.current_frame.copy()
            
            # Show the captured photo in the preview
            self._display_captured_photo(self.captured_frame)
            
            # Queue info update
            self.ui_update_queue.put(('info', f"✓ Face detected! Ready to save.\n"))
            
            # Open dialog to get person's name
            self._show_name_dialog()
            
        except Exception as e:
            tkmsg.showerror("Detection Error", 
                f"Failed to detect face:\n{str(e)}\n\n"
                "Please ensure your face is clearly visible and try again.")
            return
    
    # ----------  video loop  ----------
    def update_frame(self):
        try:
            while self.is_running:
                ret, frame = self.cam_svc.read()
                if not ret:
                    break
                
                self.current_frame = frame.copy()
                display_frame = frame.copy()
                
                # Detect face and draw box every N frames (performance optimization)
                if self.frame_count % self.process_every_n_frames == 0:
                    try:
                        faces = DeepFace.extract_faces(
                            frame,
                            detector_backend="opencv",
                            enforce_detection=False,
                            align=False
                        )
                        
                        self.last_face_count = len(faces)
                        
                        for face in faces:
                            region = face.get('facial_area', {})
                            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
                            if w > 0 and h > 0:
                                # Draw rectangle
                                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (46, 204, 113), 3)
                                
                                # Draw label background
                                label = "Face Detected"
                                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(display_frame, (x, y - lh - 10), (x + lw + 6, y), (46, 204, 113), -1)
                                cv2.putText(display_frame, label, (x + 3, y - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    except Exception as e:
                        self.last_face_count = 0
                        # If detection fails, just show the frame without boxes
                
                # Draw face count overlay
                self._draw_overlay(display_frame)
                
                # Convert and display
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.thumbnail((640, 480), Image.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=img, size=img.size)
                
                # Queue video update
                self.ui_update_queue.put(('video', ctk_img))
                
                self.frame_count += 1
        
        except Exception as e:
            print(f"Frame update error: {e}")
        finally:
            self.cam_svc.stop()
    
    # ----------  name dialog  ----------
    def _show_name_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Enter Person Name")
        dialog.geometry("400x200")
        dialog.transient(self)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (dialog.winfo_screenheight() // 2) - (200 // 2)
        dialog.geometry(f"400x200+{x}+{y}")
        
        # Content
        ctk.CTkLabel(dialog, text="Enter the person's name:", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=20)
        
        name_entry = ctk.CTkEntry(dialog, width=300, placeholder_text="e.g., Sumarti Joko")
        name_entry.pack(pady=10)
        name_entry.focus()
        
        result_label = ctk.CTkLabel(dialog, text="", text_color="red")
        result_label.pack(pady=5)
        
        def save_photo():
            name = name_entry.get().strip()
            
            if not name:
                result_label.configure(text="⚠ Name cannot be empty!")
                return
            
            # Sanitize name for folder name (remove special characters)
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
            if not safe_name:
                result_label.configure(text="⚠ Invalid name!")
                return
            
            # Create person's folder
            person_folder = os.path.join(self.db_path, safe_name)
            
            # Security: Validate path to prevent path traversal attacks
            person_folder_abs = os.path.abspath(person_folder)
            db_path_abs = os.path.abspath(self.db_path)
            
            if not person_folder_abs.startswith(db_path_abs + os.sep):
                result_label.configure(text="⚠ Invalid path detected!")
                self._update_info_text(f"✗ Security: Invalid path rejected: {safe_name}\n")
                return
            
            try:
                if not os.path.exists(person_folder):
                    os.makedirs(person_folder)
                    self._update_info_text(f"✓ Created new folder for: {safe_name}\n")
                else:
                    self._update_info_text(f"✓ Adding to existing folder: {safe_name}\n")
                
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{safe_name}_{timestamp}.jpg"
                filepath = os.path.join(person_folder, filename)
                
                # Save the photo
                cv2.imwrite(filepath, self.captured_frame)
                
                self._update_info_text(f"✓ Photo saved successfully!\n   Path: {filepath}\n")
                tkmsg.showinfo("Success", f"Photo saved successfully!\n\nName: {safe_name}\nFile: {filename}")
                
                dialog.destroy()
                
            except Exception as e:
                result_label.configure(text=f"✗ Error saving photo: {str(e)}")
                self._update_info_text(f"✗ Error saving photo: {str(e)}\n")
        
        def on_enter(event):
            save_photo()
        
        name_entry.bind('<Return>', on_enter)
        
        # Buttons
        btn_frame = ctk.CTkFrame(dialog)
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(btn_frame, text="Save", command=save_photo, 
                     fg_color="green", hover_color="darkgreen", width=120).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", command=dialog.destroy,
                     fg_color="red", hover_color="darkred", width=120).pack(side="left", padx=5)
    
    # ----------  UI helpers  ----------
    def _draw_overlay(self, frame):
        """Draw status overlay on frame"""
        # Draw semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw face count
        status_text = f"Faces Detected: {self.last_face_count}"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (46, 204, 113) if self.last_face_count > 0 else (255, 255, 255), 2)
        
        # Draw ready indicator
        if self.last_face_count == 1:
            cv2.putText(frame, "Ready to Capture", (frame.shape[1] - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (46, 204, 113), 2)
        elif self.last_face_count > 1:
            cv2.putText(frame, "Multiple Faces!", (frame.shape[1] - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 107, 53), 2)
    
    def _display_captured_photo(self, frame):
        """Display the captured photo in the preview panel"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((640, 480), Image.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=img, size=img.size)
        
        self.captured_label.configure(image=ctk_img, text="")
        self.captured_label.ctk_img = ctk_img
    
    def _update_info_text(self, message):
        """Update the info text box with new message (thread-safe)"""
        # Queue the update instead of direct modification
        self.ui_update_queue.put(('info', message))
    
    def _process_ui_queue(self):
        """Process UI updates from background thread (thread-safe)"""
        try:
            while not self.ui_update_queue.empty():
                update_type, data = self.ui_update_queue.get_nowait()
                if update_type == 'video':
                    self.video_label.configure(image=data, text="")
                    self.video_label.ctk_img = data
                elif update_type == 'info':
                    # Thread-safe info text update
                    self.info_text.configure(state="normal")
                    self.info_text.insert("end", data)
                    self.info_text.see("end")
                    self.info_text.configure(state="disabled")
        except:
            pass
        finally:
            # Schedule next check
            self.after(50, self._process_ui_queue)
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        self.cam_svc.stop()
    
    def __del__(self):
        try:
            self.cleanup()
        except:
            pass


# ----------  Updated App class to include 4th tab  ----------
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Photo Analysis Suite – DeepFace & Digital Forensics")
        self.geometry("1000x700")  # Slightly larger for better layout
        
        # Set up cleanup handler
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=1, padx=10, pady=10)

        # Add all tabs
        self.tabview.add("Face Verification")
        self.tabview.add("Digital Forensics")
        self.tabview.add("Live Recognition")
        self.tabview.add("Add to Database")  # NEW 4th tab
        
        # Initialize tabs and store references for cleanup
        self.face_verification_tab = FaceVerificationTab(self.tabview.tab("Face Verification"))
        self.face_verification_tab.pack(fill="both", expand=1)
        
        self.digital_forensics_tab = DigitalForensicsTab(self.tabview.tab("Digital Forensics"))
        self.digital_forensics_tab.pack(fill="both", expand=1)
        
        self.live_recognition_tab = LiveRecognitionTab(self.tabview.tab("Live Recognition"))
        self.live_recognition_tab.pack(fill="both", expand=1)
        
        self.add_face_tab = AddFaceDatabaseTab(self.tabview.tab("Add to Database"))
        self.add_face_tab.pack(fill="both", expand=1)
    
    def on_closing(self):
        """Cleanup resources before closing"""
        # Cleanup tabs that use camera resources
        for tab in [self.live_recognition_tab, self.add_face_tab]:
            if hasattr(tab, 'cleanup'):
                try:
                    tab.cleanup()
                except Exception as e:
                    print(f"Cleanup error: {e}")
        self.destroy()


if __name__ == "__main__":
    # ----------  WARM-UP MODELS (skip missing ones)  ----------
    print("Warming up models …")
    optional = {"Dlib"}                      # add more here if you like
    for m in ["VGG-Face", "Facenet", "ArcFace", "OpenFace", "DeepID", "Dlib", "SFace"]:
        try:
            _ = get_model(m)
            print(f"  ✓ {m}")
        except ImportError as e:
            if m in optional:
                print(f"  ⚠ {m} skipped – {e}")
            else:
                raise   # really required model failed
    print("All available models ready.")
    App().mainloop()
 
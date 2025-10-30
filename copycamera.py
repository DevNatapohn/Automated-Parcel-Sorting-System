import cv2
import numpy as np
import os
from datetime import datetime
from typing import Optional, Tuple
import threading
import time


class ParcelCamera:
    """Real-time camera system for capturing parcel images"""
    
    def __init__(self, output_folder: str = "parcel_images", 
                 camera_index: int = 0,
                 auto_capture: bool = False,
                 min_contour_area: int = 50000):
        """
        Args:
            output_folder: โฟลเดอร์สำหรับบันทึกภาพ
            camera_index: index ของกล้อง (0 = กล้องหลัก)
            auto_capture: ถ่ายภาพอัตโนมัติเมื่อตรวจจับวัตถุ
            min_contour_area: พื้นที่ขั้นต่ำสำหรับตรวจจับพัสดุ
        """
        self.output_folder = output_folder
        self.camera_index = camera_index
        self.auto_capture = auto_capture
        self.min_contour_area = min_contour_area
        
        # สร้างโฟลเดอร์ถ้ายังไม่มี
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"✅ สร้างโฟลเดอร์: {output_folder}")
        
        # Camera settings
        self.cap = None
        self.is_running = False
        self.captured_images = []
        
        # Detection settings
        self.detection_cooldown = 2.0  # วินาที ก่อนจับภาพถัดไป
        self.last_capture_time = 0
        
        # ROI (Region of Interest) settings
        self.roi_percentage = 0.7  # 70% ของหน้าจอ
        
        # Background subtraction
        self.background = None
        self.bg_frames_count = 0
        self.bg_frames_needed = 30  # จำนวนเฟรมสำหรับสร้าง background
        
    def initialize_camera(self) -> bool:
        """เปิดกล้อง"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"❌ ไม่สามารถเปิดกล้อง index {self.camera_index}")
                return False
            
            # ตั้งค่าความละเอียด
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # เพิ่มการตั้งค่ากล้องสำหรับความชัดที่ดีขึ้น
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Auto focus
            self.cap.set(cv2.CAP_PROP_FOCUS, 0)  # Focus setting
            
            # อ่านขนาดจริง
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✅ เปิดกล้องสำเร็จ - ความละเอียด: {self.frame_width}x{self.frame_height}")
            self.is_running = True
            return True
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการเปิดกล้อง: {e}")
            return False
    
    def get_roi_bounds(self, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """คำนวณพื้นที่ ROI (กรอบสแกน)"""
        h, w = frame_shape[:2]
        
        roi_w = int(w * self.roi_percentage)
        roi_h = int(h * self.roi_percentage)
        
        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h
        
        return x1, y1, x2, y2
    
    def draw_roi_frame(self, frame: np.ndarray, detected: bool = False) -> np.ndarray:
        """วาดกรอบ ROI และ UI"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.get_roi_bounds(frame.shape)
        
        # สีกรอบ (เขียว = ตรวจจับ, ฟ้า = ปกติ)
        color = (0, 255, 0) if detected else (255, 200, 0)
        thickness = 3 if detected else 2
        
        # วาดกรอบหลัก
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # วาดมุมเน้น (corner markers)
        corner_length = 40
        corner_thickness = 5
        
        corners = [
            (x1, y1), (x2, y1),  # บนซ้าย, บนขวา
            (x1, y2), (x2, y2)   # ล่างซ้าย, ล่างขวา
        ]
        
        for i, (cx, cy) in enumerate(corners):
            if i == 0:  # บนซ้าย
                cv2.line(frame, (cx, cy), (cx + corner_length, cy), color, corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy + corner_length), color, corner_thickness)
            elif i == 1:  # บนขวา
                cv2.line(frame, (cx, cy), (cx - corner_length, cy), color, corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy + corner_length), color, corner_thickness)
            elif i == 2:  # ล่างซ้าย
                cv2.line(frame, (cx, cy), (cx + corner_length, cy), color, corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy - corner_length), color, corner_thickness)
            elif i == 3:  # ล่างขวา
                cv2.line(frame, (cx, cy), (cx - corner_length, cy), color, corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy - corner_length), color, corner_thickness)
        
        # เส้นกึ่งกลาง (crosshair)
        center_x, center_y = w // 2, h // 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
        
        # Text overlay - Header
        header_bg = np.zeros((80, w, 3), dtype=np.uint8)
        header_bg[:] = (40, 40, 40)
        frame[0:80] = cv2.addWeighted(frame[0:80], 0.3, header_bg, 0.7, 0)
        
        cv2.putText(frame, "PARCEL SCANNER", (20, 35), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Images: {len(self.captured_images)}", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Status text
        if detected:
            status_text = "PARCEL DETECTED!"
            cv2.putText(frame, status_text, (w - 350, 35), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
        
        # Instructions - Footer
        footer_y = h - 100
        instructions = [
            "SPACE: Capture",
            "C: Clear All",
            "A: Auto Mode",
            "Q/ESC: Exit"
        ]
        
        footer_bg = np.zeros((100, w, 3), dtype=np.uint8)
        footer_bg[:] = (40, 40, 40)
        frame[footer_y:h] = cv2.addWeighted(frame[footer_y:h], 0.3, footer_bg, 0.7, 0)
        
        for i, instruction in enumerate(instructions):
            x_pos = 20 + (i * 200)
            cv2.putText(frame, instruction, (x_pos, footer_y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        mode_text = f"Mode: {'AUTO' if self.auto_capture else 'MANUAL'}"
        cv2.putText(frame, mode_text, (w - 200, footer_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        return frame
    
    def detect_parcel(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """ตรวจจับพัสดุในกรอบ ROI"""
        x1, y1, x2, y2 = self.get_roi_bounds(frame.shape)
        roi = frame[y1:y2, x1:x2]
        
        # แปลงเป็น grayscale เฉพาะสำหรับการตรวจจับ
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # ลด noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # หา contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # หา contour ที่ใหญ่ที่สุด
        detected = False
        largest_contour = None
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > self.min_contour_area:
                detected = True
        
        return detected, roi
    
    def capture_image(self, frame: np.ndarray, auto: bool = False) -> Optional[str]:
        """บันทึกภาพพัสดุ"""
        # ตรวจสอบ cooldown (สำหรับ auto capture)
        if auto:
            current_time = time.time()
            if current_time - self.last_capture_time < self.detection_cooldown:
                return None
            self.last_capture_time = current_time
        
        # สร้างชื่อไฟล์
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"parcel_{timestamp}.jpg"
        filepath = os.path.join(self.output_folder, filename)
        
        # บันทึกภาพ
        try:
            # Crop ROI
            x1, y1, x2, y2 = self.get_roi_bounds(frame.shape)
            roi = frame[y1:y2, x1:x2]
            
            # บันทึกด้วยคุณภาพสูง (บันทึกภาพ roi ต้นฉบับ)
            cv2.imwrite(filepath, roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            self.captured_images.append({
                'filename': filename,
                'filepath': filepath,
                'timestamp': timestamp,
                'auto': auto
            })
            
            print(f"📸 {'[AUTO]' if auto else '[MANUAL]'} บันทึกภาพ: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ ไม่สามารถบันทึกภาพ: {e}")
            return None
    
    def clear_captured_images(self):
        """ลบภาพที่ถ่ายทั้งหมด"""
        try:
            for img_data in self.captured_images:
                filepath = img_data['filepath']
                if os.path.exists(filepath):
                    os.remove(filepath)
            
            count = len(self.captured_images)
            self.captured_images.clear()
            print(f"🗑️  ลบภาพทั้งหมด ({count} ไฟล์)")
            
        except Exception as e:
            print(f"⚠️ ข้อผิดพลาดในการลบภาพ: {e}")
    
    def run(self):
        """เริ่มการทำงานของกล้อง"""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*60)
        print("📷 เปิดกล้องสแกนพัสดุ")
        print("="*60)
        print("⌨️  Controls:")
        print("   SPACE       - ถ่ายภาพ")
        print("   C           - ลบภาพทั้งหมด")
        print("   A           - สลับโหมด Auto/Manual")
        print("   Q / ESC     - ออกจากโปรแกรม")
        print("="*60)
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ ไม่สามารถอ่านภาพจากกล้อง")
                    break
                
                # ตรวจจับพัสดุ
                detected, roi = self.detect_parcel(frame)
                
                # Auto capture
                if self.auto_capture and detected:
                    self.capture_image(frame, auto=True)
                
                # วาด UI
                display_frame = self.draw_roi_frame(frame.copy(), detected)
                
                # แสดงผล
                cv2.imshow('Parcel Scanner', display_frame)
                
                # รับคำสั่งจากคีย์บอร์ด
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space - ถ่ายภาพ
                    self.capture_image(frame, auto=False)
                
                elif key == ord('c') or key == ord('C'):  # C - ลบทั้งหมด
                    self.clear_captured_images()
                
                elif key == ord('a') or key == ord('A'):  # A - สลับโหมด
                    self.auto_capture = not self.auto_capture
                    mode = "AUTO" if self.auto_capture else "MANUAL"
                    print(f"🔄 สลับโหมดเป็น: {mode}")
                
                elif key == ord('q') or key == ord('Q') or key == 27:  # Q/ESC - ออก
                    print("\n👋 กำลังปิดกล้อง...")
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️ หยุดการทำงานโดยผู้ใช้")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """ปิดกล้องและทำความสะอาด"""
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        print(f"\n✅ ปิดกล้องเรียบร้อย")
        print(f"📸 ภาพที่ถ่ายทั้งหมด: {len(self.captured_images)} ภาพ")
        
        if self.captured_images:
            print(f"📁 บันทึกไว้ที่: {self.output_folder}/")
    
    def get_captured_images(self):
        """ดึงรายการภาพที่ถ่าย"""
        return self.captured_images.copy()


def main():
    """ทดสอบการทำงานของกล้อง"""
    camera = ParcelCamera(
        output_folder="parcel_images",
        camera_index=0,
        auto_capture=False,  # เริ่มด้วยโหมด manual
        min_contour_area=50000
    )
    
    camera.run()
    
    # แสดงภาพที่ถ่าย
    captured = camera.get_captured_images()
    if captured:
        print("\n📋 รายการภาพที่ถ่าย:")
        for i, img in enumerate(captured, 1):
            print(f"   {i}. {img['filename']} ({'AUTO' if img['auto'] else 'MANUAL'})")


if __name__ == "__main__":
    main()
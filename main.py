#!/usr/bin/env python3
"""
Complete Pipeline Controller
Camera → OCR Processing → Dobot Sorting → Database
"""

import os
import time
import threading
from queue import Queue
from datetime import datetime
from dotenv import load_dotenv

# Import modules
from camera import ParcelCamera
from process import CompleteParcelSortingSystem
from dobot_controller import DobotController


class CompleteSortingPipeline:
    """ระบบ Pipeline แบบครบวงจร: Camera → OCR → Dobot → Database"""
    
    def __init__(self, config: dict = None):
        # โหลดค่า configuration
        load_dotenv()
        
        self.config = config or {}
        
        # --- API Keys ---
        self.typhoon_api_key = self.config.get('typhoon_api_key') or os.getenv("TYPHOON_API_KEY")
        self.db_api_url = self.config.get('db_api_url') or os.getenv("DB_API_URL")
        self.db_api_key = self.config.get('db_api_key') or os.getenv("DB_API_KEY")
        
        # --- Folders ---
        self.image_folder = self.config.get('image_folder', "parcel_images")
        self.output_folder = self.config.get('output_folder', "parcel_results")
        
        # --- Camera Settings ---
        self.camera_index = self.config.get('camera_index', 0)
        self.auto_capture = self.config.get('auto_capture', False)
        
        # --- OCR Settings ---
        self.enhance_images = self.config.get('enhance_images', True)
        self.save_to_db = self.config.get('save_to_db', False)
        
        # --- Dobot Settings ---
        self.enable_dobot = self.config.get('enable_dobot', True)
        self.dobot_port = self.config.get('dobot_port', 'COM5')
        self.dobot_speed = self.config.get('dobot_speed', 100)
        self.dobot_simulation = self.config.get('dobot_simulation', False)
        
        # --- Processing Settings ---
        self.auto_process = self.config.get('auto_process', True)
        self.process_interval = self.config.get('process_interval', 2)
        
        # --- Queues ---
        self.ocr_queue = Queue()      # Queue สำหรับ OCR
        self.dobot_queue = Queue()    # Queue สำหรับ Dobot
        
        # --- Components ---
        self.camera = None
        self.ocr_processor = None
        self.dobot_controller = None
        
        # --- Threading ---
        self.ocr_thread = None
        self.dobot_thread = None
        self.monitor_thread = None
        self.is_running = False
        
        # --- Statistics ---
        self.stats = {
            'images_captured': 0,
            'ocr_processed': 0,
            'ocr_success': 0,
            'ocr_failed': 0,
            'dobot_sorted': 0,
            'dobot_failed': 0,
            'db_saved': 0,
            'db_failed': 0
        }
    
    def validate_configuration(self) -> bool:
        """ตรวจสอบการตั้งค่า"""
        print("\n🔍 กำลังตรวจสอบการตั้งค่า...")
        print("="*60)
        
        errors = []
        
        # ตรวจสอบ Typhoon API
        if not self.typhoon_api_key:
            errors.append("❌ ไม่พบ TYPHOON_API_KEY")
        else:
            print(f"✅ Typhoon API: {'*' * 20}{self.typhoon_api_key[-4:]}")
        
        # ตรวจสอบ Database (ถ้าเปิดใช้)
        if self.save_to_db:
            if not self.db_api_url:
                errors.append("⚠️  ไม่พบ DB_API_URL (ข้าม Database)")
                self.save_to_db = False
            else:
                print(f"✅ Database: {self.db_api_url}")
        
        # ตรวจสอบ Dobot (ถ้าเปิดใช้)
        if self.enable_dobot:
            print(f"✅ Dobot: Port {self.dobot_port}, Speed {self.dobot_speed}%")
            if self.dobot_simulation:
                print("   ⚠️  โหมด: SIMULATION (ไม่ต่อ Dobot จริง)")
        else:
            print("⚠️  Dobot: ปิดใช้งาน")
        
        # ตรวจสอบโฟลเดอร์
        for folder in [self.image_folder, self.output_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"✅ สร้างโฟลเดอร์: {folder}/")
            else:
                print(f"✅ โฟลเดอร์: {folder}/")
        
        print("="*60)
        
        if errors:
            print("\n⚠️  พบข้อผิดพลาด:")
            for error in errors:
                print(f"   {error}")
            return False
        
        print("✅ การตั้งค่าถูกต้อง")
        return True
    
    def initialize_components(self):
        """เตรียมส่วนประกอบทั้งหมด"""
        print("\n🚀 กำลังเตรียมระบบ...")
        print("="*60)
        
        # 1. กล้อง
        print("📷 [1/3] เตรียมกล้อง...")
        self.camera = ParcelCamera(
            output_folder=self.image_folder,
            camera_index=self.camera_index,
            auto_capture=self.auto_capture
        )
        
        # 2. OCR Processor
        print("🔍 [2/3] เตรียมระบบ OCR...")
        self.ocr_processor = CompleteParcelSortingSystem(
            typhoon_api_key=self.typhoon_api_key,
            output_folder=self.output_folder,
            db_api_url=self.db_api_url if self.save_to_db else None,
            db_api_key=self.db_api_key if self.save_to_db else None
        )
        
        # 3. Dobot Controller
        if self.enable_dobot:
            print("🤖 [3/3] เตรียม Dobot Controller...")
            self.dobot_controller = DobotController(
                port=self.dobot_port,
                speed=self.dobot_speed,
                simulation_mode=self.dobot_simulation
            )
            
            # เชื่อมต่อ Dobot
            success = self.dobot_controller.connect()
            if not success and not self.dobot_simulation:
                print("   ⚠️  เปลี่ยนเป็น Simulation Mode")
        else:
            print("⚠️  [3/3] ข้าม Dobot Controller")
        
        print("="*60)
        print("✅ เตรียมระบบเสร็จสมบูรณ์")
    
    def ocr_worker(self):
        """Worker Thread: ประมวลผล OCR"""
        print("🔄 เริ่ม OCR Worker Thread")
        
        while self.is_running:
            try:
                if not self.ocr_queue.empty():
                    image_path = self.ocr_queue.get()
                    
                    print(f"\n{'='*60}")
                    print(f"🔍 OCR: {os.path.basename(image_path)}")
                    print(f"{'='*60}")
                    
                    # ประมวลผล OCR
                    result = self.ocr_processor.process_single_parcel(
                        image_path=image_path,
                        enhance_image=self.enhance_images,
                        save_to_db=self.save_to_db
                    )
                    
                    # อัพเดทสถิติ
                    self.stats['ocr_processed'] += 1
                    
                    if result['success']:
                        self.stats['ocr_success'] += 1
                        
                        province = result.get('province')
                        tracking = result.get('tracking_number')
                        
                        print(f"\n✅ OCR สำเร็จ!")
                        print(f"   📦 Tracking: {tracking}")
                        print(f"   📍 ปลายทาง: {province}")
                        
                        if result.get('db_saved'):
                            self.stats['db_saved'] += 1
                            print(f"   💾 Database ID: {result['db_parcel_id']}")
                        
                        # ส่งต่อไปยัง Dobot Queue
                        if self.enable_dobot and province and province != 'ไม่ระบุ':
                            self.dobot_queue.put({
                                'image_path': image_path,
                                'province': province,
                                'tracking': tracking,
                                'result': result
                            })
                            print(f"   🤖 ส่งต่อไปยัง Dobot Queue")
                        else:
                            print(f"   ⚠️  ข้าม Dobot (ไม่ระบุจังหวัด หรือปิดใช้)")
                    
                    else:
                        self.stats['ocr_failed'] += 1
                        print(f"\n❌ OCR ล้มเหลว: {result.get('error')}")
                    
                    self.ocr_queue.task_done()
                    self.print_stats()
                
                else:
                    time.sleep(0.5)
            
            except Exception as e:
                print(f"❌ ข้อผิดพลาดใน OCR Worker: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
        
        print("🛑 หยุด OCR Worker Thread")
    
    def dobot_worker(self):
        """Worker Thread: ควบคุม Dobot"""
        print("🔄 เริ่ม Dobot Worker Thread")
        
        while self.is_running:
            try:
                if not self.dobot_queue.empty():
                    parcel_data = self.dobot_queue.get()
                    
                    province = parcel_data['province']
                    tracking = parcel_data['tracking']
                    
                    print(f"\n{'='*60}")
                    print(f"🤖 Dobot: คัดแยกพัสดุ {tracking} → {province}")
                    print(f"{'='*60}")
                    
                    # สั่งงาน Dobot
                    success = self.dobot_controller.pick_and_place(province)
                    
                    if success:
                        self.stats['dobot_sorted'] += 1
                        print(f"✅ Dobot คัดแยกสำเร็จ")
                        
                        # ย้ายภาพไป processed
                        self.move_to_processed(parcel_data['image_path'])
                    else:
                        self.stats['dobot_failed'] += 1
                        print(f"❌ Dobot คัดแยกล้มเหลว")
                    
                    # กลับ Home
                    self.dobot_controller.move_home("รอพัสดุชิ้นต่อไป")
                    
                    self.dobot_queue.task_done()
                    self.print_stats()
                
                else:
                    time.sleep(0.5)
            
            except Exception as e:
                print(f"❌ ข้อผิดพลาดใน Dobot Worker: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
        
        print("🛑 หยุด Dobot Worker Thread")
    
    def monitor_new_images(self):
        """Monitor Thread: ตรวจจับภาพใหม่"""
        print("👁️  เริ่ม Image Monitor Thread")
        
        processed_files = set()
        
        while self.is_running:
            try:
                current_files = set()
                
                if os.path.exists(self.image_folder):
                    for filename in os.listdir(self.image_folder):
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            current_files.add(filename)
                
                new_files = current_files - processed_files
                
                if new_files:
                    for filename in sorted(new_files):
                        filepath = os.path.join(self.image_folder, filename)
                        
                        if os.path.exists(filepath):
                            time.sleep(0.5)  # รอให้ไฟล์เขียนเสร็จ
                            
                            self.ocr_queue.put(filepath)
                            processed_files.add(filename)
                            self.stats['images_captured'] += 1
                            
                            print(f"\n📸 ตรวจพบภาพใหม่: {filename}")
                            print(f"   📋 OCR Queue: {self.ocr_queue.qsize()}")
                            print(f"   🤖 Dobot Queue: {self.dobot_queue.qsize()}")
                
                time.sleep(self.process_interval)
            
            except Exception as e:
                print(f"❌ ข้อผิดพลาดใน Monitor Thread: {e}")
                time.sleep(1)
        
        print("🛑 หยุด Image Monitor Thread")
    
    def move_to_processed(self, image_path: str):
        """ย้ายภาพที่ประมวลผลแล้วไปโฟลเดอร์ processed"""
        try:
            processed_folder = os.path.join(self.image_folder, "processed")
            if not os.path.exists(processed_folder):
                os.makedirs(processed_folder)
            
            filename = os.path.basename(image_path)
            dest_path = os.path.join(processed_folder, filename)
            
            if os.path.exists(image_path):
                os.rename(image_path, dest_path)
                print(f"   📁 ย้ายภาพไป: processed/{filename}")
        
        except Exception as e:
            print(f"   ⚠️ ไม่สามารถย้ายภาพ: {e}")
    
    def print_stats(self):
        """แสดงสถิติ"""
        print(f"\n{'='*60}")
        print(f"📊 สถิติการทำงาน")
        print(f"{'='*60}")
        print(f"📸 ภาพที่ถ่าย:       {self.stats['images_captured']}")
        print(f"🔍 OCR ประมวลผล:     {self.stats['ocr_processed']}")
        print(f"   ✅ สำเร็จ:         {self.stats['ocr_success']}")
        print(f"   ❌ ล้มเหลว:       {self.stats['ocr_failed']}")
        
        if self.enable_dobot:
            print(f"🤖 Dobot คัดแยก:     {self.stats['dobot_sorted']}")
            print(f"   ❌ ล้มเหลว:       {self.stats['dobot_failed']}")
        
        if self.save_to_db:
            print(f"💾 Database:         {self.stats['db_saved']} (ล้มเหลว: {self.stats['db_failed']})")
        
        print(f"📋 รอประมวลผล:")
        print(f"   OCR Queue:        {self.ocr_queue.qsize()}")
        if self.enable_dobot:
            print(f"   Dobot Queue:      {self.dobot_queue.qsize()}")
        print(f"{'='*60}")
    
    def run(self):
        """เริ่มการทำงานของระบบทั้งหมด"""
        print("\n" + "="*60)
        print("🎯 COMPLETE PARCEL SORTING SYSTEM")
        print("   Camera → OCR → Dobot → Database")
        print("="*60)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # ตรวจสอบการตั้งค่า
        if not self.validate_configuration():
            print("\n❌ ไม่สามารถเริ่มระบบได้")
            return
        
        # เตรียมส่วนประกอบ
        self.initialize_components()
        
        # เริ่ม threads
        self.is_running = True
        
        if self.auto_process:
            # OCR Worker
            self.ocr_thread = threading.Thread(
                target=self.ocr_worker,
                daemon=True
            )
            self.ocr_thread.start()
            
            # Dobot Worker (ถ้าเปิดใช้)
            if self.enable_dobot:
                self.dobot_thread = threading.Thread(
                    target=self.dobot_worker,
                    daemon=True
                )
                self.dobot_thread.start()
            
            # Image Monitor
            self.monitor_thread = threading.Thread(
                target=self.monitor_new_images,
                daemon=True
            )
            self.monitor_thread.start()
            
            print("\n✅ เริ่มระบบประมวลผลอัตโนมัติ")
        
        # เริ่มกล้อง (blocking)
        print("\n📷 เปิดกล้อง...")
        try:
            self.camera.run()
        except KeyboardInterrupt:
            print("\n⚠️  หยุดโดยผู้ใช้")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """ปิดระบบ"""
        print("\n🛑 กำลังปิดระบบ...")
        
        self.is_running = False
        
        # รอ queue ว่าง
        if not self.ocr_queue.empty():
            print(f"⏳ รอ OCR ({self.ocr_queue.qsize()} ภาพ)...")
            self.ocr_queue.join()
        
        if self.enable_dobot and not self.dobot_queue.empty():
            print(f"⏳ รอ Dobot ({self.dobot_queue.qsize()} ชิ้น)...")
            self.dobot_queue.join()
        
        # รอ threads
        for thread in [self.ocr_thread, self.dobot_thread, self.monitor_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        # ปิด Dobot
        if self.dobot_controller:
            self.dobot_controller.disconnect()
        
        # สรุปผล
        print("\n" + "="*60)
        print("📊 สรุปผลการทำงาน")
        print("="*60)
        self.print_stats()
        
        # แสดงสถิติ OCR
        if self.ocr_processor:
            print("\n" + self.ocr_processor.generate_sorting_report([]))
        
        # แสดงสถิติ Dobot
        if self.dobot_controller:
            self.dobot_controller.print_stats()
        
        print("\n✅ ปิดระบบเรียบร้อย")
        print(f"📁 ผลลัพธ์:")
        print(f"   - ภาพ: {self.image_folder}/")
        print(f"   - ผลลัพธ์: {self.output_folder}/")


def main():
    """ฟังก์ชันหลัก"""
    
    config = {
        # API Settings
        'typhoon_api_key': None,  # อ่านจาก .env
        'db_api_url': None,       # อ่านจาก .env
        'db_api_key': None,       # อ่านจาก .env
        
        # Folder Settings
        'image_folder': 'parcel_images',
        'output_folder': 'parcel_results',
        
        # Camera Settings
        'camera_index': 1,
        'auto_capture': False,
        
        # OCR Settings
        'enhance_images': True,
        'save_to_db': True,  # เปลี่ยนเป็น True ถ้าต้องการบันทึก DB
        
        # Dobot Settings
        'enable_dobot': True,           # เปลี่ยนเป็น False ถ้าไม่มี Dobot
        'dobot_port': 'COM3',           # แก้ไข COM Port ของ Dobot
        'dobot_speed': 100,             # ความเร็ว 0-100
        'dobot_simulation': False,      # True = ไม่เชื่อมต่อ Dobot จริง
        
        # Processing Settings
        'auto_process': True,
        'process_interval': 2,
    }
    
    # สร้างและรัน pipeline
    pipeline = CompleteSortingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
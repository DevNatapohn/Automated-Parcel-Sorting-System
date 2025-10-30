#!/usr/bin/env python3
"""
Dobot Controller - ควบคุมแขนกล Dobot สำหรับคัดแยกพัสดุ
**แก้ไข:** เน้นการใช้ wait=True เพื่อให้แน่ใจว่า Dobot เคลื่อนที่ไปถึงตำแหน่งก่อนเริ่มคำสั่งถัดไป ลดปัญหาตำแหน่งคลาดเคลื่อน (Drift)
"""

from pydobot import Dobot
import time
from typing import Optional, Dict, Tuple

class DobotController:
    """Class สำหรับควบคุมแขนกล Dobot"""

    # --- จุดพิกัดต่างๆ ---
    HOME = [125.92, 177.82, 42.11, 54.70]
    PICKUP = [-8.51, 215.85, -8.19,92.26]  # จุดหยิบพัสดุ (หลังสแกน OCR)
    
    # จุดวางตามจังหวัด (คุณระบุให้ใช้พิกัดซ้ำกันทั้งหมด)
    DROP_POINTS = {
        "นครนายก": [231.49, 0.25, -18.38, 0.06],
        "นครสวรรค์": [232.60, 140.08, -19.86, 31.06],
        "เชียงใหม่": [231.49, 0.25, -18.38, 0.06],
        "สระบุรี": [232.60, 140.08, -19.86, 31.06],
    }

    # ความสูงปลอดภัย
    SAFETY_Z = 50.0
    
    # ความเร็ว
    FAST_SPEED = 100
    SLOW_SPEED = 50
    
    # ดีเลย์หัวดูด
    SUCTION_DELAY = 0.05
    
    def __init__(self, port: str = "COM5", speed: int = 100, simulation_mode: bool = False):
        """
        Args:
            port: COM port ของ Dobot
            speed: ความเร็วการเคลื่อนที่ (0-100)
            simulation_mode: ถ้า True จะไม่เชื่อมต่อ Dobot จริง (ทดสอบ)
        """
        self.port = port
        self.speed = speed
        self.simulation_mode = simulation_mode
        self.dobot: Optional[Dobot] = None
        self.is_connected = False
        
        # สถิติ
        self.stats = {
            'total_picks': 0,
            'successful_drops': 0,
            'failed_drops': 0,
            'by_province': {}
        }
    
    def connect(self) -> bool:
        """เชื่อมต่อกับ Dobot"""
        if self.simulation_mode:
            print("🤖 [SIMULATION MODE] ไม่ได้เชื่อมต่อ Dobot จริง")
            self.is_connected = True
            return True
        
        try:
            print(f"🔌 กำลังเชื่อมต่อ Dobot ที่ {self.port}...")
            # การเชื่อมต่อ Dobot อาจใช้เวลานาน หากไม่สำเร็จ อาจเกิด Time out
            self.dobot = Dobot(port=self.port, verbose=False) 
            
            print(f"✅ เชื่อมต่อ Dobot สำเร็จ!")
            
            # ตั้งความเร็วและความเร่ง
            self.dobot.speed(self.speed, self.speed)
            print(f"⚡ ตั้งความเร็ว: {self.speed}%")
            
            self.is_connected = True
            
            # กลับ Home เสมอเมื่อเริ่มต้น (ช่วยลดปัญหา Drift)
            self.move_home("เริ่มต้นระบบและ Homing")
            
            return True
            
        except Exception as e:
            print(f"❌ ไม่สามารถเชื่อมต่อ Dobot: {e}")
            print("💡 เปลี่ยนเป็น Simulation Mode เพื่อทดสอบโค้ด")
            self.simulation_mode = True
            self.is_connected = True
            # ไม่สามารถเชื่อมต่อได้ แต่ยังคงอนุญาตให้รันโค้ดในโหมดจำลอง
            return False
    
    def disconnect(self):
        """ปิดการเชื่อมต่อ"""
        if self.dobot and not self.simulation_mode:
            try:
                print("👋 กำลังปิดการเชื่อมต่อ Dobot...")
                # กลับ Home ก่อนปิด
                self.move_home("กลับฐานก่อนปิด")
                self.dobot.close()
                print("✅ ปิดการเชื่อมต่อเรียบร้อย")
            except Exception as e:
                print(f"⚠️ ข้อผิดพลาดในการปิดการเชื่อมต่อ: {e}")
        
        self.is_connected = False
    
    def move_to(self, x: float, y: float, z: float, r: float, 
                message: str = "", wait: bool = True):
        """
        เคลือนที่ไปยังตำแหน่งที่กำหนด **เน้น wait=True เพื่อความแม่นยำ**
        """
        if message:
            print(f"  ➡️  {message} → ({x:.1f}, {y:.1f}, {z:.1f})")
        
        if self.simulation_mode:
            time.sleep(0.3)  # จำลองเวลาการเคลื่อนที่
            return
        
        if self.dobot:
            try:
                # สำคัญ: การใช้ wait=True เพื่อรอให้คำสั่งเคลื่อนที่ในคิวเสร็จสิ้น
                self.dobot.move_to(x, y, z, r, wait=wait)
            except Exception as e:
                print(f"  ⚠️ ข้อผิดพลาดในการเคลื่อนที่: {e}")
        else:
             print("  ❌ Dobot object is not initialized (Simulation mode or connection failed)")
    
    def suction_on(self):
        """เปิดหัวดูด"""
        print("  🔵 เปิดหัวดูด")
        
        if self.simulation_mode:
            time.sleep(self.SUCTION_DELAY)
            return
        
        if self.dobot:
            try:
                self.dobot.suck(True)
                time.sleep(self.SUCTION_DELAY)
            except Exception as e:
                print(f"  ⚠️ ข้อผิดพลาดในการเปิดหัวดูด: {e}")
    
    def suction_off(self):
        """ปิดหัวดูด"""
        print("  🔴 ปิดหัวดูด")
        
        if self.simulation_mode:
            time.sleep(self.SUCTION_DELAY)
            return
        
        if self.dobot:
            try:
                self.dobot.suck(False)
                time.sleep(self.SUCTION_DELAY)
            except Exception as e:
                print(f"  ⚠️ ข้อผิดพลาดในการปิดหัวดูด: {e}")
    
    def move_home(self, message: str = ""):
        """กลับตำแหน่ง Home"""
        msg = f"Home Position{' - ' + message if message else ''}"
        # ใช้ move_to ซึ่งมีการตั้งค่า wait=True อยู่แล้ว
        self.move_to(*self.HOME, message=msg)
    
    def move_to_pickup(self):
        """ไปยังตำแหน่งหยิบพัสดุ (Pick Sequence)"""
        # 1. ขึ้นไปเหนือจุดหยิบ (SAFETY_Z)
        self.move_to(
            self.PICKUP[0], 
            self.PICKUP[1], 
            self.SAFETY_Z, 
            self.PICKUP[3],
            "เหนือจุดหยิบพัสดุ (Safety Z)"
        )
        
        # 2. ลงไปหยิบ
        self.move_to(*self.PICKUP, message="ลงหยิบพัสดุ")
    
    def move_to_drop(self, province: str) -> bool:
        """
        ไปยังตำแหน่งวางพัสดุตามจังหวัด (Drop Sequence)
        """
        # ค้นหาจุดวาง
        drop_point = self.DROP_POINTS.get(province)
        
        if not drop_point:
            print(f"  ❌ ไม่พบจุดวางสำหรับ '{province}'")
            return False
        
        # 1. ขึ้นไปเหนือจุดวาง (SAFETY_Z)
        self.move_to(
            drop_point[0],
            drop_point[1],
            self.SAFETY_Z,
            drop_point[3],
            f"เหนือจุดวาง '{province}' (Safety Z)"
        )
        
        # 2. ลงไปวาง
        self.move_to(*drop_point, message=f"ลงวางพัสดุที่ '{province}'")
        
        return True
    
    def pick_and_place(self, province: str) -> bool:
        """
        หยิบพัสดุและวางตามจังหวัด (กระบวนการหลัก)
        """
        if not self.is_connected:
            print("❌ Dobot ไม่ได้เชื่อมต่อ")
            return False
        
        print(f"\n{'='*60}")
        print(f"🤖 เริ่มกระบวนการคัดแยก → '{province}'")
        print(f"{'='*60}")
        
        try:
            # 1. ไปหยิบ
            print("📦 [1/5] เคลื่อนที่ไปยังจุดหยิบ...")
            self.move_to_pickup()
            
            print("📦 [2/5] หยิบพัสดุ...")
            self.suction_on()
            
            # ยกขึ้นหลังหยิบ (กลับไปที่ SAFETY_Z ก่อนเคลื่อนที่ไปวาง)
            self.move_to(
                self.PICKUP[0],
                self.PICKUP[1],
                self.SAFETY_Z,
                self.PICKUP[3],
                "ยกพัสดุขึ้น (Safety)"
            )
            
            # 2. ตรวจสอบและไปวาง
            target_province = province if province in self.DROP_POINTS else "จุดสำรอง"
            
            print(f"📦 [3/5] ค้นหา/ยืนยันจุดวาง '{target_province}'...")
            
            # 3. ไปวาง
            print(f"📦 [4/5] เคลื่อนที่ไปยังจุดวาง '{target_province}'...")
            success = self.move_to_drop(target_province)
            
            if not success:
                # ไม่ควรเกิดหากใช้ "จุดสำรอง" แต่เผื่อไว้
                self.stats['failed_drops'] += 1
                return False
            
            print("📦 [5/5] วางพัสดุ...")
            self.suction_off()
            
            # ยกขึ้นหลังวาง (กลับไปที่ SAFETY_Z ก่อนกลับ Home)
            drop_point = self.DROP_POINTS[target_province]
            self.move_to(
                drop_point[0],
                drop_point[1],
                self.SAFETY_Z,
                drop_point[3],
                "ยกขึ้นหลังวาง (Safety)"
            )
            
            # อัพเดทสถิติ
            self.stats['total_picks'] += 1
            self.stats['successful_drops'] += 1
            self.stats['by_province'][target_province] = self.stats['by_province'].get(target_province, 0) + 1
            
            print(f"✅ คัดแยกพัสดุไปยัง '{target_province}' สำเร็จ!")
            
            return True
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในกระบวนการคัดแยก: {e}")
            
            # Emergency: ปิดหัวดูดและกลับ Home (ถ้าทำได้)
            try:
                self.suction_off()
                self.move_home("ฉุกเฉิน")
            except:
                pass
            
            self.stats['failed_drops'] += 1
            return False
    
    def add_drop_point(self, province: str, coordinates: list):
        """
        เพิ่มจุดวางใหม่ (เผื่อไว้สำหรับการขยายจังหวัดในอนาคต)
        """
        if len(coordinates) == 4 and all(isinstance(c, (int, float)) for c in coordinates):
            self.DROP_POINTS[province] = coordinates
            print(f"✅ เพิ่มจุดวาง '{province}' → {coordinates}")
        else:
            print("❌ พิกัดต้องเป็น list ที่มี 4 องค์ประกอบ: [x, y, z, r]")
    
    def get_stats(self) -> Dict:
        """ดึงสถิติการทำงาน"""
        return self.stats.copy()
    
    def print_stats(self):
        """แสดงสถิติการทำงาน"""
        print("\n" + "="*60)
        print("📊 สถิติการทำงานของ Dobot")
        print("="*60)
        print(f"🔢 จำนวนการหยิบทั้งหมด: {self.stats['total_picks']}")
        print(f"✅ วางสำเร็จ: {self.stats['successful_drops']}")
        print(f"❌ วางไม่สำเร็จ: {self.stats['failed_drops']}")
        
        if self.stats['by_province']:
            print("\n📍 สรุปตามจังหวัด:")
            for province, count in sorted(self.stats['by_province'].items()):
                print(f"   - {province}: {count} ชิ้น")
        
        print("="*60)


# ฟังก์ชันทดสอบ
def test_dobot():
    """ฟังก์ชันทดสอบการทำงานของ Dobot"""
    
    print("🧪 ทดสอบการทำงานของ Dobot Controller")
    print("="*60)
    
    # สร้าง controller (Simulation Mode)
    controller = DobotController(
        port="COM5",
        speed=100,
        simulation_mode=True  # ทดสอบโดยไม่ต่อ Dobot จริง
    )
    
    # เชื่อมต่อ
    controller.connect()
    
    # ทดสอบคัดแยกพัสดุ
    # เพิ่มจังหวัดที่ไม่มีใน DROP_POINTS เพื่อทดสอบ "จุดสำรอง"
    test_provinces = ["นครนายก", "เชียงใหม่", "สระบุรี", "ชลบุรี", "นครสวรรค์"]
    
    for province in test_provinces:
        print(f"\n🧪 ทดสอบ: พัสดุปลายทาง '{province}'")
        success = controller.pick_and_place(province)
        
        if success:
            print(f"   ✅ สำเร็จ")
        else:
            print(f"   ❌ ล้มเหลว")
        
        time.sleep(0.5)
    
    # แสดงสถิติ
    controller.print_stats()
    
    # ปิดการเชื่อมต่อ
    controller.disconnect()


if __name__ == "__main__":
    test_dobot()
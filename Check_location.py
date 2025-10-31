import pydobot
from serial.tools import list_ports
import time
import sys

# --- 1. กำหนดพอร์ตและเชื่อมต่อกับ Dobot ---
# 🎯 แก้ไข: กำหนดพอร์ตเป็น 'COM5' ชัดเจนตามที่สแกนพบ
DOBOT_PORT = 'COM3'

print(f"Searching for Dobot on port {DOBOT_PORT}...")

try:
    # 🎯 แก้ไข: ลองเชื่อมต่อโดยใช้พอร์ตที่ระบุ
    device = pydobot.Dobot(port=DOBOT_PORT, verbose=False)
    # ตั้งค่าความเร็ว (สำคัญ)
    device.speed(velocity=100, acceleration=100) 
    
    print(f"✅ Connected to Dobot on port: {DOBOT_PORT}")
    time.sleep(1)

except Exception as e:
    print(f"❌ Error connecting to Dobot on {DOBOT_PORT}: {e}")
    # หากเชื่อมต่อล้มเหลว (เช่น IndexError) จะออกจากโปรแกรม
    print("❌ Failed to establish communication with Dobot. Exiting.")
    sys.exit(1) # ออกจากโปรแกรม

# --- 2. ลูปสำหรับอ่านและบันทึกค่าพิกัด ---
try:
    print("\n--- Dobot Position Recorder ---")
    print("💡 คำแนะนำ: คุณสามารถใช้ Dobot Studio เพื่อปรับพิกัด และดูการเปลี่ยนแปลงในโปรแกรมนี้ได้")
    
    while True:
        # รอให้ผู้ใช้กด Enter
        input(">> Move the arm to your desired position and press Enter to get coordinates... ")
        
        # อ่านค่าพิกัดปัจจุบันจากแขนกล
        # pose() จะคืนค่า (x, y, z, r, j1, j2, j3, j4)
        current_pose = device.pose()
        
        # แสดงผลค่าพิกัด x, y, z, r (จัดรูปแบบทศนิยม 2 ตำแหน่งเพื่อให้อ่านง่าย)
        x, y, z, r = current_pose[0], current_pose[1], current_pose[2], current_pose[3]
        print(f"Position Captured: (x, y, z) = ({x:.2f}, {y:.2f}, {z:.2f}) | r = {r:.2f}")
        print("-" * 35)

        # ถามว่าจะบันทึกตำแหน่งต่อไปหรือไม่
        again = input("Record another position? (y/n): ").lower()
        if again != 'y':
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")
except Exception as e:
    print(f"\nAn unexpected error occurred during recording: {e}")
finally:
    # --- 3. ปิดการเชื่อมต่อ ---
    print("Closing connection to Dobot.")
    try:
        device.close()
    except:
        pass # ป้องกัน error ซ้ำซ้อน

#process.py
import cv2
from openai import OpenAI
import json
import re
from datetime import datetime
from PIL import Image
import base64
import os
from typing import Dict, Optional, List
import numpy as np
import requests


class DatabaseAPI:
    """Class สำหรับเชื่อมต่อกับ PHP API และบันทึกข้อมูลลง MySQL"""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',  # ใช้ Authorization แทน
            'Content-Type': 'application/json'
        })
    
    def save_parcel_to_db(self, parcel_data: Dict) -> Dict:
        """
        บันทึกข้อมูลพัสดุลง database
        
        Args:
            parcel_data: ข้อมูลพัสดุที่ได้จาก OCR
            
        Returns:
            response จาก API
        """
        try:
            # เตรียมข้อมูลให้ตรงกับ structure ที่ PHP API ต้องการ
            payload = self._prepare_payload(parcel_data)
            
            print(f"📤 กำลังส่งข้อมูลไปยัง: {self.api_url}")
            print(f"📦 Payload Preview:")
            print(f"   - Sender: {payload['sender']['name']} ({payload['sender']['province']})")
            print(f"   - Recipient: {payload['recipient']['name']} ({payload['recipient']['province']})")
            print(f"   - Tracking: {payload['parcel']['tracking_number']}")
            
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            if result.get('success'):
                print(f"✅ บันทึกข้อมูลสำเร็จ - Parcel ID: {result.get('parcel_id')}")
            else:
                print(f"❌ บันทึกไม่สำเร็จ: {result.get('message')}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ API: {e}")
            return {
                "success": False,
                "message": str(e)
            }
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")
            return {
                "success": False,
                "message": str(e)
            }
    
    def _prepare_payload(self, parcel_data: Dict) -> Dict:
        """
        แปลงข้อมูลจาก OCR เป็น format ที่ database ต้องการ
        """
        sender = parcel_data.get('sender', {})
        recipient = parcel_data.get('recipient', {})
        
        payload = {
            "action": "create_parcel",  # ระบุ action สำหรับ PHP
            "sender": {
                "name": sender.get('name', ''),
                "phone": sender.get('phone', ''),
                "address_details": sender.get('address', ''),
                "province": sender.get('province', '')
            },
            "recipient": {
                "name": recipient.get('name', ''),
                "phone": recipient.get('phone', ''),
                "address_details": recipient.get('address', ''),
                "province": recipient.get('province', '')
            },
            "parcel": {
                "status": "Pending",
                "tracking_number": parcel_data.get('tracking_number', '')
            }
        }
        
        return payload
    
    def get_parcel_by_tracking(self, tracking_number: str) -> Optional[Dict]:
        """ดึงข้อมูลพัสดุจาก tracking number"""
        try:
            payload = {
                "action": "get_parcel",
                "tracking_number": tracking_number
            }
            
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"❌ ไม่สามารถดึงข้อมูล: {e}")
            return None
    
    def update_parcel_status(self, parcel_id: int, status: str) -> Dict:
        """อัพเดทสถานะพัสดุ"""
        try:
            payload = {
                "action": "update_status",
                "parcel_id": parcel_id,
                "status": status
            }
            
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"❌ ไม่สามารถอัพเดทสถานะ: {e}")
            return {
                "success": False,
                "message": str(e)
            }


class TyphoonOCR:
    """Class สำหรับเชื่อมต่อกับ Typhoon OCR API"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.opentyphoon.ai/v1"
        )
        
    def extract_text_from_image(self, image_path: str) -> str:
        """อ่านข้อความจากรูปภาพด้วย Typhoon Vision"""
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self.client.chat.completions.create(
                model="typhoon-ocr-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an OCR expert. Extract data from parcel labels and return ONLY valid JSON format. No other text."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": """Read this parcel label and extract:
- Sender (name, phone, address, province)
- Recipient (name, phone, address, province)
- Tracking number (if any)

Return ONLY this JSON structure (no markdown, no explanation):
{
    "sender": {
        "name": "sender name",
        "phone": "phone number",
        "address": "full address",
        "province": "province name"
    },
    "recipient": {
        "name": "recipient name",
        "phone": "phone number",
        "address": "full address",
        "province": "province name"
    },
    "tracking_number": "tracking number or empty string"
}"""
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ Error calling Typhoon API: {e}")
            return ""


class ImagePreprocessor:
    """Class สำหรับปรับปรุงคุณภาพภาพก่อนทำ OCR"""
    
    def __init__(self, output_folder: str = "enhanced_images"):
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
    def enhance_image(self, image_path: str) -> str:
        """ปรับปรุงคุณภาพภาพ"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ ไม่สามารถอ่านภาพ: {image_path}")
                return image_path
            
            # แปลงเป็น grayscale
    
            
            # ปรับความคมชัด
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img)
            
            # ลด noise
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # ปรับ threshold
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # บันทึกภาพที่ปรับปรุงแล้ว
            filename = os.path.basename(image_path)
            output_path = os.path.join(self.output_folder, f"enhanced_{filename}")
            cv2.imwrite(output_path, binary)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error enhancing image: {e}")
            return image_path


class CompleteParcelSortingSystem:
    """ระบบคัดแยกพัสดุแบบครบวงจร (รองรับ Database)"""
    
    def __init__(self, typhoon_api_key: str, output_folder: str = "parcel_results",
                 db_api_url: str = None, db_api_key: str = None):
        self.ocr = TyphoonOCR(typhoon_api_key)
        self.preprocessor = ImagePreprocessor()
        self.output_folder = output_folder
        self.sorted_folder = os.path.join(output_folder, "sorted_by_province")
        
        # เชื่อมต่อ Database API (ถ้ามี)
        self.db_api = None
        if db_api_url and db_api_key:
            self.db_api = DatabaseAPI(db_api_url, db_api_key)
            print("✅ เชื่อมต่อ Database API สำเร็จ")
        else:
            print("⚠️  ไม่ได้เชื่อมต่อ Database API (จะบันทึกเฉพาะ JSON)")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        if not os.path.exists(self.sorted_folder):
            os.makedirs(self.sorted_folder)
        
        # รายชื่อจังหวัดทั้งหมด
        self.all_provinces = [
            "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
            "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท", "ชัยภูมิ",
            "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง", "ตราด", "ตาก",
            "นครนายก", "นครปฐม", "นครพนม", "นครราชสีมา", "นครศรีธรรมราช",
            "นครสวรรค์", "นนทบุรี", "นราธิวาส", "น่าน", "บึงกาฬ", "บุรีรัมย์",
            "ปทุมธานี", "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา",
            "พะเยา", "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
            "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
            "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี",
            "ลพบุรี", "ลำปาง", "ลำพูน", "เลย", "ศรีสะเกษ", "สกลนคร",
            "สงขลา", "สตูล", "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร",
            "สระแก้ว", "สระบุรี", "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี",
            "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย", "หนองบัวลำภู",
            "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์", "อุทัยธานี", "อุบลราชธานี"
        ]
        
        # กำหนดเส้นทางการจัดส่งตามภาค
        self.region_mapping = {
            "ภาคเหนือ": ["เชียงใหม่", "เชียงราย", "ลำปาง", "ลำพูน", "แม่ฮ่องสอน", 
                         "น่าน", "พะเยา", "แพร่", "อุตรดิตถ์", "ตาก", "สุโขทัย", 
                         "พิษณุโลก", "เพชรบูรณ์", "กำแพงเพชร", "นครสวรรค์", "พิจิตร", "อุทัยธานี"],
            
            "ภาคตะวันออกเฉียงเหนือ": ["นครราชสีมา", "ขอนแก่น", "อุดรธานี", "อุบลราชธานี",
                                      "บุรีรัมย์", "สุรินทร์", "ศรีสะเกษ", "ยโสธร", "ชัยภูมิ",
                                      "มหาสารคาม", "ร้อยเอ็ด", "กาฬสินธุ์", "สกลนคร", "นครพนม",
                                      "มุกดาหาร", "เลย", "หนองคาย", "บึงกาฬ", "หนองบัวลำภู", "อำนาจเจริญ"],
            
            "ภาคกลาง": ["กรุงเทพมหานคร", "นนทบุรี", "ปทุมธานี", "สมุทรปราการ",
                        "นครปฐม", "สมุทรสาคร", "สมุทรสงคราม", "พระนครศรีอยุธยา",
                        "อ่างทอง", "สิงห์บุรี", "ชัยนาท", "ลพบุรี", "สระบุรี"],
            
            "ภาคตะวันออก": ["ชลบุรี", "ระยอง", "จันทบุรี", "ตราด", "ฉะเชิงเทรา",
                           "ปราจีนบุรี", "นครนายก", "สระแก้ว"],
            
            "ภาคตะวันตก": ["กาญจนบุรี", "ราชบุรี", "สุพรรณบุรี", "เพชรบุรี", 
                          "ประจวบคีรีขันธ์"],
            
            "ภาคใต้": ["สงขลา", "ภูเก็ต", "สุราษฎร์ธานี", "นครศรีธรรมราช", "ตรัง",
                      "พัทลุง", "ปัตตานี", "ยะลา", "นราธิวาส", "กระบี่", "พังงา",
                      "ระนอง", "ชุมพร", "สตูล"]
        }
        
        # สถิติการประมวลผล
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "db_saved": 0,
            "db_failed": 0,
            "by_region": {},
            "by_province": {}
        }
        
        # เก็บข้อมูลพัสดุทั้งหมด
        self.parcels = []
    
    def get_region(self, province: str) -> str:
        """หาภาคจากจังหวัด"""
        for region, provinces in self.region_mapping.items():
            if province in provinces:
                return region
        return "ไม่ระบุภาค"
    
    def extract_province_from_text(self, text: str) -> Optional[str]:
        """
        ค้นหาชื่อจังหวัดจากข้อความ
        รองรับรูปแบบ: "จ.ระยอง", "จังหวัดระยอง", "ระยอง"
        """
        if not text:
            return None
        
        # ทำความสะอาดข้อความ
        text = text.strip()
        
        # Pattern 1: จ.ชื่อจังหวัด หรือ จังหวัดชื่อจังหวัด
        province_patterns = [
            r'จ\.([^\s\d]+)',
            r'จังหวัด([^\s\d]+)',
        ]
        
        for pattern in province_patterns:
            match = re.search(pattern, text)
            if match:
                province_candidate = match.group(1).strip()
                # ตรวจสอบว่าตรงกับจังหวัดที่มีหรือไม่
                for province in self.all_provinces:
                    if province_candidate == province or province_candidate in province:
                        return province
        
        # Pattern 2: ค้นหาชื่อจังหวัดโดยตรง
        for province in self.all_provinces:
            # ตรงทั้งหมด
            if province in text:
                return province
            
            # ตรงแบบไม่มีช่องว่าง
            province_no_space = province.replace(" ", "")
            text_no_space = text.replace(" ", "")
            if province_no_space in text_no_space:
                return province
        
        # Pattern 3: ลองแบบ fuzzy matching
        text_words = set(re.findall(r'[\u0E00-\u0E7F]+', text))
        for province in self.all_provinces:
            province_words = set(re.findall(r'[\u0E00-\u0E7F]+', province))
            if province_words.intersection(text_words):
                return province
        
        return None
    
    def smart_split_sender_recipient(self, text: str) -> tuple:
        """
        🎯 ฟังก์ชันใหม่: แยกข้อความเป็นส่วนผู้ส่งและผู้รับด้วย NLP patterns
        
        Returns:
            (sender_text, recipient_text)
        """
        print("\n🧠 กำลังใช้ NLP แยกผู้ส่ง-ผู้รับ...")
        
        # Method 1: หาด้วยคำว่า "ผู้ส่ง" และ "ผู้รับ" (ภาษาไทย)
        sender_markers = [
            r'ผู้ส่ง\s*[:：]?',
            r'sender\s*[:：]?',
            r'from\s*[:：]?'
        ]
        
        recipient_markers = [
            r'ผู้รับ\s*[:：]?',
            r'recipient\s*[:：]?',
            r'to\s*[:：]?'
        ]
        
        # หา position ของ marker แต่ละตัว
        sender_positions = []
        recipient_positions = []
        
        for pattern in sender_markers:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                sender_positions.append(match.start())
        
        for pattern in recipient_markers:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                recipient_positions.append(match.start())
        
        # ถ้าเจอทั้งสอง marker
        if sender_positions and recipient_positions:
            # เอา marker แรกของแต่ละฝั่ง
            sender_start = min(sender_positions)
            recipient_start = min(recipient_positions)
            
            print(f"   ✅ เจอ marker: ผู้ส่ง @{sender_start}, ผู้รับ @{recipient_start}")
            
            if sender_start < recipient_start:
                sender_text = text[sender_start:recipient_start].strip()
                recipient_text = text[recipient_start:].strip()
            else:
                recipient_text = text[recipient_start:sender_start].strip()
                sender_text = text[sender_start:].strip()
            
            return sender_text, recipient_text
        
        # Method 2: ถ้าไม่เจอ marker ชัดเจน ลองแบ่งตามลำดับการปรากฏของข้อมูล
        print("   ⚠️ ไม่เจอ marker ชัดเจน - ใช้วิธีแบ่งตามโครงสร้าง...")
        
        phone_pattern = r'(?:โทร|tel|phone)\s*[:：]?\s*[0-9\s\-]{9,}'
        phones = list(re.finditer(phone_pattern, text, re.IGNORECASE))
        
        if len(phones) >= 2:
            # แบ่งระหว่างเบอร์โทร set แรกกับ set ที่สอง
            split_point = phones[1].start()
            
            # ถอยหลังไปหา marker ของผู้รับ
            before_split = text[:split_point]
            for pattern in recipient_markers:
                match = re.search(pattern + r'.*$', before_split, re.IGNORECASE)
                if match:
                    split_point = match.start()
                    break
            
            sender_text = text[:split_point].strip()
            recipient_text = text[split_point:].strip()
            
            print(f"   ✅ แบ่งตามเบอร์โทร @ position {split_point}")
            return sender_text, recipient_text
        
        # Method 3: แบ่งครึ่ง (fallback)
        print("   ⚠️ ใช้วิธี fallback: แบ่งครึ่ง")
        lines = text.split('\n')
        mid = len(lines) // 2
        
        sender_text = '\n'.join(lines[:mid])
        recipient_text = '\n'.join(lines[mid:])
        
        return sender_text, recipient_text
    
    def normalize_ocr_json_result(self, data: Dict) -> Dict:
        """
        🎯 ฟังก์ชันใหม่: ตรวจสอบและแก้ไข JSON structure ที่ OCR ส่งมาผิดพลาด
        """
        if not isinstance(data, dict):
            return data
        
        print("\n🔧 ตรวจสอบโครงสร้าง JSON...")
        
        # กรณี 1: sender เป็น string ยาว ๆ (ผิดปกติ)
        sender = data.get('sender', {})
        
        if isinstance(sender, str) and len(sender) > 100:
            print("   ⚠️ ตรวจพบ: sender เป็น string ยาว (ไม่ใช่ dict)")
            print(f"   📏 ความยาว: {len(sender)} ตัวอักษร")
            
            # แยกข้อความออกเป็นผู้ส่ง-ผู้รับ
            sender_text, recipient_text = self.smart_split_sender_recipient(sender)
            
            print(f"\n   📤 ส่วนผู้ส่ง ({len(sender_text)} ตัวอักษร):")
            print("   " + "-" * 50)
            print("   " + sender_text[:150].replace('\n', '\n   '))
            
            print(f"\n   📥 ส่วนผู้รับ ({len(recipient_text)} ตัวอักษร):")
            print("   " + "-" * 50)
            print("   " + recipient_text[:150].replace('\n', '\n   '))
            
            # แปลงเป็น dict
            sender_dict = self._extract_person_info(sender_text, is_recipient=False)
            recipient_dict = self._extract_person_info(recipient_text, is_recipient=True)
            
            # สร้าง structure ใหม่
            data['sender'] = sender_dict
            data['recipient'] = recipient_dict
            
            print("\n   ✅ แก้ไขโครงสร้าง JSON สำเร็จ")
        
        # กรณี 2: มี recipient แล้ว แต่เป็น string
        elif isinstance(data.get('recipient'), str):
            print("   ⚠️ ตรวจพบ: recipient เป็น string (ควรเป็น dict)")
            recipient_text = data.get('recipient', '')
            data['recipient'] = self._extract_person_info(recipient_text, is_recipient=True)
        
        # กรณี 3: ไม่มี recipient เลย
        elif 'recipient' not in data or not data['recipient']:
            print("   ⚠️ ตรวจพบ: ไม่มีข้อมูล recipient")
            
            # ลองหาจาก sender (กรณี OCR ใส่รวมกัน)
            if isinstance(sender, str):
                sender_text, recipient_text = self.smart_split_sender_recipient(sender)
                data['sender'] = self._extract_person_info(sender_text, is_recipient=False)
                data['recipient'] = self._extract_person_info(recipient_text, is_recipient=True)
        
        return data
    
    def normalize_province_data(self, data: Dict) -> Dict:
        """ปรับแต่งข้อมูลจังหวัดให้ถูกต้อง"""
        if not data:
            return data
        
        print("\n🔍 กำลังตรวจสอบและแก้ไขจังหวัด...")
        
        # ตรวจสอบและแก้ไขชื่อจังหวัดของผู้ส่ง
        sender = data.get('sender', {})
        if isinstance(sender, dict):
            sender_province = sender.get('province', '')
            if sender_province:
                print(f"   📤 ผู้ส่ง - จังหวัด: '{sender_province}'")
                normalized = self.extract_province_from_text(sender_province)
                if normalized:
                    sender['province'] = normalized
                    print(f"       → ปรับเป็น: '{normalized}' ✅")
        
        # ตรวจสอบและแก้ไขชื่อจังหวัดของผู้รับ (สำคัญ!)
        recipient = data.get('recipient', {})
        if isinstance(recipient, dict):
            recipient_province = recipient.get('province', '')
            print(f"   📥 ผู้รับ - จังหวัดเดิม: '{recipient_province}'")
            
            if recipient_province:
                normalized = self.extract_province_from_text(recipient_province)
                if normalized:
                    recipient['province'] = normalized
                    print(f"       → ปรับเป็น: '{normalized}' ✅")
                else:
                    print(f"       → ไม่ตรงกับจังหวัดที่มี - ลองหาจากที่อยู่...")
                    address = recipient.get('address', '')
                    if address:
                        normalized = self.extract_province_from_text(address)
                        if normalized:
                            recipient['province'] = normalized
                            print(f"       → พบจากที่อยู่: '{normalized}' ✅")
            else:
                print(f"       → ไม่มีข้อมูลจังหวัด - ลองหาจากที่อยู่...")
                address = recipient.get('address', '')
                if address:
                    normalized = self.extract_province_from_text(address)
                    if normalized:
                        recipient['province'] = normalized
                        print(f"       → พบจากที่อยู่: '{normalized}' ✅")
                    else:
                        print(f"       → ไม่พบจังหวัด ❌")
        
        return data
    
    def determine_delivery_route(self, province: str, region: str) -> str:
        """กำหนดเส้นทางการส่งพัสดุ"""
        if province == "ไม่ระบุ" or region == "ไม่ระบุภาค":
            return "❌ ไม่สามารถระบุเส้นทางได้ - ต้องตรวจสอบด้วยตนเอง"
        
        distribution_centers = {
            "ภาคเหนือ": "ศูนย์กระจายสินค้าภาคเหนือ (เชียงใหม่)",
            "ภาคตะวันออกเฉียงเหนือ": "ศูนย์กระจายสินค้าภาคตะวันออกเฉียงเหนือ (นครราชสีมา)",
            "ภาคกลาง": "ศูนย์กระจายสินค้าภาคกลาง (กรุงเทพฯ)",
            "ภาคตะวันออก": "ศูนย์กระจายสินค้าภาคตะวันออก (ชลบุรี)",
            "ภาคตะวันตก": "ศูนย์กระจายสินค้าภาคตะวันตก (กาญจนบุรี)",
            "ภาคใต้": "ศูนย์กระจายสินค้าภาคใต้ (สงขลา)"
        }
        
        center = distribution_centers.get(region, "ศูนย์กระจายสินค้าทั่วไป")
        return f"✅ ส่งไป {center} → {province}"
    
    def parse_ocr_result(self, ocr_text: str) -> Optional[Dict]:
        """แปลงผลลัพธ์ OCR เป็น JSON - รองรับหลายรูปแบบ"""
        try:
            # ลบ markdown code blocks
            text = ocr_text.strip()
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'^```\s*', '', text)
            text = re.sub(r'\s*```', '', text)
            text = text.strip()
            
            # ลองแปลง JSON
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
            
            # ลองหา JSON object
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass
            
            return None
            
        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
            return None
    
    def _extract_person_info(self, text: str, is_recipient: bool = False) -> Dict:
        """แยกข้อมูลบุคคลจากข้อความ"""
        info = {"name": "", "phone": "", "address": "", "province": ""}
        
        try:
            prefix = "   📥" if is_recipient else "   📤"
            
            # Extract name
            name_patterns = [
                r'(?:ชื่อ|name)[:\s]*([^\n\r]+?)(?=\n|ที่อยู่|address|โทร|tel|$)',
                r'^([^\n\r]+?)(?=\n)',  # first line
            ]
            for pattern in name_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    name = match.group(1).strip()
                    # Clean up
                    name = re.sub(r'(?:ชื่อ|name)[:\s]*', '', name, flags=re.IGNORECASE)
                    if name and len(name) > 2:
                        info["name"] = name
                        print(f"{prefix} ชื่อ: {name}")
                        break
            
            # Extract phone
            phone_patterns = [
                r'(?:โทร|tel|phone)[:\s]*([0-9\s\-]{9,})',
                r'\b(0\d{2}[\s\-]?\d{3}[\s\-]?\d{4})\b',
                r'\b(\d{3}[\s\-]?\d{3}[\s\-]?\d{4})\b'
            ]
            for pattern in phone_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    phone = match.group(1).strip()
                    phone = re.sub(r'[^\d]', '', phone)  # keep only digits
                    if len(phone) >= 9:
                        info["phone"] = phone
                        print(f"{prefix} โทร: {phone}")
                        break
            
            # Extract address and province
            address_patterns = [
                r'(?:ที่อยู่|address)[:\s]*([^\n\r]+?)(?=\n(?:โทร|tel)|$)',
                r'(?:จ\.|จังหวัด)[\s]*([^\n\r]+)',
            ]
            
            for pattern in address_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    address = match.group(1).strip()
                    # Clean up
                    address = re.sub(r'(?:ที่อยู่|address)[:\s]*', '', address, flags=re.IGNORECASE)
                    if address:
                        info["address"] = address
                        print(f"{prefix} ที่อยู่: {address[:50]}...")
                        break
            
            # หาจังหวัดโดยเฉพาะสำหรับผู้รับ
            if is_recipient:
                print(f"{prefix} 🎯 กำลังค้นหาจังหวัดปลายทาง...")
                
                # Layer 1: หาจากที่อยู่
                if info["address"]:
                    province = self.extract_province_from_text(info["address"])
                    if province:
                        info["province"] = province
                        print(f"{prefix} ✅ พบจังหวัดจากที่อยู่: '{province}'")
                
                # Layer 2: หาจากทั้งข้อความ
                if not info["province"]:
                    province = self.extract_province_from_text(text)
                    if province:
                        info["province"] = province
                        print(f"{prefix} ✅ พบจังหวัดจากข้อความทั้งหมด: '{province}'")
                
                # Layer 3: หาจาก pattern "จ." หรือ "จังหวัด"
                if not info["province"]:
                    province_patterns = [
                        r'(?:จ\.|จังหวัด)\s*([^\s\d,]+)',
                    ]
                    for pattern in province_patterns:
                        match = re.search(pattern, text)
                        if match:
                            province_name = match.group(1).strip()
                            province = self.extract_province_from_text(province_name)
                            if province:
                                info["province"] = province
                                print(f"{prefix} ✅ พบจังหวัดจาก pattern: '{province}'")
                                break
                
                if not info["province"]:
                    print(f"{prefix} ❌ ไม่พบจังหวัดในข้อความผู้รับ")
            else:
                # ผู้ส่ง: หาจังหวัดแบบธรรมดา
                if info["address"]:
                    province = self.extract_province_from_text(info["address"])
                    if province:
                        info["province"] = province
                        print(f"{prefix} จังหวัด: {province}")
            
        except Exception as e:
            print(f"⚠️ Error parsing person info: {e}")
        
        return info
    
    def extract_data_from_text(self, text: str) -> Dict:
        """แยกข้อมูลจาก text ที่ OCR อ่านได้"""
        result = {
            "sender": {"name": "", "phone": "", "address": "", "province": ""},
            "recipient": {"name": "", "phone": "", "address": "", "province": ""},
            "tracking_number": ""
        }
        
        try:
            print("\n🔍 กำลังแยกบริบทผู้ส่งและผู้รับ...")
            
            # ใช้ฟังก์ชันใหม่ในการแยก
            sender_text, recipient_text = self.smart_split_sender_recipient(text)
            
            # Debug: แสดงข้อความที่แยกได้
            if sender_text:
                print(f"\n📤 ข้อความฝั่งผู้ส่ง:")
                print("-" * 60)
                print(sender_text[:200] + "..." if len(sender_text) > 200 else sender_text)
                print("-" * 60)
            
            if recipient_text:
                print(f"\n📥 ข้อความฝั่งผู้รับ (FOCUS ที่นี่):")
                print("-" * 60)
                print(recipient_text[:200] + "..." if len(recipient_text) > 200 else recipient_text)
                print("-" * 60)
            
            # Extract ข้อมูลผู้ส่ง
            if sender_text:
                result["sender"] = self._extract_person_info(sender_text, is_recipient=False)
            
            # Extract ข้อมูลผู้รับ (โฟกัสหาจังหวัด)
            if recipient_text:
                result["recipient"] = self._extract_person_info(recipient_text, is_recipient=True)
            
            # Extract tracking number
            tracking_patterns = [
                r'tracking[_\s]*(?:number)?[:\s]*([A-Z0-9]+)',
                r'เลข(?:พัสดุ|ติดตาม)?[:\s]*([A-Z0-9]+)',
                r'\b([A-Z]{2,}\d{8,})\b'
            ]
            
            for pattern in tracking_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result["tracking_number"] = match.group(1).strip()
                    break
            
        except Exception as e:
            print(f"⚠️ Error extracting text: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def process_single_parcel(self, image_path: str, enhance_image: bool = True, 
                             save_to_db: bool = True) -> Dict:
        """ประมวลผลพัสดุเดียวแบบครบวงจร (รองรับ Database)"""
        print(f"\n📦 กำลังประมวลผลพัสดุ: {os.path.basename(image_path)}")
        
        result = {
            "success": False,
            "image_file": os.path.basename(image_path),
            "image_path": image_path,
            "tracking_number": None,
            "province": None,
            "region": None,
            "data": None,
            "db_saved": False,
            "db_parcel_id": None,
            "error": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 1. ปรับปรุงภาพถ้าต้องการ
            processed_image = image_path
            if enhance_image:
                print("📸 กำลังปรับปรุงคุณภาพภาพ...")
                processed_image = self.preprocessor.enhance_image(image_path)
                print(f"✅ ปรับปรุงภาพสำเร็จ")
            
            # 2. อ่านข้อมูลด้วย OCR
            print("🔍 กำลังอ่านข้อมูลด้วย Typhoon Vision OCR...")
            ocr_result = self.ocr.extract_text_from_image(processed_image)
            
            if not ocr_result:
                result["error"] = "ไม่สามารถอ่านข้อมูลจากภาพได้"
                self.stats["failed"] += 1
                print(f"❌ {result['error']}")
                return result
            
            # Debug: แสดงผล OCR ดิบ
            print(f"\n📝 OCR Result (raw):")
            print("=" * 60)
            print(ocr_result[:500] + "..." if len(ocr_result) > 500 else ocr_result)
            print("=" * 60)
            
            # 3. แปลงผลลัพธ์เป็น JSON
            parcel_data = self.parse_ocr_result(ocr_result)
            
            # ถ้าไม่ได้ JSON ให้ลองแยกข้อมูลจาก text
            if not parcel_data:
                print("⚠️ ไม่ได้รับ JSON จาก OCR - ลองแยกข้อมูลจาก text...")
                
                if isinstance(ocr_result, str):
                    parcel_data = self.extract_data_from_text(ocr_result)
                else:
                    result["error"] = "ไม่สามารถแปลงข้อมูลเป็น JSON ได้"
                    result["raw_ocr"] = ocr_result
                    self.stats["failed"] += 1
                    print(f"❌ {result['error']}")
                    return result
            
            # ถ้ายังไม่ได้ข้อมูล
            if not parcel_data or (not parcel_data.get('sender') and not parcel_data.get('recipient')):
                result["error"] = "ไม่สามารถดึงข้อมูลพัสดุได้"
                result["raw_ocr"] = ocr_result
                self.stats["failed"] += 1
                print(f"❌ {result['error']}")
                return result
            
            # Debug: แสดงข้อมูลที่ได้จาก OCR
            print(f"📄 ข้อมูลที่อ่านได้:")
            print(json.dumps(parcel_data, ensure_ascii=False, indent=2))
            
            # 3.5 🎯 ใช้ฟังก์ชันใหม่: แก้ไขโครงสร้าง JSON
            parcel_data = self.normalize_ocr_json_result(parcel_data)
            
            # 3.6 ปรับแต่งชื่อจังหวัดให้ถูกต้อง
            parcel_data = self.normalize_province_data(parcel_data)
            
            # Debug: แสดงข้อมูลหลัง normalize
            print(f"📝 ข้อมูลหลังปรับแต่ง:")
            print(json.dumps(parcel_data, ensure_ascii=False, indent=2))
            
            # 4. สร้าง tracking number ถ้าไม่มี
            tracking = parcel_data.get('tracking_number')
            if not tracking or tracking == "":
                tracking = f"PKG{datetime.now().strftime('%Y%m%d%H%M%S')}"
                parcel_data['tracking_number'] = tracking
            
            # 5. ดึงข้อมูลจังหวัดปลายทาง
            recipient = parcel_data.get('recipient', {})
            province = 'ไม่ระบุ'
            
            print("\n" + "="*60)
            print("🎯 กำลังกำหนดจังหวัดปลายทาง (จากผู้รับเท่านั้น)")
            print("="*60)
            
            if isinstance(recipient, dict):
                province = recipient.get('province', '')
                
                if province and province != 'ไม่ระบุ':
                    print(f"✅ พบจังหวัดจากข้อมูลผู้รับ: '{province}'")
                else:
                    print("⚠️ ไม่มีจังหวัดในข้อมูลผู้รับ - กำลังค้นหา...")
                    
                    recipient_address = recipient.get('address', '')
                    if recipient_address:
                        print(f"🔍 ค้นหาจากที่อยู่ผู้รับ: '{recipient_address[:50]}...'")
                        found_province = self.extract_province_from_text(recipient_address)
                        if found_province:
                            province = found_province
                            recipient['province'] = province
                            print(f"✅ พบจังหวัดจากที่อยู่ผู้รับ: '{province}'")
                    
                    if not province or province == 'ไม่ระบุ':
                        print("❌ ไม่พบจังหวัดในข้อมูลผู้รับ")
                        province = 'ไม่ระบุ'
            else:
                print(f"❌ ข้อมูลผู้รับไม่ถูกต้อง: {type(recipient)}")
            
            print("="*60)
            
            # 6. หาภาค
            region = self.get_region(province)
            print(f"🗺️  ภาค: '{region}'")
            
            # 7. กำหนดเส้นทาง
            delivery_route = self.determine_delivery_route(province, region)
            
            # 8. อัพเดทผลลัพธ์
            result.update({
                "success": True,
                "tracking_number": tracking,
                "province": province,
                "region": region,
                "delivery_route": delivery_route,
                "data": parcel_data
            })
            
            # 9. บันทึกลง Database (ถ้าเปิดใช้งาน)
            if save_to_db and self.db_api:
                print("\n💾 กำลังบันทึกข้อมูลลง Database...")
                
                db_response = self.db_api.save_parcel_to_db(parcel_data)
                
                if db_response.get('success'):
                    result["db_saved"] = True
                    result["db_parcel_id"] = db_response.get('parcel_id')
                    result["db_sender_id"] = db_response.get('sender_id')
                    result["db_recipient_id"] = db_response.get('recipient_id')
                    self.stats["db_saved"] += 1
                    print(f"✅ บันทึกลง Database สำเร็จ - Parcel ID: {result['db_parcel_id']}")
                else:
                    result["db_saved"] = False
                    result["db_error"] = db_response.get('message')
                    self.stats["db_failed"] += 1
                    print(f"❌ บันทึกลง Database ไม่สำเร็จ: {result['db_error']}")
            
            # 10. อัพเดทสถิติ
            self.stats["successful"] += 1
            self.stats["by_region"][region] = self.stats["by_region"].get(region, 0) + 1
            self.stats["by_province"][province] = self.stats["by_province"].get(province, 0) + 1
            
            # 11. เก็บข้อมูล
            self.parcels.append(result)
            
            print(f"\n✅ อ่านข้อมูลสำเร็จ:")
            print(f"   🔢 Tracking: {tracking}")
            print(f"   📍 ปลายทาง: {province} ({region})")
            print(f"   🚚 เส้นทางการส่ง: {delivery_route}")
            if result.get("db_saved"):
                print(f"   💾 Database ID: {result['db_parcel_id']}")
            
        except Exception as e:
            result["error"] = str(e)
            self.stats["failed"] += 1
            print(f"❌ เกิดข้อผิดพลาด: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.stats["total_processed"] += 1
        
        return result
    
    def batch_process(self, image_folder: str, enhance_images: bool = True, 
                     save_to_db: bool = True) -> List[Dict]:
        """ประมวลผลพัสดุหลายรายการ"""
        results = []
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        
        print("\n" + "="*60)
        print("🚀 เริ่มประมวลผลพัสดุทั้งหมด")
        print("="*60)
        
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(supported_formats)]
        
        total_files = len(image_files)
        print(f"📦 พบพัสดุทั้งหมด: {total_files} รายการ\n")
        
        for idx, filename in enumerate(image_files, 1):
            print(f"\n[{idx}/{total_files}] {filename}")
            print("-" * 60)
            
            image_path = os.path.join(image_folder, filename)
            result = self.process_single_parcel(image_path, enhance_images, save_to_db)
            results.append(result)
        
        return results
    
    def save_individual_json(self, result: Dict) -> Optional[str]:
        """บันทึกพัสดุแต่ละรายการ"""
        if not result["success"]:
            return None
        
        tracking = result["tracking_number"]
        province = result["province"]
        
        province_folder = os.path.join(self.sorted_folder, province)
        if not os.path.exists(province_folder):
            os.makedirs(province_folder)
        
        filename = f"{tracking}.json"
        filepath = os.path.join(province_folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def copy_image_to_province_folder(self, result: Dict) -> Optional[str]:
        """คัดลอกภาพพัสดุ"""
        if not result["success"]:
            return None
        
        province = result["province"]
        image_path = result["image_path"]
        
        province_folder = os.path.join(self.sorted_folder, province)
        if not os.path.exists(province_folder):
            os.makedirs(province_folder)
        
        import shutil
        image_filename = os.path.basename(image_path)
        dest_path = os.path.join(province_folder, image_filename)
        
        try:
            shutil.copy2(image_path, dest_path)
            return dest_path
        except Exception as e:
            print(f"⚠️ ไม่สามารถคัดลอกภาพ: {e}")
            return None
    
    def save_batch_json(self, results: List[Dict], output_file: str = None) -> str:
        """บันทึกผลลัพธ์รวม"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"batch_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_folder, output_file)
        
        batch_data = {
            "batch_info": {
                "total_processed": self.stats["total_processed"],
                "successful": self.stats["successful"],
                "failed": self.stats["failed"],
                "db_saved": self.stats["db_saved"],
                "db_failed": self.stats["db_failed"],
                "timestamp": datetime.now().isoformat()
            },
            "statistics": {
                "by_region": self.stats["by_region"],
                "by_province": self.stats["by_province"]
            },
            "parcels": results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 บันทึกผลลัพธ์รวมที่: {filepath}")
        return filepath
    
    def generate_sorting_report(self, results: List[Dict]) -> str:
        """สร้างรายงาน"""
        report = []
        report.append("="*80)
        report.append("📊 รายงานการคัดแยกพัสดุตามจังหวัด (พร้อม Database Integration)")
        report.append("="*80)
        report.append(f"วันที่: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        report.append("")
        
        report.append("📈 สรุปภาพรวม")
        report.append("-" * 80)
        report.append(f"พัสดุทั้งหมด:        {self.stats['total_processed']} รายการ")
        report.append(f"ประมวลผลสำเร็จ:    {self.stats['successful']} รายการ")
        report.append(f"ประมวลผลไม่สำเร็จ: {self.stats['failed']} รายการ")
        
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            report.append(f"อัตราความสำเร็จ:    {success_rate:.2f}%")
        
        report.append(f"จำนวนจังหวัด:      {len(self.stats['by_province'])} จังหวัด")
        report.append("")
        
        # เพิ่มสถิติ Database
        if self.db_api:
            report.append("💾 สถิติ Database")
            report.append("-" * 80)
            report.append(f"บันทึกสำเร็จ:      {self.stats['db_saved']} รายการ")
            report.append(f"บันทึกไม่สำเร็จ:   {self.stats['db_failed']} รายการ")
            
            if self.stats['successful'] > 0:
                db_success_rate = (self.stats['db_saved'] / self.stats['successful']) * 100
                report.append(f"อัตราความสำเร็จ:    {db_success_rate:.2f}%")
            report.append("")
        
        if self.stats['by_province']:
            report.append("📍 การกระจายตัวตามจังหวัด")
            report.append("-" * 80)
            
            sorted_provinces = sorted(self.stats['by_province'].items(), 
                                     key=lambda x: x[1], reverse=True)
            
            for province, count in sorted_provinces:
                percentage = (count / self.stats['successful']) * 100 if self.stats['successful'] > 0 else 0
                bar = "█" * min(int(percentage / 2), 40)
                region = self.get_region(province)
                report.append(f"{province:20s} ({region:15s}) {count:3d} พัสดุ {bar} {percentage:5.1f}%")
            
            report.append("")
        
        if self.stats['by_region']:
            report.append("🗺️  สรุปตามภาค")
            report.append("-" * 80)
            
            for region in sorted(self.stats['by_region'].keys()):
                count = self.stats['by_region'][region]
                percentage = (count / self.stats['successful']) * 100 if self.stats['successful'] > 0 else 0
                bar = "█" * int(percentage / 2)
                report.append(f"{region:30s} {count:3d} พัสดุ {bar} {percentage:5.1f}%")
            
            report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_report(self, results: List[Dict], output_file: str = None):
        """บันทึกรายงาน"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"sorting_report_{timestamp}.txt"
        
        filepath = os.path.join(self.output_folder, output_file)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.generate_sorting_report(results))
        
        print(f"💾 บันทึกรายงานที่: {filepath}")
        return filepath


def main():
    """ฟังก์ชันหลัก (รองรับ Database)"""
    from os import getenv
    from dotenv import load_dotenv
    
    # โหลดค่าจากไฟล์ .env
    load_dotenv()
    
    # API Keys และ Configuration
    TYPHOON_API_KEY = getenv("TYPHOON_API_KEY")
    DB_API_URL = getenv("DB_API_URL", "http://localhost/test/data.php")  # URL ของ PHP API
    DB_API_KEY = getenv("DB_API_KEY", "your_secure_api_key_here")  # API Key สำหรับ authentication
    
    OUTPUT_FOLDER = "parcel_results"
    IMAGE_FOLDER = "parcel_images"
    
    # ตรวจสอบ API Key
    if not TYPHOON_API_KEY:
        print("❌ ไม่พบ TYPHOON_API_KEY")
        print("💡 วิธีตั้งค่า:")
        print("   export TYPHOON_API_KEY='your_typhoon_api_key'")
        return
    
    # แสดงการตั้งค่า
    print("="*60)
    print("🚀 ระบบคัดแยกพัสดุแบบครบวงจร (NLP + Database)")
    print("="*60)
    print(f"📁 Image Folder: {IMAGE_FOLDER}")
    print(f"📁 Output Folder: {OUTPUT_FOLDER}")
    print(f"🌐 Database API: {DB_API_URL}")
    print(f"🔑 Database API Key: {'*' * len(DB_API_KEY) if DB_API_KEY else 'Not Set'}")
    print("="*60)
    
    # ตรวจสอบโฟลเดอร์ภาพ
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ ไม่พบโฟลเดอร์: {IMAGE_FOLDER}")
        print(f"💡 กรุณาสร้างโฟลเดอร์และใส่ภาพพัสดุ")
        return
    
    # สร้างระบบพร้อมเชื่อมต่อ Database
    system = CompleteParcelSortingSystem(
        typhoon_api_key=TYPHOON_API_KEY,
        output_folder=OUTPUT_FOLDER,
        db_api_url=DB_API_URL,
        db_api_key=DB_API_KEY
    )
    
    # ประมวลผลพัสดุทั้งหมด (พร้อมบันทึกลง Database)
    results = system.batch_process(
        image_folder=IMAGE_FOLDER,
        enhance_images=True,
        save_to_db=True  # เปลี่ยนเป็น False ถ้าไม่ต้องการบันทึกลง Database
    )
    
    print("\n💾 กำลังบันทึกและจัดเก็บพัสดุ...")
    
    # บันทึก JSON และคัดลอกภาพ
    for result in results:
        if result["success"]:
            system.save_individual_json(result)
            system.copy_image_to_province_folder(result)
    
    # แสดงสถิติ Database
    print(f"\n📊 สถิติการบันทึก Database:")
    print(f"   ✅ สำเร็จ: {system.stats['db_saved']} รายการ")
    print(f"   ❌ ไม่สำเร็จ: {system.stats['db_failed']} รายการ")
    
    # บันทึกผลลัพธ์รวมและรายงาน
    system.save_batch_json(results)
    print("\n" + system.generate_sorting_report(results))
    system.save_report(results)
    
    print("\n✅ ประมวลผลเสร็จสมบูรณ์!")
    print(f"\n📂 ผลลัพธ์ทั้งหมดถูกบันทึกใน: {OUTPUT_FOLDER}/")
    print(f"📂 พัสดุแยกตามจังหวัด: {OUTPUT_FOLDER}/sorted_by_province/")


if __name__ == "__main__":
    main()
<?php
// data.php เอาไปใส่ใน filezila เลย
header('Content-Type: application/json; charset=utf-8');
require_once 'connect.php';


$headers = getallheaders();


$api_key = '';
if (isset($headers['Authorization'])) {
    $auth = $headers['Authorization'];
    if (preg_match('/Bearer\s+(.*)$/i', $auth, $matches)) {
        $api_key = $matches[1];
    }
}

if (empty($api_key)) {
    $api_key = $headers['X-API-Key'] ?? $headers['X-Api-Key'] ?? '';
}

// ตรวจสอบ API Key
define('VALID_API_KEY', 'ใส่ตรงนี้'); // เปลี่ยนตรงนี้ให้ตรงกับ .env

if ($api_key !== VALID_API_KEY) {
    http_response_code(401);
    echo json_encode([
        'success' => false,
        'message' => 'Invalid API Key',
        'debug' => [
            'received_key' => substr($api_key, 0, 10) . '...',
            'headers_found' => array_keys($headers)
        ]
    ]);
    exit;
}

// รับข้อมูล JSON
$input = file_get_contents('php://input');
$data = json_decode($input, true);

if (!$data) {
    http_response_code(400);
    echo json_encode([
        'success' => false,
        'message' => 'Invalid JSON data'
    ]);
    exit;
}

$action = $data['action'] ?? '';

try {
    switch ($action) {
        case 'create_parcel':
            $result = createParcel($conn, $data);
            echo json_encode($result);
            break;
            
        case 'get_parcel':
            $result = getParcel($conn, $data['tracking_number']);
            echo json_encode($result);
            break;
            
        default:
            http_response_code(400);
            echo json_encode([
                'success' => false,
                'message' => 'Unknown action'
            ]);
    }
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode([
        'success' => false,
        'message' => $e->getMessage()
    ]);
}

function createParcel($conn, $data) {
    $conn->begin_transaction();
    
    try {
        // 1. Insert Sender
        $sender = $data['sender'];
        $stmt = $conn->prepare("
            INSERT INTO bao_senders (name, phone, address_details, province) 
            VALUES (?, ?, ?, ?)
        ");
        $stmt->bind_param('ssss', 
            $sender['name'], 
            $sender['phone'], 
            $sender['address_details'], 
            $sender['province']
        );
        $stmt->execute();
        $sender_id = $conn->insert_id;
        
        // 2. Insert Recipient
        $recipient = $data['recipient'];
        $stmt = $conn->prepare("
            INSERT INTO bao_recipients (name, phone, address_details, province) 
            VALUES (?, ?, ?, ?)
        ");
        $stmt->bind_param('ssss', 
            $recipient['name'], 
            $recipient['phone'], 
            $recipient['address_details'], 
            $recipient['province']
        );
        $stmt->execute();
        $recipient_id = $conn->insert_id;
        
        // 3. Insert Parcel
        $parcel = $data['parcel'];
        $stmt = $conn->prepare("
            INSERT INTO bao_parcels (sender_id, recipient_id, status) 
            VALUES (?, ?, ?)
        ");
        $stmt->bind_param('iis', 
            $sender_id, 
            $recipient_id, 
            $parcel['status']
        );
        $stmt->execute();
        $parcel_id = $conn->insert_id;
        
        $conn->commit();
        
        return [
            'success' => true,
            'message' => 'Parcel created successfully',
            'parcel_id' => $parcel_id,
            'sender_id' => $sender_id,
            'recipient_id' => $recipient_id
        ];
        
    } catch (Exception $e) {
        $conn->rollback();
        throw $e;
    }
}

function getParcel($conn, $tracking_number) {
    // Implementation for getting parcel data
    return [
        'success' => false,
        'message' => 'Not implemented yet'
    ];
}
?>
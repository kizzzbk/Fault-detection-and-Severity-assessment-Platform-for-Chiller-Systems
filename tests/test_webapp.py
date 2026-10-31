from fastapi.testclient import TestClient

from webapp.server import app

client = TestClient(app)

def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}

def test_index_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")

def test_api_config_endpoint():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "sliders" in data
    assert "baseline" in data
    assert len(data["sliders"]) > 0

def test_simulate_validation_error():
    """Kiểm tra Pydantic validation: hours < 1.0 phải trả về HTTP 422 thay vì lỗi sập server 500."""
    # Theo định nghĩa SimRequest: hours phải có ge=1.0, le=5.0
    payload = {
        "hours": 0.5,  # Không hợp lệ vì nhỏ hơn 1.0
        "faults": {"fwc": 1},
    }
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 422

def test_simulate_baseline_normal():
    """Kiểm tra mô phỏng chế độ hoạt động bình thường (không cài lỗi)."""
    payload = {
        "hours": 1.0,
        "faults": {},
        "seed": 100,
    }
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scenario"][0] == "normal"

def test_simulate_happy_path_with_fault():
    """Kiểm tra logic chẩn đoán thực sự của AI khi bơm lỗi fwc."""
    payload = {
        "hours": 1.0,               # Chạy mô phỏng 1 giờ (nhẹ và nhanh nhất)
        "faults": {"fwc": 2},       # Chủ động bơm lỗi 'fwc' (Reduced Condenser Water Flow)
        "seed": 42,                 # Cố định seed để kết quả mô phỏng luôn lặp lại 100%
    }
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    diag = data["diagnosis"]

    # 1. KIỂM TRA LOGIC SỨC KHỎE: Máy có lỗi thì sức khỏe PHẢI sụt giảm
    assert diag["health_ratio"] < 1.0, "Bơm lỗi vào nhưng AI vẫn chấm máy khỏe 100% là sai logic!"
    assert diag["fault_seconds"] > 0, "Phải ghi nhận được thời gian máy gặp sự cố!"

    # 2. KIỂM TRA ĐỘ CHÍNH XÁC CỦA AI: AI có bắt đúng tên lỗi 'fwc' không?
    # Trong các phân đoạn mà AI chẩn đoán, phải xuất hiện lỗi 'fwc'
    detected_faults = [seg["fault"] for seg in diag["segments"]]
    assert "fwc" in detected_faults, f"Bơm lỗi fwc nhưng AI lại đoán thành: {detected_faults}"

    # 3. KIỂM TRA TÍNH TOÀN VẸN CỦA DỮ LIỆU CẢM BIẾN
    # 1 giờ mô phỏng với chu kỳ 10 giây/mẫu (DT_S=10) -> phải sinh ra đúng 360 dòng dữ liệu
    expected_rows = int(1.0 * 3600 / 10)
    assert len(data["rows"]) == expected_rows

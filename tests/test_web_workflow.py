from playwright.sync_api import Page, expect


BASE_URL = "http://127.0.0.1:8000"


def test_tc04_web_workflow(page: Page):

    # 1. Mở website
    page.goto(BASE_URL)

    expect(
        page.locator("#screen-sim")
    ).to_be_visible()

    # 2. Lựa chọn thời gian mô phỏng 1 giờ
    hours_button = page.locator(
        '#hours button[data-h="1"]'
    )

    hours_button.click()

    expect(hours_button).to_have_class("on")

    # 3. Lấy cấu hình FWC để xác định mức độ 2
    config_response = page.request.get(
        f"{BASE_URL}/api/config"
    )

    assert config_response.ok

    config = config_response.json()

    fwc_config = next(
        slider
        for slider in config["sliders"]
        if slider["fault"] == "fwc"
    )

    nominal = fwc_config["nominal"]
    level4 = fwc_config["levels"]["4"]

    target_value = nominal + 0.5 * (level4 - nominal)

    # 4. Điều chỉnh thanh trượt FWC
    page.locator("#slider-CDW_DT").evaluate(
        """(element, value) => {
            element.value = String(value);
            element.dispatchEvent(
                new Event("input", { bubbles: true })
            );
        }""",
        target_value,
    )

    expect(
        page.locator("#probe-CDW_DT .probe-tags")
    ).to_contain_text("SL2")

    # 5. Chạy mô phỏng và kiểm tra dữ liệu gửi đi
    with page.expect_request(
        lambda request:
            request.url.endswith("/api/simulate")
            and request.method == "POST"
    ) as request_info:

        page.locator("#btn-run").click()

    request = request_info.value
    payload = request.post_data_json

    assert payload["hours"] == 1
    assert payload["faults"]["fwc"] == 2

    # 6. Chờ màn hình báo cáo xuất hiện
    expect(
        page.locator("#screen-report")
    ).to_be_visible(timeout=60000)

    # 7. Kiểm tra các thành phần báo cáo
    expect(
        page.locator("#verdict")
    ).to_be_visible()

    expect(
        page.locator("#r-time")
    ).to_be_visible()

    expect(
        page.locator("#data-table tbody tr")
    ).to_have_count(360)

    # 8. Quay lại màn hình mô phỏng
    page.locator("#btn-back").click()

    expect(
        page.locator("#screen-sim")
    ).to_be_visible()

    expect(
        page.locator("#screen-report")
    ).to_be_hidden()
# ScrewTrack — đo độ xoay đai ốc ray

Công cụ dòng lệnh C++ (OpenCV) cho biết mỗi đai ốc ray đã xoay bao nhiêu độ giữa các lần đo, để phát hiện đai ốc lỏng
dần. Mỗi lần đo là một thư mục ảnh chụp từ trên xuống; ảnh cùng tên file ở các thư mục là cùng một vị trí. Thư mục đầu
tiên là mốc so sánh. Với mỗi ảnh có hai đai ốc: phía trên ray (`top`) và phía dưới ray (`bottom`).

Kết quả là góc xoay so với mốc, đã trừ góc ray, trong khoảng [-30°, 30°) (lục giác lặp lại mỗi 60°). Mục tiêu sai số
±1°, cảnh báo lỏng khi |góc| ≥ 3°. Đai ốc không rõ, gỉ, bị che hoặc không tìm thấy thì báo **không đo được** kèm lý do;
công cụ không bao giờ đoán góc.

Thiết kế: `docs/superpowers/specs/2026-11-07-nut-rotation-design.md`.

## Cần có

- macOS, CMake ≥ 3.16, trình biên dịch C++17.
- OpenCV 4.13 cài qua Homebrew (`brew install opencv`), có module `dnn`.
- Python 3.10+ chỉ để xuất model một lần (bước dưới).

## Xuất model ONNX (làm một lần)

C++ không đọc được `model/best.pt`, nên phải xuất sang `model/best.onnx` (file này bị `.gitignore`, mỗi bản clone phải
tự tạo). Trong thư mục `ScrewTrack/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r model/requirements.txt     # có ultralytics
.venv/bin/yolo export model=model/best.pt format=onnx imgsz=640
```

Lệnh cuối tạo `model/best.onnx`. Không sửa `model/best.pt`. Nếu thiếu `model/best.onnx`, chương trình và các test báo
`cannot load ONNX model ... (export it with: ...)`.

## Build và test

```bash
cmake -S . -B build && cmake --build build
ctest --test-dir build --output-on-failure      # khoảng 15 s
```

Các test cần `Data/img/` (ảnh mẫu) và `model/best.onnx`.

## Chạy

```bash
build/screwtrack model/best.onnx <ref_dir> <session_dir>...
```

- `<ref_dir>`: lần đo mốc, các ảnh `*.jpg` / `*.JPG`.
- `<session_dir>...`: một hoặc nhiều lần đo sau, ảnh cùng tên với mốc.
- In CSV ra màn hình (một dòng tiêu đề cho cả lần chạy). Lưu lại bằng `> ket_qua.csv`.
- Mã thoát: 0 bình thường; 1 lỗi khi chạy (ví dụ thiếu model); 2 sai tham số, đường dẫn không phải thư mục, hoặc thư
  mục mốc không có ảnh `*.jpg`.

Ví dụ chạy thử (so ảnh mẫu với chính nó, mọi góc đo được phải là 0.00):

```bash
build/screwtrack model/best.onnx Data/img Data/img
```

### Các cột CSV

| Cột | Ý nghĩa |
|---|---|
| `session` | Tên thư mục lần đo. |
| `image` | Tên file ảnh (giống tên trong thư mục mốc). |
| `side` | `top` (đai ốc phía trên ray trong ảnh) hoặc `bottom`. |
| `status` | `ok` (\|góc\| < 3°), `loose` (\|góc\| ≥ 3°) hoặc `unmeasurable` (không đo được). |
| `rotation_deg` | Góc xoay so với mốc, độ, 2 chữ số thập phân; dấu theo atan2 với trục y hướng xuống (dương = theo chiều kim đồng hồ trên ảnh). Trống khi không đo được. Trạng thái tính từ đúng giá trị in ra (2.996 in `3.00` và là `loose`). |
| `reason` | Trống khi đo được; ngược lại là lý do (bảng dưới). |

### Lý do không đo được (`reason`)

| Lý do | Ý nghĩa |
|---|---|
| `image missing` | Không có ảnh cùng tên trong lần đo, hoặc file rỗng/không đọc được. |
| `nut not found` | Model không tìm thấy đai ốc ở phía này (trong ảnh mốc hoặc ảnh lần đo). |
| `unclear: too small` | Khung đai ốc nhỏ hơn 150 px. |
| `unclear: low contrast` | Vùng đai ốc quá phẳng màu. |
| `unclear: blurry` | Vùng đai ốc bị mờ. |
| `unclear: no clear hex` | Không thấy rõ hình lục giác (gỉ, bẩn, bị che ngay trong ảnh mốc, hoặc ảnh lần đo không còn rõ). |
| `unclear: no rail line` | Không tìm được mép ray để bù góc đặt thiết bị. |
| `unclear: misaligned` | Không ghép được vùng quanh đai ốc giữa hai ảnh (ảnh khác chỗ, lệch quá xa). |
| `unclear: covered` | Góc lục giác đổi không đều quanh đai ốc: bỏ đi một cung 45° bất kỳ làm góc đổi hơn 0,5°. Thường do vật che một phần đai ốc (que, dải, sỏi, lá) trong một trong hai ảnh. |

Khi cả ảnh mốc lẫn ảnh lần đo đều lỗi, lý do của ảnh mốc được báo.

## Kiểm tra với nhãn

```bash
build/eval_labels model/best.onnx Data
```

So bộ phát hiện, bộ lọc chất lượng và góc lục giác với nhãn labelme trong `Data/js/` (khoảng 20 s).

## Rủi ro độ chính xác còn mở

- Mọi số liệu về góc xoay đến từ **xoay giả lập** (xoay vùng tròn quanh đai ốc trong ảnh). Chưa có ảnh chụp lại thật
  của cùng một đai ốc, nên độ nhiễu thực của thiết bị chưa đo được và ±1° **chưa được chứng minh**.
- Góc đo thường hơi thấp: xoay thật 3° đọc khoảng 2,5–3,2°, 4° đọc 3,3–3,7°, 10° đọc thấp 0,7–1,7°. Vì vậy cảnh báo 3°
  thực tế bật ở khoảng 3,5°.
- So với nhãn vẽ tay (đường đi qua khung của model): trung vị 0,98°, p90 3,2°, tệ nhất 4,8°. Không chứng minh được ±1°.
- Các ngưỡng được chỉnh trên chính 93 ảnh mẫu, không có tập kiểm tra riêng.
- Chỉ 122 trên 184 đai ốc có nhãn (66%) được đo; model bỏ sót 16% đai ốc.
- Vòng đo lệch khỏi tâm đai ốc trôi khoảng 0,25° cho mỗi độ xoay (tới 1,12° ở 4° trên đai ốc gỉ khi khung model lỏng).
- Ngưỡng z = 13 phụ thuộc kích thước vùng cắt (tức khoảng cách camera); đổi khoảng cách thì phải chỉnh lại.
- Kiểm tra che khuất (`unclear: covered`) chỉ được chỉnh trên **vật che giả lập** (dải thẳng tối/sáng phủ 5–40% khung
  từ bốn phía, que 25 px, hình lá) và xoay giả lập trên 122 đai ốc mẫu. Kết quả: giữ 94% lần xoay sạch 0–6° (96–98% ở
  0–4°); đai ốc bị che mà không xoay không còn ca nào đọc ≥ 3° (trước đó 1304/9086 ca), nhưng vẫn còn 7 ca đọc
  1,0–1,2°. Khoảng 2–6% lần xoay sạch bị báo nhầm `unclear: covered`, nhiều hơn khi góc lớn và khi vòng đo có sỏi không
  xoay theo. Chưa thử vật che thật (bùn loang, bóng đổ, lá thật); chưa kiểm tra trường hợp ảnh **mốc** bị che.
- Thiết bị đặt lệch góc được bù bằng góc ray: xoay cả ảnh ±1,5° và ±3° (24 ảnh mẫu) cho sai số tối đa 0,52°, không ca
  nào `misaligned`. Chưa thử xoay lớn hơn 3° hay xoay quanh điểm khác tâm ảnh.
- Các script dùng để chỉnh ngưỡng không nằm trong repo.

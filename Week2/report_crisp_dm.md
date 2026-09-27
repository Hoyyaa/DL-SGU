# Báo cáo đồ án: Dự đoán giá nhà Ames (House Prices)

> **Bản báo cáo đã có kết quả chạy.** Trước khi nộp, thay thông tin trong ngoặc vuông ở trang bìa và xác nhận lại bảng phân công theo công việc thực tế. Điểm Kaggle chỉ điền sau khi upload submission.

---

## Trang bìa

**TRƯỜNG:** [Tên trường]  
**KHOA / BỘ MÔN:** [Tên khoa/bộ môn]  
**HỌC PHẦN:** [Tên học phần]  
**GIẢNG VIÊN:** [Họ tên giảng viên]

# DỰ ĐOÁN GIÁ BÁN NHÀ TẠI AMES, IOWA
### So sánh mô hình học máy và MLP bằng PyTorch theo CRISP-DM

| STT | Họ tên thành viên | Mã sinh viên | Lớp |
|---:|---|---|---|
| 1 | [Điền họ tên] | [Điền MSSV] | [Điền lớp] |
| 2 | [Điền họ tên nếu có] | [Điền MSSV] | [Điền lớp] |
| 3 | [Điền họ tên nếu có] | [Điền MSSV] | [Điền lớp] |

**[Địa điểm], [tháng/năm]**

---

## Phân công công việc

> Thay nội dung mẫu dưới đây bằng phân công và xác nhận của nhóm. Không ghi thành viên đã làm phần việc mà người đó chưa thực hiện.

| Thành viên | Phần việc | Sản phẩm / bằng chứng |
|---|---|---|
| [Tên thành viên 1] | Đọc paper, xác định bài toán, mô tả nguồn dữ liệu và CRISP-DM | Phần 1–3 báo cáo, trích dẫn paper |
| [Tên thành viên 2] | Khám phá dữ liệu, tiền xử lý, mô hình scikit-learn | Notebook/script, bảng missing, CV |
| [Tên thành viên 3] | Xây MLP PyTorch, chạy và đánh giá | Mã MLP, checkpoint, kết quả OOF |
| [Tên thành viên 4 hoặc cả nhóm] | Kiểm tra submission, tổng hợp và hiệu đính báo cáo | Submission đã kiểm tra, bản báo cáo cuối |

## Tóm tắt

Bài toán dự đoán giá bán nhà là bài toán hồi quy có giám sát. Nguồn học thuật là bộ dữ liệu nhà ở Ames, Iowa được Dean De Cock giới thiệu; bài thực hành sử dụng phiên bản train/test của cuộc thi Kaggle House Prices. Quy trình tuân theo CRISP-DM, so sánh Ridge, ElasticNet, Gradient Boosting, Extra Trees và mạng MLP hồi quy bằng PyTorch trên cùng 5-fold out-of-fold validation. Nhãn giá được biến đổi `log1p`; RMSLE là chỉ số chính. Kết quả cuối cùng và mô hình dùng để tạo submission được lấy từ `model_comparison.csv` và log lần chạy được nộp kèm.

**Kết quả:** blend được chọn với OOF RMSLE 0,1121 và MAE 13.175,56 USD. Đây là điểm validation nội bộ, không phải điểm leaderboard Kaggle.

---

## 1. Giới thiệu và xác định bài toán

### 1.1 Cơ sở từ paper

Dean De Cock (2011) đề xuất bộ dữ liệu Ames Housing như một dự án hồi quy phong phú thay thế Boston Housing. Bản paper mô tả 2.930 giao dịch nhà ở dân dụng tại Ames, Iowa trong giai đoạn 2006–2010; các biến mô tả nhiều khía cạnh chất lượng, số lượng và diện tích của căn nhà. Bài báo cũng thảo luận ngoại lệ và khuyến nghị xem xét các căn có diện tích sinh hoạt trên 4.000 ft² vì một số giao dịch là bán một phần hoặc bất thường.

Paper cung cấp bối cảnh, nguồn dữ liệu và các vấn đề cần xem xét khi xây mô hình. Paper không yêu cầu dùng duy nhất một thuật toán. Trong bài này, yêu cầu bổ sung của học phần là xây dựng MLP bằng PyTorch và so sánh với các mô hình học máy khác.

### 1.2 Phát biểu bài toán

- **Đầu vào:** thuộc tính vật lý, chất lượng, vị trí và giao dịch của căn nhà.
- **Đầu ra:** giá bán `SalePrice` (USD).
- **Loại bài toán:** hồi quy có giám sát.
- **Mục tiêu thực nghiệm:** mô hình hóa trên log-price để giảm độ lệch do phân phối giá lệch phải, so sánh bằng RMSLE trên dự đoán ngoài fold.
- **Đối tượng sử dụng:** nhóm học tập và người nộp kết quả dự đoán cho cuộc thi Kaggle.

### 1.3 Phạm vi và giả định

Các tệp Kaggle trong thư mục có 1.460 dòng train, 1.459 dòng test; train có 79 biến dự báo cùng `Id` và `SalePrice`, còn test có 79 biến cùng `Id`. Đây là bộ chia của cuộc thi liên quan đến Ames Housing, **không phải** toàn bộ 2.930 bản ghi trong paper. Nhóm chỉ dùng nhãn train để học và validation; `SalePrice` của test không có sẵn.

Giả định dữ liệu train/test có cùng định nghĩa biến và cùng quy trình thu thập. Không diễn giải quan hệ dự báo như quan hệ nhân quả.

## 2. CRISP-DM 1 — Business Understanding

Mục tiêu kinh doanh/mô hình hóa là dự đoán giá nhà với sai số tương đối nhỏ và tạo tệp submission hợp lệ. Vì cuộc thi sử dụng RMSLE, sai số theo log-price là tiêu chí chọn mô hình chính. MAE theo USD, bias có dấu và sai số tuyệt đối lớn nhất được báo cáo bổ sung để diễn giải sai số bằng đơn vị giá thực tế.

Tiêu chí hoàn thành:

1. Có quy trình tái lập từ CSV đầu vào đến kết quả.
2. Có mô hình MLP PyTorch đúng bài toán hồi quy.
3. Có so sánh công bằng trên cùng cross-validation.
4. Tạo `submission.csv` đúng schema và chứa dự đoán hợp lệ cho mọi `Id` trong test.
5. Ghi lại hạn chế, lựa chọn xử lý outlier và điểm số validation; không tuyên bố điểm Kaggle khi chưa upload.

## 3. CRISP-DM 2 — Data Understanding

### 3.1 Nguồn và cấu trúc

Nguồn dữ liệu thực nghiệm là `train.csv` và `test.csv` đi cùng cuộc thi House Prices: Advanced Regression Techniques. `Id` là mã định danh, không dùng làm đặc trưng. Biến mục tiêu là `SalePrice`.

| Tập | Số dòng | Số cột | Nội dung |
|---|---:|---:|---|
| Train | 1.460 | 81 | 79 predictor + `Id` + `SalePrice` |
| Test | 1.459 | 80 | 79 predictor + `Id` |

### 3.2 Đặc điểm thống kê

Ở bộ train gốc, `SalePrice` nằm trong khoảng 34.900–755.000 USD; trung vị 163.000 USD, trung bình 180.921 USD. Phân phối lệch phải, có một số căn giá cao. Biến số và biến phân loại có missing values; các trường như `PoolQC`, `MiscFeature`, `Alley`, `Fence` có tỷ lệ thiếu lớn.

Mã nguồn tạo biểu đồ phân phối `SalePrice` và biểu đồ `GrLivArea` so với `SalePrice`; thống kê missing và danh sách căn diện tích lớn được in khi chạy. Hình được lưu ở `figures/eda_target_and_living_area.png`.

### 3.3 Outlier theo paper

Có bốn căn trong train Kaggle có `GrLivArea > 4.000 ft²`. Chúng là Id 524 (4.676 ft², 184.750 USD), 692 (4.316 ft², 755.000 USD), 1183 (4.476 ft², 745.000 USD) và 1299 (5.642 ft², 160.000 USD). Hai dòng diện tích lớn nhưng giá tương đối thấp đặc biệt ảnh hưởng đến sai số validation.

Theo khuyến nghị thảo luận trong paper, script loại các dòng train có `GrLivArea > 4.000` trước huấn luyện; toàn bộ test vẫn được giữ để dự đoán và submission đủ 1.459 dòng. Đây là lựa chọn có căn cứ từ paper nhưng cũng làm thay đổi quần thể huấn luyện. Cần nêu rõ lựa chọn này khi báo cáo và không khẳng định rằng paper chứng minh mọi căn lớn đều là dữ liệu lỗi. Phân tích độ nhạy giữa giữ và loại các dòng này là hướng cải tiến nếu còn thời gian.

## 4. CRISP-DM 3 — Data Preparation

1. Giữ `Id` riêng; loại `Id` khỏi đặc trưng.
2. Áp dụng quy tắc diện tích trên 4.000 ft² cho train theo mục 3.3; không xóa dòng test.
3. Dùng `log1p(SalePrice)` làm nhãn huấn luyện. Khi xuất giá USD, dùng `expm1` và chặn giá âm về 0.
4. Biến số: điền missing bằng median, sau đó StandardScaler.
5. Biến phân loại: điền missing bằng nhãn riêng `Missing`, rồi One-Hot Encoding; các category chỉ gặp ở test được bỏ qua một cách an toàn.
6. Đặt imputation/encoding/scaling bên trong pipeline được fit lại tại từng fold để tránh rò rỉ thống kê từ fold validation.
7. Dùng cùng 5 fold xáo trộn với seed cố định cho các mô hình so sánh.

Một số missing phân loại có thể biểu thị sự vắng mặt thật (ví dụ không có garage/basement). Gán thành `Missing` giúp mô hình phân biệt trường hợp này với category phổ biến; với một vài cột, mã hóa thành `None` theo ý nghĩa nghiệp vụ có thể là cải tiến tiếp theo.

## 5. CRISP-DM 4 — Modeling

### 5.1 Mô hình đối chứng

- **Ridge:** hồi quy tuyến tính có regularization L2, phù hợp với nhiều biến one-hot tương quan.
- **ElasticNet:** kết hợp regularization L1/L2.
- **Gradient Boosting (Huber):** mô hình cây tuần tự, có thể học phi tuyến; Huber loss bớt nhạy với residual lớn.
- **Extra Trees:** ensemble cây ngẫu nhiên, học tương tác và phi tuyến.

Tất cả mô hình dự đoán cùng `log1p(SalePrice)` và dùng chung quy trình cross-validation.

### 5.2 MLP hồi quy bằng PyTorch

Mạng gồm `input → Linear(128) → ReLU → Dropout(0.15) → Linear(64) → ReLU → Dropout(0.10) → Linear(32) → ReLU → Linear(1)`. Đầu ra là một giá trị liên tục, không dùng sigmoid.

- Loss: MSE trên log-price đã chuẩn hóa theo mean/std của phần fit.
- Optimizer: AdamW, learning rate 0,001, weight decay 0,0001.
- Batch size: 64; tối đa 300 epoch; early stopping patience 30.
- Reproducibility: seed 42, DataLoader generator seeded; CPU/GPU được chọn tự động.
- Mỗi outer fold chia tiếp phần train thành fit/validation nội bộ để chọn số epoch. Sau đó mô hình fold được fit lại trên toàn outer-train với epoch đã chọn rồi đánh giá trên outer-validation.
- Mô hình cuối dùng median số epoch đã chọn qua 5 fold, được fit trên toàn bộ phần train sau xử lý outlier.

### 5.3 Blend

Blend cố định, không fit trọng số trên validation: Ridge 0,30; ElasticNet 0,20; Gradient Boosting 0,18; Extra Trees 0,12; PyTorch MLP 0,20. Blend được đánh giá như một ứng viên riêng; mô hình cuối là ứng viên có OOF RMSLE thấp nhất.

## 6. CRISP-DM 5 — Evaluation

### 6.1 Thiết kế đánh giá

KFold 5 phần, shuffle bật, random state 42. Mỗi dòng OOF chỉ nhận dự đoán từ mô hình không fit trên fold chứa dòng đó. Tiền xử lý được fit lại trong từng fold. Validation phản ánh khả năng tổng quát hóa trong cùng phân phối train; đây vẫn chỉ là ước lượng nội bộ.

### 6.2 Chỉ số

- **RMSLE:** căn bậc hai trung bình bình phương chênh lệch `log1p(actual)` và `log1p(predicted)`; chỉ số chính, thấp hơn tốt hơn.
- **MAE (USD):** trung bình độ lớn sai số theo giá thực tế.
- **Bias (USD):** trung bình `prediction − actual`; dương là dự đoán cao hơn thực tế.
- **Maximum absolute error (USD):** sai số tuyệt đối lớn nhất; chỉ ra các trường hợp cực đoan.

### 6.3 Kết quả

> Điền lại từ `model_comparison.csv` sau lần chạy cuối. Không chép một điểm cũ nếu đã đổi code hoặc quy tắc outlier.

| Mô hình | OOF RMSLE | MAE (USD) | Bias (USD) | Max abs error (USD) |
|---|---:|---:|---:|---:|
| Ridge | 0,1153 | 13.685,52 | -719,16 | 140.793,05 |
| ElasticNet | 0,1142 | 13.488,83 | -708,81 | 139.261,91 |
| Gradient Boosting | 0,1313 | 16.014,35 | -1.584,79 | 175.500,97 |
| Extra Trees | 0,1362 | 16.920,59 | -3.097,93 | 227.836,24 |
| PyTorch MLP | 0,1197 | 14.492,68 | **+8,65** | 172.058,06 |
| Blend | **0,1121** | **13.175,56** | -1.221,17 | 149.738,14 |

**Mô hình được chọn:** blend có OOF RMSLE thấp nhất là **0,1121**. MLP riêng đạt RMSLE 0,1197; blend cải thiện 0,0076 so với MLP và 0,0021 so với ElasticNet trong lần chạy này.  
**Điểm Kaggle:** chưa có; cần nộp `submission.csv` để nhận điểm leaderboard.

### 6.4 Diễn giải và hạn chế

Việc báo cáo cả MAE, bias và max error tránh chỉ nhìn một metric. Cần xem hàng có residual lớn trong `oof_predictions.csv`, đối chiếu với scatter plot và ghi các trường hợp giá/dữ liệu bất thường. MLP là yêu cầu kiến trúc học sâu và được đánh giá riêng; không mặc định MLP tốt hơn mô hình cây trên bộ dữ liệu nhỏ này.

Kết quả có giới hạn: một lần seed/five-fold có sai số lấy mẫu; quy tắc loại diện tích lớn có ảnh hưởng; một số missing được xử lý chung theo kiểu dữ liệu; feature engineering và tuning còn hạn chế; chưa có nhãn test để đánh giá độc lập. Kaggle leaderboard score chưa biết cho đến khi upload submission.

## 7. CRISP-DM 6 — Deployment

Sau khi chọn ứng viên tốt nhất theo OOF RMSLE, script fit mô hình đó trên train đã xử lý và tạo dự đoán cho mọi dòng test. Submission có chính xác hai cột `Id`, `SalePrice`; giữ nguyên thứ tự Id trong `test.csv`. Các kiểm tra tự động xác nhận số dòng, thứ tự/duy nhất của Id, không có giá trị thiếu/không hữu hạn, và giá dự đoán không âm.

Tệp sinh ra:

- `submission.csv`: nộp Kaggle.
- `model_comparison.csv`: so sánh OOF.
- `oof_predictions.csv`: dự đoán OOF theo dòng, dùng phân tích sai số.
- `house_price_mlp.pt`: checkpoint PyTorch kèm kiến trúc và tham số target scaling.
- `house_price_preprocessor.joblib`: preprocessing đã fit.
- `figures/eda_target_and_living_area.png`: hình EDA.

## 8. Hướng dẫn tái lập

Trong thư mục dự án đặt `train.csv`, `test.csv`, `house_price.ipynb`, `house_price_train.py`, `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python house_price_train.py
```

Hoặc mở notebook và chạy các cell từ trên xuống; notebook gọi hàm `run_workflow` trong source Python, hiển thị biểu đồ inline và xem kết quả theo các mục CRISP-DM. Sau khi chạy:

1. Mở `model_comparison.csv`, điền bảng kết quả ở mục 6.3.
2. Kiểm tra `submission.csv` và các thông báo validation ở cuối log.
3. Lưu notebook có outputs hoặc lưu log chạy kèm source.
4. Upload `submission.csv` lên đúng cuộc thi và ghi leaderboard score thực tế nếu có.
5. Thay tất cả ô `[ ]` bằng thông tin/quan sát thật; xóa ghi chú hướng dẫn trước khi xuất PDF.

## 9. Kết luận

Bài toán phù hợp với hồi quy giá nhà trên dữ liệu Ames/Kaggle. Quy trình hiện thực đủ sáu giai đoạn CRISP-DM ở mức thực nghiệm, có tiền xử lý fold-safe, các mô hình đối chứng, MLP PyTorch, OOF evaluation và submission được kiểm tra. Kết luận mô hình tốt nhất phải dựa trên kết quả cuối của validation; không suy diễn rằng MLP hay blend nhất thiết thắng. Nghiên cứu tiếp theo nên so sánh có/không loại outlier, bổ sung feature engineering dựa trên mô tả dữ liệu, tune theo fold hoặc repeated CV, và xác nhận bằng leaderboard Kaggle.

## Tài liệu tham khảo

1. De Cock, D. (2011). “Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project.” *Journal of Statistics Education*, 19(3). https://jse.amstat.org/v19n3/decock.pdf
2. House Prices: Advanced Regression Techniques — competition/project reference and Rishabh Nimje notebook: https://risx3.github.io/house-prices/
3. Tài liệu bài giảng và sách do giảng viên cung cấp: ghi đầy đủ tác giả, tên tài liệu, phiên bản/năm và chương/trang thực sự được trích dẫn trước khi nộp.

1. Các bước triển khai

Trong bài lab này, chúng tôi triển khai pipeline NLP cơ bản trên tập dữ liệu văn bản lớn bằng Apache Spark MLlib. Pipeline gồm các bước sau:

1.1 Đọc dữ liệu

Dataset: c4-train.00000-of-01024-30K.json.gz (định dạng JSON nén gzip).
Đọc dữ liệu vào DataFrame của Spark.
Lấy mẫu 1000 bản ghi để chạy thử nhanh, tránh tốn nhiều tài nguyên.

1.2 Tokenization (tách từ)

Tokenizer tách câu thành các từ dựa trên khoảng trắng.
Ví dụ: "Spark is a unified analytics engine" → [Spark, is, a, unified, analytics, engine]
Lưu ý: Tokenizer đơn giản không loại bỏ dấu câu; RegexTokenizer có thể dùng để kiểm soát tốt hơn.

1.3 Loại bỏ Stop Words

StopWordsRemover loại bỏ các từ phổ biến không mang nhiều nghĩa như "the", "is", "and".
Giảm nhiễu và tập trung vào các từ quan trọng để vector hóa.

1.4 Vector hóa Word2Vec

Mỗi câu sau khi tách từ được chuyển thành vector đặc trưng bằng Word2Vec.
Word2Vec nắm bắt ngữ nghĩa của từ trong ngữ cảnh, khác với TF-IDF chỉ dựa trên tần suất.
Vector tạo ra là dense vector, đại diện cho toàn bộ câu.

Tham số Word2Vec:
  vector size: 100
  minimum word count: 1

1.5 Tạo Pipeline

Pipeline tích hợp tất cả các bước:
  Tokenizer → StopWordsRemover → Word2Vec
Pipeline fit trên DataFrame mẫu và transform để tạo vector đặc trưng cho từng câu.

1.6 Logistic Regression

Tạo nhãn giả:
Câu chứa từ "spark" → label 1
  Các câu khác → label 0
  Chia dữ liệu train/test = 80/20
Fit mô hình Logistic Regression với vector Word2Vec.
Đánh giá mô hình bằng accuracy sử dụng MulticlassClassificationEvaluator.

2. Cách chạy code và log kết quả (How to Run and Log Results)

Mở terminal, chuyển tới thư mục spark_labs:
cd spark_labs

Chạy ứng dụng bằng SBT:
sbt run

Lần chạy đầu tiên mất thời gian do compile và tải dependencies.
Kết quả console được lưu vào file:
results/lab17_pipeline_output.txt

Thử nghiệm các biến thể pipeline
Thay tokenizer, giảm số lượng features, thêm LogisticRegression hoặc Word2Vec.

3. Giải thích kết quả (Obtained Results)

Tokens: RegexTokenizer tách kỹ hơn, Basic Tokenizer tách đơn giản hơn.
TF-IDF Vectors:
  numFeatures = 20000 → vector chi tiết, đầy đủ thông tin.
  numFeatures = 1000 → vector ngắn hơn, mất một số thông tin.
Logistic Regression: accuracy ~0.5, nguyên nhân: dữ liệu ít (1000 bản ghi), label không cân bằng.
Word2Vec: tạo vector dense biểu diễn ngữ nghĩa, có thể cải thiện chất lượng phân loại khi dùng dữ liệu lớn hơn.

4. Khó khăn và giải pháp (Difficulties & Solutions)
Khó khăn ->	Giải pháp
Dữ liệu lớn, Spark nặng -> 	Giới hạn số bản ghi (limit(1000))
Accuracy thấp ->	Tăng dữ liệu, dùng Word2Vec embedding, tuning hyperparameters
Cảnh báo bộ nhớ	-> Giảm numFeatures, chạy trên driver với dataset nhỏ

1. Các bước triển khai

Trong bài lab này, chúng tôi triển khai pipeline NLP cơ bản trên tập dữ liệu văn bản bằng Apache Spark MLlib. Pipeline gồm các bước chính sau:

1.1 Đọc dữ liệu

  Dataset: c4-train.00000-of-01024-30K.json.gz (định dạng JSON nén gzip).
  Đọc dữ liệu vào DataFrame của Spark.
  Giới hạn dữ liệu mẫu (ví dụ: 1000 bản ghi) để chạy thử nhanh, tránh tốn tài nguyên.

1.2 Tokenization (tách từ)

  Tokenizer tách câu thành các từ dựa trên khoảng trắng.
  Ví dụ:
  “Spark is a unified analytics engine” → [spark, is, a, unified, analytics, engine].
  Có thể thay bằng RegexTokenizer để kiểm soát việc tách dấu câu.

1.3 Loại bỏ Stop Words

  Sử dụng StopWordsRemover loại bỏ các từ phổ biến như “the”, “is”, “and” nhằm giảm nhiễu.
  Giúp mô hình tập trung vào các từ có ý nghĩa ngữ nghĩa mạnh hơn.

1.4 Vector hóa bằng Word2Vec

  Dữ liệu sau khi tách từ được vector hóa bằng Word2Vec.
  Word2Vec giúp biểu diễn các từ dựa trên ngữ cảnh xuất hiện, nắm bắt được quan hệ ngữ nghĩa.
  Tham số:
    vectorSize = 100
    minCount = 1
  Vector đầu ra là dense vector biểu diễn toàn bộ câu.

1.5 Chuẩn hóa vector (Normalization)

  Sau khi tạo vector từ Word2Vec, thêm bước chuẩn hóa vector đặc trưng bằng Normalizer.
  Mục tiêu: chuẩn hóa độ dài vector (L2-norm = 1), giúp việc so sánh cosine similarity chính xác và công bằng hơn.
  Cụ thể:
  
  val normalizer = new Normalizer()
    .setInputCol("features")
    .setOutputCol("norm_features")
  
  Pipeline cập nhật:
  Tokenizer → StopWordsRemover → Word2Vec → Normalizer

1.6 Logistic Regression

  Tạo nhãn giả:
    Câu chứa từ “spark” → label = 1
    Câu khác → label = 0
  Chia dữ liệu train/test: 80/20
  Fit mô hình LogisticRegression để thử nghiệm khả năng phân loại cơ bản.
  Đánh giá mô hình bằng MulticlassClassificationEvaluator theo accuracy.

1.7 Demo tìm văn bản tương tự (Cosine Similarity)

  Sau khi pipeline chạy xong, chọn một văn bản ngẫu nhiên làm query.
  Tính độ tương đồng cosine giữa văn bản đó và toàn bộ dataset.
  Sắp xếp giảm dần theo similarity → hiển thị Top 10 văn bản giống nhất.
  Ví dụ đầu ra:
    1.0   Spark is a unified analytics engine for large-scale data processing.
    0.36  Data pipelines often use Spark for distributed data processing.
    0.18  Machine learning enables systems to learn from data and improve automatically.
    ...

2. Cách chạy code và log kết quả
2.1 Chạy trên Windows

  Mở terminal, chuyển đến thư mục dự án:
    cd D:\ScalaProjects\nlp\spark_labs
  
  Chạy chương trình bằng sbt:
    sbt run
  
  Sau khi chọn số thứ tự (2), chương trình sẽ:
    Đọc dữ liệu JSON.
    Chạy pipeline NLP.
    Tạo vector, huấn luyện Logistic Regression.
    In ra kết quả cosine similarity top 10.

2.2 Kết quả và log

Kết quả hiển thị trên console và được lưu tại:
    results/lab17_pipeline_output.txt
  Log chi tiết pipeline được lưu tại:
    log/lab17_metrics.log
  Spark UI có thể truy cập tại:
    http://localhost:4040

3. Giải thích kết quả

Tokenizer:	Tách từ cơ bản theo khoảng trắng -> Hoạt động tốt với tiếng Anh, nhưng chưa xử lý dấu câu
StopWordsRemover:	Loại bỏ từ không quan trọng -> Giảm nhiễu, giúp vector gọn hơn
Word2Vec:	Tạo vector dense chứa thông tin ngữ nghĩa -> Vector hóa tốt, phù hợp với cosine similarity
Normalizer:	Chuẩn hóa vector để so sánh cosine -> Giúp độ tương đồng chính xác hơn
Logistic Regression:	Accuracy ≈ 0.5 -> Do dữ liệu nhỏ, label giả định nên kết quả trung bình
Cosine Similarity:	Trả về 10 văn bản gần nhất -> Kết quả hợp lý: văn bản chứa “Spark” hoặc cùng chủ đề được xếp cao

4. Khó khăn và giải pháp

Dataset lớn Spark nặng -> Giới hạn limit(1000) khi đọc
Accuracy thấp -> Tăng dữ liệu thật, tuning hyperparameters
Bộ nhớ cảnh báo -> Giảm numFeatures hoặc dùng cache(false)
Output log rối, khó đọc -> Thêm Logger.getLogger("org").setLevel(Level.ERROR) để ẩn log Spark
So sánh cosine sai lệch -> Thêm Normalizer để chuẩn hóa vector trước khi tính similarity

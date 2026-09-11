# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Qua thực nghiệm với 4 mức temperature:
> - **T = 0.0:** Phản hồi tập trung vào sự thật văn hóa ẩm thực quen thuộc (cà phê trứng Hà Nội, thành phần lòng đỏ, sữa đặc), câu từ chuẩn xác, tất định và ngắn gọn.
> - **T = 0.5:** Chuyển sang chủ đề địa lý du lịch (đường bờ biển dài 3.260 km, các bãi biển Đà Nẵng, Nha Trang, Phú Quốc), cách diễn đạt mở rộng và mềm mại hơn.
> - **T = 1.0:** Chủ đề phong phú và góc nhìn sinh động hơn hẳn (vua trái cây nhiệt đới sầu riêng, mô tả mùi hương người thích kẻ sợ, cùng xoài, thanh long, vải...).
> - **T = 1.5:** Lựa chọn một sự thật kỳ vĩ và giàu tính khám phá (hang Sơn Đoòng dài 9 km, cao 200 m, chứa cả hệ sinh thái và rừng riêng trong hang).
> **Quy luật rút ra:** Temperature càng thấp thì mô hình càng ưu tiên các token có xác suất cao nhất, câu trả lời mang tính tất định, an toàn và khuôn mẫu. Ngược lại, temperature càng cao sẽ mở rộng phân phối xác suất, giúp câu trả lời đa dạng về chủ đề, giàu hình ảnh và mang tính sáng tạo cao hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature trong khoảng từ 0.0 đến 0.3. Lý do là chatbot hỗ trợ khách hàng đòi hỏi tính nhất quán, chính xác cao và bám sát thông tin chính sách/dữ liệu của doanh nghiệp, giúp giảm thiểu tối đa hiện tượng ảo giác (hallucination) và tránh các câu trả lời tùy hứng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Theo bảng giá (0.010 USD/1k token output cho GPT-4o so với 0.0006 USD/1k token output cho mini), GPT-4o đắt hơn khoảng 16.7 lần (với 30.000 lượt gọi x 350 token = 10.5M token, GPT-4o tốn ~$105/ngày còn mini chỉ tốn ~$6.3/ngày). GPT-4o xứng đáng khi cần giải quyết các bài toán lập luận logic phức tạp, viết code chuyên sâu hoặc phân tích hợp đồng pháp lý. Trong khi đó, mini phù hợp cho tác vụ phân loại ý định (intent classification), tóm tắt tin nhắn ngắn hoặc trả lời FAQ phổ biến.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Qua kết quả chạy thực tế, hai phản hồi khác biệt rõ rệt trên 3 khía cạnh:
> - **Hình ảnh ví dụ & Ẩn dụ:** Bản giáo viên hình tượng hóa blockchain thành *"cuốn sổ ghi chép mà ai cũng nhìn thấy, khi ghi thông tin mới mọi người cùng đồng ý và không thể bị xóa sửa"*. Bản tài chính bỏ hoàn toàn ẩn dụ đời thường và đi thẳng vào kiến trúc hệ thống.
> - **Từ vựng & Độ sâu:** Bản giáo viên dùng từ thuần Việt giản dị (*"cuốn sổ", "ghi chép", "máy tính khắp thế giới", "tin tưởng"*). Bản chuyên gia tài chính dùng hệ thống thuật ngữ kỹ thuật chuẩn mực (*"lưu trữ dữ liệu phân tán", "mã băm (hash)", "xác thực tính toàn vẹn", "mạng lưới các nút (nodes)"*).
> - **Cấu trúc trình bày:** Bản giáo viên viết dạng tự sự liền mạch, giải thích từng bước logic cho trẻ nhỏ; trong khi bản tài chính cấu trúc dạng tài liệu kỹ thuật có gạch đầu dòng phân tách rõ các thành phần (Khối / Block, Mạng lưới phân tán / Distributed Network).  
> **Kết luận:** System prompt đóng vai trò như một bộ định hình ngữ cảnh nhận thức (persona conditioning), giúp điều chỉnh linh hoạt vốn từ, độ sâu chuyên môn và văn phong mà không làm suy giảm tính chính xác cốt lõi của khái niệm.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt 100 từ, công thức ước lượng `100 / 0.75 ≈ 133 token`, nhưng thực tế `tiktoken` (bộ mã hóa BPE) đếm được khoảng 170 - 200 token (chênh lệch từ 25% đến 50%). Tiếng Việt tốn nhiều token hơn tiếng Anh cùng độ dài vì thuật toán phân tách từ vựng của LLM (BPE tokenizer) được huấn luyện chủ yếu trên kho ngữ liệu tiếng Anh; các từ tiếng Việt có dấu và từ ghép thường bị tách thành nhiều subword hoặc byte token riêng lẻ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng tương tác trực tiếp thời gian thực như chatbot dòng lệnh, giao diện hội thoại web/app vì nó giảm Time-to-First-Token (TTFT), giúp người dùng không phải chờ đợi lâu khi sinh câu trả lời dài. Ngược lại, non-streaming phù hợp hơn trong các tác vụ xử lý nền (batch job), phân loại dữ liệu, trích xuất thông tin có cấu trúc (JSON schema/function calling) hoặc khi cần kiểm duyệt (content moderation) toàn bộ câu trả lời trước khi gửi tới người dùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng dần khoảng thời gian chờ sau mỗi lần thử, tạo khoảng đệm thời gian đủ lớn để hệ thống server kịp phục hồi khi xảy ra sự cố quá tải hoặc rate limit. Nếu hàng nghìn client cùng retry với một khoảng delay cố định giống nhau (ví dụ đều chờ đúng 1 giây), toàn bộ các request này sẽ cùng lúc ập vào server theo từng nhịp sóng định kỳ (hiện tượng Thundering Herd), tiếp tục làm sập server và khiến tình trạng nghẽn thêm trầm trọng.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt: "Bạn là một trợ giảng AI tận tâm của khóa học AI Thực Chiến. Hãy giải thích các khái niệm kỹ thuật một cách trực quan, ngắn gọn bằng tiếng Việt và luôn kèm theo một ví dụ thực tế."  
``Giải thích:`` Cụm từ "ngắn gọn bằng tiếng Việt" nhằm tối ưu độ trễ (latency), tiết kiệm token và đảm bảo phản hồi thân thiện, đúng ngữ cảnh sinh viên Việt Nam.  
Cụm "luôn kèm theo một ví dụ thực tế" giúp cụ thể hóa các lý thuyết LLM trừu tượng thành các bài toán thực tiễn để học viên ghi nhớ lâu hơn.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là bộ nhớ ngữ cảnh bị cắt cứng sau 3 lượt hội thoại gần nhất (sliding window 6 messages), khiến trợ lý quên mất các thông tin quan trọng được người dùng chia sẻ từ các lượt trước đó. Đề xuất cải thiện: Triển khai cơ chế Tóm tắt bộ nhớ (Context Summarization). Cụ thể, khi lịch sử vượt quá 3 lượt, thay vì xóa hoàn toàn các lượt cũ, ta dùng một LLM nhỏ (như gpt-4o-mini) tóm tắt các lượt trước thành 1 đoạn ngắn cô đọng và lưu trữ trong system prompt hoặc đầu history làm ngữ cảnh nền cho cuộc trò chuyện.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026

"""Prompts and JSON schemas for every step.

The prompts are written in Vietnamese on purpose: tickets, documents and replies are Vietnamese, and the
reported results were produced with exactly these prompts (changing them invalidates the LLM cache).
"""

from __future__ import annotations

ROLE = ("Bạn là trợ lý AI hỗ trợ nhân viên chăm sóc khách hàng (CSKH) của chuỗi bán lẻ Điện máy Lam Sơn. "
        "Bạn chỉ được dựa vào tài liệu nội bộ được cung cấp, không dùng kiến thức bên ngoài, không bịa chính sách, số liệu, thời hạn hay chi phí.")


def format_evidence(hits) -> str:
    rows = []
    for h in hits:
        c = h.chunk
        status = "đang hiệu lực" if c.status == "active" else "ĐÃ HẾT HIỆU LỰC"
        rows.append(f"[{c.chunk_id}] (tài liệu: {c.title}; hiệu lực từ {c.effective_date}; {status})\nMục: {c.section_title}\n{c.text}")
    return "\n\n".join(rows) if rows else "(không có tài liệu)"


# --------------------------------------------------------------------------- #
# 1. Ticket analysis (Corrective RAG)
# --------------------------------------------------------------------------- #
ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "description": "loại yêu cầu: bảo hành, đổi trả, hoàn tiền, giao hàng, trả góp, thanh toán, thành viên, khuyến mãi, sản phẩm, lỗi kỹ thuật, đơn hàng, khác"},
        "product": {"type": ["string", "null"], "description": "sản phẩm/model khách nhắc tới, null nếu không có"},
        "entities": {"type": "array", "items": {"type": "string"}, "description": "thời gian, số tiền, địa điểm, mã lỗi... khách cung cấp"},
        "missing_info": {"type": "array", "items": {"type": "string"},
                         "description": "thông tin BẮT BUỘC còn thiếu khiến không thể xử lý; rỗng nếu đủ"},
        "search_queries": {"type": "array", "items": {"type": "string"}, "description": "2-3 truy vấn tìm kiếm ngắn, tiếng Việt có dấu"},
    },
    "required": ["intent", "product", "entities", "missing_info", "search_queries"],
}


def analyze_prompt(ticket: str):
    system = ROLE + " Nhiệm vụ: phân tích ticket của khách trước khi tìm kiếm tài liệu."
    user = f"""Ticket của khách:
\"\"\"{ticket}\"\"\"

Yêu cầu:
1. Xác định loại yêu cầu, sản phẩm và các thông tin khách đã cung cấp.
2. missing_info: chỉ liệt kê thông tin mà nếu thiếu thì KHÔNG THỂ trả lời đúng câu hỏi khách đang hỏi. Phân biệt hai loại yêu cầu:
   a) Hỏi về chính sách, quy trình, thời hạn, chi phí, điều kiện (ví dụ "bảo hành mất bao lâu", "có được đổi không", "phí lắp đặt bao nhiêu"):
      chỉ thiếu thông tin khi câu trả lời thay đổi theo thông tin đó mà khách chưa nói, thường là SẢN PHẨM/LOẠI SẢN PHẨM, khu vực,
      thời điểm mua, lý do đổi trả, mã lỗi. Nếu khách đã nêu loại sản phẩm hoặc tình huống đủ để tra chính sách thì missing_info = [].
      KHÔNG BAO GIỜ yêu cầu số điện thoại, mã đơn hàng, họ tên, địa chỉ, IMEI/serial cho loại yêu cầu này.
   b) Yêu cầu tra cứu hoặc thao tác trên MỘT đơn hàng/phiếu bảo hành cụ thể (tra trạng thái đơn, hủy đơn, đổi địa chỉ, kiểm tra máy còn bảo hành,
      hỏi phiếu bảo hành của mình bao giờ xong, xuất hóa đơn): khi đó mới cần mã đơn hàng, số điện thoại hoặc số phiếu nếu khách chưa cung cấp.
   Ví dụ (minh họa, không phải ticket thật): "Lò vi sóng mua 2 tuần bị chập điện, sửa có mất phí không?" -> [] (hỏi chính sách, đã rõ sản phẩm).
   "Shop cho hỏi đổi hàng mất bao nhiêu tiền?" -> ["sản phẩm", "lý do đổi"]. "Kiểm tra giúp đơn hàng của em đã xuất kho chưa" -> ["mã đơn hàng hoặc số điện thoại"].
3. search_queries: viết 2-3 truy vấn tìm kiếm ngắn gọn bằng thuật ngữ chính sách (ví dụ "thời gian xử lý bảo hành", "phí đổi trả đổi ý"),
   chuyển văn nói/không dấu thành tiếng Việt chuẩn có dấu."""
    return system, user, ANALYZE_SCHEMA


# --------------------------------------------------------------------------- #
# 2. Evidence grading (Corrective RAG)
# --------------------------------------------------------------------------- #
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "description": "với mỗi đoạn ĐANG HIỆU LỰC có nói về điểm khách hỏi: quy định cụ thể đoạn đó nêu (số ngày, số tháng, số tiền, điều kiện)",
                   "items": {"type": "object", "properties": {"chunk_id": {"type": "string"}, "statement": {"type": "string"}},
                             "required": ["chunk_id", "statement"]}},
        "verdict": {"type": "string", "enum": ["sufficient", "insufficient", "missing_info", "no_evidence", "conflict"]},
        "relevant_ids": {"type": "array", "items": {"type": "string"}, "description": "mã đoạn được chấp nhận làm bằng chứng"},
        "conflict_ids": {"type": "array", "items": {"type": "string"}, "description": "mã các đoạn mâu thuẫn nhau (nếu verdict=conflict)"},
        "missing_info": {"type": "array", "items": {"type": "string"}, "description": "câu hỏi cần hỏi lại khách (nếu verdict=missing_info)"},
        "rewritten_query": {"type": "string", "description": "truy vấn mới, khác các truy vấn đã dùng, nhắm vào tài liệu chính sách còn thiếu; luôn điền (dùng khi cần tìm lại)"},
        "explanation": {"type": "string"},
        "confidence": {"type": "number", "description": "0-1, độ tin cậy của đánh giá"},
    },
    "required": ["claims", "verdict", "relevant_ids", "conflict_ids", "missing_info", "rewritten_query", "explanation", "confidence"],
}


def grade_prompt(ticket: str, analysis: dict, hits, round_no: int, max_rounds: int, used_queries: list[str]):
    system = ROLE + " Nhiệm vụ: đánh giá chất lượng bằng chứng trước khi trả lời. Hãy nghiêm khắc."
    user = f"""Ticket của khách:
\"\"\"{ticket}\"\"\"

Phân tích sơ bộ: {analysis}
Vòng truy xuất: {round_no}/{max_rounds}. Các truy vấn đã dùng: {used_queries}

Tài liệu tìm được:
{format_evidence(hits)}

Bước 1 - claims: duyệt TẤT CẢ các đoạn ĐANG HIỆU LỰC ở trên (cả chính sách, FAQ, thể lệ khuyến mãi, trang sản phẩm, thông tin hãng);
với mỗi đoạn có đề cập đến điểm khách hỏi, ghi lại quy định cụ thể mà đoạn đó nêu (giữ nguyên con số, điều kiện).
Bước 2 - so sánh các claims với nhau, rồi chọn đúng MỘT verdict:
- conflict: có từ hai đoạn ĐANG HIỆU LỰC nêu giá trị KHÁC NHAU cho cùng một điểm áp dụng cho trường hợp của khách
  (ví dụ 7 ngày và 15 ngày, 12 tháng và 24 tháng, 10 km và 20 km), kể cả khi một đoạn là thể lệ khuyến mãi, trang sản phẩm, FAQ hay thông tin hãng,
  và không tài liệu nào nói rõ quy định nào được ưu tiên. Không tự chọn một bên. Ghi conflict_ids (các đoạn mâu thuẫn).
- sufficient: các đoạn đang hiệu lực trả lời đầy đủ, trực tiếp câu hỏi và KHÔNG mâu thuẫn nhau. Phải có ít nhất một claim nêu TRỰC TIẾP
  điều khách hỏi (khách hỏi thời gian thì phải có claim nêu thời gian, hỏi phí thì phải có claim nêu phí); nếu không có thì không được chọn sufficient. Liệt kê relevant_ids (chỉ các đoạn thực sự dùng để trả lời).
- insufficient: chủ đề thuộc phạm vi một cửa hàng điện máy (bảo hành, đổi trả, hoàn tiền, giao hàng, lắp đặt, trả góp, thanh toán, khuyến mãi, sản phẩm, lỗi kỹ thuật...)
  nhưng các đoạn hiện có chưa chứa quy định cần thiết. Viết rewritten_query KHÁC các truy vấn đã dùng, nhắm vào loại tài liệu chính sách/quy trình còn thiếu
  (ví dụ "các trường hợp từ chối bảo hành", "thời gian xử lý bảo hành"), không nhắc lại tên model.
- missing_info: câu trả lời phụ thuộc vào thông tin mà CHỈ KHÁCH mới cung cấp được (ví dụ chính sách khác nhau theo loại sản phẩm mà khách không nói sản phẩm nào,
  hoặc khách yêu cầu tra cứu/thao tác trên một đơn hàng cụ thể mà không có mã đơn). Ghi câu hỏi cần hỏi lại khách vào missing_info.
  Không dùng missing_info khi cái còn thiếu là quy định của cửa hàng (đó là insufficient), và không hỏi số điện thoại, mã đơn, họ tên khi khách chỉ hỏi chính sách.
- no_evidence: yêu cầu nằm ngoài những gì kho tài liệu bao quát (sản phẩm/hãng/dịch vụ cửa hàng không kinh doanh, chương trình không tồn tại, việc ngoài phạm vi CSKH),
  hoặc đã hết số vòng truy xuất mà vẫn không tìm thấy quy định.

Quy tắc:
- "Phân tích sơ bộ" chỉ để tham khảo; missing_info trong đó có thể sai. Nếu tài liệu đã trả lời được câu hỏi cho đúng tình huống khách mô tả
  (kể cả trả lời có điều kiện, ví dụ "nếu lỗi do nhà sản xuất thì..."), hãy chọn sufficient.
- Tài liệu ĐÃ HẾT HIỆU LỰC không được dùng làm bằng chứng và không tạo ra mâu thuẫn (chỉ là bản cũ).
- KHÔNG phải mâu thuẫn: các quy định áp dụng cho trường hợp khác nhau (khác loại sản phẩm, khác hãng, khác hình thức mua);
  quy định chung kèm ngoại lệ ("riêng tivi, tủ lạnh... thì ..."); các mốc nối tiếp nhau (quá 15 ngày được mượn máy, quá 30 ngày được đổi máy);
  các đoạn thuộc cùng một tài liệu. Mâu thuẫn thật luôn nằm ở hai tài liệu khác nhau nói khác nhau về CÙNG một trường hợp.
- Không suy diễn quy định không có trong tài liệu."""
    return system, user, GRADE_SCHEMA


# --------------------------------------------------------------------------- #
# 3. Answer generation (shared by all three architectures)
# --------------------------------------------------------------------------- #
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["answer", "clarify", "escalate"]},
        "reply": {"type": "string", "description": "phản hồi soạn sẵn gửi khách, xưng Lam Sơn, gọi khách là anh/chị"},
        "handling_steps": {"type": "array", "items": {"type": "string"}, "description": "hướng xử lý cho nhân viên CSKH"},
        "clarifying_questions": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "array", "items": {"type": "string"}, "description": "mã đoạn tài liệu làm căn cứ"},
        "confidence": {"type": "number", "description": "0-1"},
        "reason": {"type": "string"},
    },
    "required": ["decision", "reply", "handling_steps", "clarifying_questions", "citations", "confidence", "reason"],
}


def answer_prompt(ticket: str, hits, forced: bool = False):
    system = ROLE + " Nhiệm vụ: đề xuất hướng xử lý và soạn phản hồi cho ticket, có trích dẫn nguồn."
    rule = ("Bằng chứng dưới đây đã được kiểm duyệt là đủ để trả lời; hãy trả lời (decision=answer) chỉ bằng các đoạn này."
            if forced else
            "Nếu tài liệu không đủ để trả lời chắc chắn: decision=clarify khi cần khách bổ sung thông tin, "
            "decision=escalate khi tài liệu không có căn cứ hoặc mâu thuẫn; khi đó không đưa ra kết luận về chính sách.")
    user = f"""Ticket của khách:
\"\"\"{ticket}\"\"\"

Tài liệu nội bộ:
{format_evidence(hits)}

Yêu cầu:
- {rule}
- Không dùng tài liệu đã hết hiệu lực.
- Mỗi thông tin chính sách (thời hạn, chi phí, điều kiện) phải có trong tài liệu; ghi mã đoạn vào citations.
- reply: ngắn gọn, lịch sự, đúng trọng tâm; handling_steps: các bước nhân viên cần làm."""
    return system, user, ANSWER_SCHEMA


# --------------------------------------------------------------------------- #
# 4. LLM reranker (optional)
# --------------------------------------------------------------------------- #
RERANK_SCHEMA = {
    "type": "object",
    "properties": {"scores": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "score": {"type": "number"}},
                                                          "required": ["id", "score"]}}},
    "required": ["scores"],
}


def rerank_prompt(query: str, hits):
    system = "Bạn chấm mức độ liên quan của tài liệu với câu hỏi chăm sóc khách hàng."
    user = f"Câu hỏi: {query}\n\nTài liệu:\n{format_evidence(hits)}\n\nChấm điểm 0-10 cho từng mã đoạn (10 = trả lời trực tiếp câu hỏi)."
    return system, user, RERANK_SCHEMA


# --------------------------------------------------------------------------- #
# 5. LLM-as-judge (evaluation)
# --------------------------------------------------------------------------- #
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean", "description": "câu trả lời đúng với đáp án tham chiếu và đủ các ý bắt buộc"},
        "faithfulness": {"type": "number", "description": "tỉ lệ các khẳng định trong câu trả lời được tài liệu trích dẫn hỗ trợ, 0-1"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "citations_support": {"type": "boolean", "description": "các đoạn được trích dẫn có thực sự chứa căn cứ cho câu trả lời"},
    },
    "required": ["correct", "faithfulness", "unsupported_claims", "citations_support"],
}


def judge_prompt(ticket: str, reference: str, key_facts: list[str], answer: str, evidence_text: str):
    system = "Bạn là giám khảo đánh giá chất lượng trả lời của hệ thống RAG chăm sóc khách hàng. Chấm khách quan, nghiêm khắc."
    user = f"""Ticket: {ticket}

Đáp án tham chiếu: {reference}
Ý bắt buộc: {key_facts}

Câu trả lời của hệ thống:
\"\"\"{answer}\"\"\"

Các đoạn tài liệu hệ thống trích dẫn:
{evidence_text or '(không trích dẫn)'}

Đánh giá:
- correct: câu trả lời có đúng kết luận của đáp án tham chiếu và chứa các ý bắt buộc không (diễn đạt khác vẫn tính đúng).
- faithfulness: tách câu trả lời thành các khẳng định về chính sách/sự kiện; tính tỉ lệ khẳng định được các đoạn trích dẫn hỗ trợ.
- unsupported_claims: liệt kê khẳng định không có căn cứ trong đoạn trích dẫn.
- citations_support: các đoạn trích dẫn có chứa căn cứ cho kết luận chính không."""
    return system, user, JUDGE_SCHEMA

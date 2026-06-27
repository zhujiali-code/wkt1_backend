"""导游后端服务层。

提供服务模块：
- asr_service: 语音识别（ASR）服务，支持 DashScope 和 Mock 模式
- tts_service: 语音合成（TTS）服务，支持 DashScope 和 Mock 模式
- bailian_app_service: 百炼（阿里云 AI）应用调用服务
- vision_service: 视觉识别服务，基于多模态大模型输出纯视觉描述
- artifact_search_service: 文物检索服务，桥接视觉描述与百炼知识库检索
- photo_guide_service: 拍照导游服务，根据视觉描述和匹配结果生成导游讲解
- voice_qa_service: 语音问答服务，将 ASR→LLM→TTS 串联为完整链路
- ai_session_store: AI 语音问答会话状态和线程安全存储
"""

"""拍照导游服务模块。

根据视觉描述（VisualDescription）和知识库匹配结果（ArtifactMatchResult），
生成适合语音播报的导游讲解。

核心决策逻辑：
1. 匹配成功（is_matched=True）→ SPECIFIC_MODE（具体展品讲解）
2. 类别已知但无具体匹配 → CATEGORY_MODE（类别引导）
3. 无法识别 → RETAKE_MODE（提示重拍）

讲解文本通过百炼通用问答应用生成，不需要挂视觉知识库。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.artifact_search_service import ArtifactMatchResult
from services.bailian_app_service import FALLBACK_TEXT, BailianAppService
from services.vision_service import VisualDescription

# 导游讲解模式
SPECIFIC_MODE = "specific_explain"     # 具体展品讲解（匹配成功）
CATEGORY_MODE = "category_guide"       # 类别引导（无具体匹配但类别已知）
RETAKE_MODE = "retake_request"         # 提示重拍

# 各类别的讲解主题提示
CATEGORY_THEMES = {
    "玉器": "应国文化、身份、礼仪、审美",
    "陶瓷": "鲁山花瓷、郏县钧瓷、地方陶瓷文化",
    "青铜器": "古代礼制、贵族生活、应国文化",
    "石器": "早期生产生活、工具痕迹、材质和用途",
    "书画": "题材、笔墨、章法、地方文化记忆",
    "建筑构件": "建筑工艺、装饰寓意、空间礼制",
    "其他": "展览主题、参观路线、平顶山历史脉络",
    "无法判断": "平顶山博物馆展览主题",
}

# 各类别的本地降级讲解文案（LLM 不可用时使用）
LOCAL_CATEGORY_GUIDES = {
    "玉器": "这张照片更像玉器类展品。看玉器，可以先看颜色和温润感，再看造型是不是和身份、礼仪有关。平顶山一带的应国文化里，玉器常能帮助我们理解贵族审美和礼制。要不要再靠近拍一张细节？",
    "陶瓷": "这张照片更像陶瓷类展品。看陶瓷，可以先看器形，再看釉色、纹饰和口沿足底。平顶山周边有鲁山花瓷、郏县钧瓷等陶瓷文化线索，能看出地方工艺的变化。想不想继续听陶瓷怎么看？",
    "青铜器": "这张照片更像青铜器类展品。看青铜器，可以先看器形用途，再看纹饰和锈色。它常和古代礼制、贵族生活、应国文化有关，不一定要认出具体名字，也能读出身份和仪式感。",
    "石器": "这张照片更像石器类展品。看石器，可以注意材质、边缘磨损和形状用途。它们常指向早期生产生活，比如切割、打磨或祭祀场景。要不要换个角度拍清楚轮廓？",
    "书画": "这张照片更像书画类展品。看书画，可以先看题材，再看线条、墨色、留白和题跋印章。即使看不清具体作者，也能从画面气息和内容理解它想表达的文化趣味。",
    "建筑构件": "这张照片更像建筑构件。看这类展品，可以观察纹样、榫卯或装饰位置，想象它原来在建筑中的功能。它往往连接着工艺、礼制和审美。需要我继续讲建筑构件怎么看吗？",
}

# 提示重拍的默认文案
RETAKE_ANSWER = "这张照片信息不太够。请把展品放在画面中间，靠近一点，避开展柜反光后重拍。"


@dataclass(frozen=True)
class PhotoGuideResult:
    """拍照导游结果。

    Attributes:
        mode: 讲解模式（specific_explain/category_guide/retake_request）
        grounded: 回答是否基于知识库检索（True）还是降级文案（False）
        answer_text: 适合语音播报的导游回答文本
        gate_reason: 模式选择原因（便于调试）
        match_id: 匹配到的文物 ID（用于日志追踪）
        match_name: 匹配到的文物名称
    """
    mode: str
    grounded: bool
    answer_text: str
    gate_reason: str
    match_id: str = "none"
    match_name: str = "无"


class PhotoGuideService:
    """拍照导游服务。

    接收 VisualDescription 和 ArtifactMatchResult，
    根据匹配结果调用百炼通用问答应用生成讲解。

    Attributes:
        bailian_app_service: 百炼通用问答应用服务（不挂视觉知识库）
    """

    def __init__(self, bailian_app_service: BailianAppService | None = None):
        """初始化拍照导游服务。

        Args:
            bailian_app_service: 百炼通用问答应用实例，None 时只使用本地降级文案
        """
        self.bailian_app_service = bailian_app_service

    def build_answer(
        self,
        desc: VisualDescription,
        match: ArtifactMatchResult,
        *,
        device: str = "",
        image_id: str = "",
    ) -> PhotoGuideResult:
        """根据视觉描述和匹配结果构建导游回答。

        Args:
            desc: 视觉描述
            match: 知识库匹配结果
            device: 设备标识
            image_id: 图片 ID

        Returns:
            PhotoGuideResult: 导游讲解结果
        """
        mode, gate_reason = _choose_mode(desc, match)
        print(f"[GUIDE] 选中模式 mode={mode} gate_reason={gate_reason}", flush=True)

        if mode == RETAKE_MODE:
            return PhotoGuideResult(
                mode=mode, grounded=False, answer_text=RETAKE_ANSWER,
                gate_reason=gate_reason, match_id=match.match_id, match_name=match.match_name,
            )

        if mode == SPECIFIC_MODE:
            answer = self._ask_specific(desc, match, device=device, image_id=image_id)
            if _is_valid(answer):
                return PhotoGuideResult(
                    mode=mode, grounded=True, answer_text=_clean(answer),
                    gate_reason=gate_reason, match_id=match.match_id, match_name=match.match_name,
                )
            if self.bailian_app_service is None:
                return PhotoGuideResult(
                    mode=mode, grounded=False,
                    answer_text=f"这件展品很像{match.match_name}。你可以先关注它的材质、造型和纹饰，再结合展签确认具体名称。",
                    gate_reason="降级 本地候选讲解", match_id=match.match_id, match_name=match.match_name,
                )

        # 降级到类别引导
        answer = self._ask_category(desc, device=device, image_id=image_id)
        if _is_valid(answer):
            return PhotoGuideResult(
                mode=CATEGORY_MODE, grounded=True, answer_text=_clean(answer),
                gate_reason="候选回答不可用，降级到类别引导",
            )
        return PhotoGuideResult(
            mode=CATEGORY_MODE, grounded=False,
            answer_text=LOCAL_CATEGORY_GUIDES.get(desc.category, RETAKE_ANSWER),
            gate_reason="降级 本地类别讲解",
        )

    async def build_answer_async(
        self,
        desc: VisualDescription,
        match: ArtifactMatchResult,
        *,
        device: str = "",
        image_id: str = "",
    ) -> PhotoGuideResult:
        """异步构建导游回答。

        FastAPI 路由使用这个入口，百炼 HTTP 请求保持异步。
        """
        mode, gate_reason = _choose_mode(desc, match)
        print(f"[GUIDE] 选中模式 mode={mode} gate_reason={gate_reason}", flush=True)

        if mode == RETAKE_MODE:
            return PhotoGuideResult(
                mode=mode, grounded=False, answer_text=RETAKE_ANSWER,
                gate_reason=gate_reason, match_id=match.match_id, match_name=match.match_name,
            )

        if mode == SPECIFIC_MODE:
            answer = await self._ask_specific_async(desc, match, device=device, image_id=image_id)
            if _is_valid(answer):
                return PhotoGuideResult(
                    mode=mode, grounded=True, answer_text=_clean(answer),
                    gate_reason=gate_reason, match_id=match.match_id, match_name=match.match_name,
                )
            if self.bailian_app_service is None:
                return PhotoGuideResult(
                    mode=mode, grounded=False,
                    answer_text=f"这件展品很像{match.match_name}。你可以先关注它的材质、造型和纹饰，再结合展签确认具体名称。",
                    gate_reason="降级 本地候选讲解", match_id=match.match_id, match_name=match.match_name,
                )

        # 降级到类别引导
        answer = await self._ask_category_async(desc, device=device, image_id=image_id)
        if _is_valid(answer):
            return PhotoGuideResult(
                mode=CATEGORY_MODE, grounded=True, answer_text=_clean(answer),
                gate_reason="候选回答不可用，降级到类别引导",
            )
        return PhotoGuideResult(
            mode=CATEGORY_MODE, grounded=False,
            answer_text=LOCAL_CATEGORY_GUIDES.get(desc.category, RETAKE_ANSWER),
            gate_reason="降级 本地类别讲解",
        )

    # ---- 同步/异步 LLM 调用 ----

    def _ask_specific(self, desc: VisualDescription, match: ArtifactMatchResult, *, device: str, image_id: str) -> str:
        """调用 LLM 进行具体展品讲解。"""
        if self.bailian_app_service is None:
            return ""
        return self.bailian_app_service.ask(self._specific_prompt(desc, match))

    async def _ask_specific_async(self, desc: VisualDescription, match: ArtifactMatchResult, *, device: str, image_id: str) -> str:
        """异步调用 LLM 进行具体展品讲解。"""
        if self.bailian_app_service is None:
            return ""
        return await self.bailian_app_service.ask_async(self._specific_prompt(desc, match))

    def _specific_prompt(self, desc: VisualDescription, match: ArtifactMatchResult) -> str:
        """构建具体展品讲解 prompt。"""
        confidence_note = ""
        if match.confidence < 0.8:
            confidence_note = "匹配置信度中等，请使用'很像/可能是/建议以现场说明为准'这类保守措辞。"
        else:
            confidence_note = '可以说"很可能是"，但仍建议不要说成绝对确定。'

        return (
            f"请为游客讲解文物：{match.match_name}。\n"
            f"类别：{desc.category}。\n"
            f"匹配依据：{match.evidence or '视觉特征吻合'}。\n"
            f"{confidence_note}\n"
            "要求：\n"
            "- 用口语化中文回答，适合语音播报，80-150字\n"
            "- 包含鉴赏要点（材质、造型、纹饰等看点）\n"
            "- 不要编造知识库中没有的年代、出土地点、展柜位置\n"
            "- 不要在回答中说'根据知识库'、'检索结果显示'等技术用语\n"
            "- 不要Markdown，不要项目符号\n"
        )

    def _ask_category(self, desc: VisualDescription, *, device: str, image_id: str) -> str:
        """调用 LLM 进行类别引导讲解。"""
        if self.bailian_app_service is None:
            return ""
        return self.bailian_app_service.ask(self._category_prompt(desc))

    async def _ask_category_async(self, desc: VisualDescription, *, device: str, image_id: str) -> str:
        """异步调用 LLM 进行类别引导讲解。"""
        if self.bailian_app_service is None:
            return ""
        return await self.bailian_app_service.ask_async(self._category_prompt(desc))

    def _category_prompt(self, desc: VisualDescription) -> str:
        """构建类别引导 prompt。"""
        themes = CATEGORY_THEMES.get(desc.category, "平顶山博物馆展览主题")
        features = "、".join(desc.shape_features[:3] + desc.decoration_features[:3]) or "无"
        return (
            f"游客拍到的具体文物名称暂时无法确认，不能编造具体文物名称。\n"
            f'请围绕"{desc.category}"这类展品讲怎么看，并尽量结合相关主题：{themes}。\n'
            f"照片可见特征：{features}。\n"
            f"不确定因素：{desc.risk or '无'}。\n"
            "回答适合语音播报，50到120字，不要Markdown，不要项目符号，不要说识别失败。"
        )


def _choose_mode(desc: VisualDescription, match: ArtifactMatchResult) -> tuple[str, str]:
    """根据视觉描述和匹配结果选择讲解模式。

    决策规则：
    1. 匹配成功且置信度 >= 0.6 → SPECIFIC_MODE
    2. 类别已知 → CATEGORY_MODE
    3. 无法识别 → RETAKE_MODE

    Args:
        desc: 视觉描述
        match: 匹配结果

    Returns:
        tuple[str, str]: (讲解模式, 选择原因)
    """
    if match.is_matched:
        return SPECIFIC_MODE, f"知识库匹配成功 match_id={match.match_id} confidence={match.confidence:.2f}"
    if desc.category not in ("无法判断", "未知", ""):
        return CATEGORY_MODE, f"无匹配但类别已知 category={desc.category}"
    if not desc.is_clear:
        return RETAKE_MODE, "图片不清晰"
    return RETAKE_MODE, "无法识别"


def response_payload(
    *,
    device: str,
    image_id: str,
    desc: VisualDescription,
    match: ArtifactMatchResult,
    guide: PhotoGuideResult,
) -> dict[str, Any]:
    """构建 API 响应体 JSON。

    合并 VisualDescription、ArtifactMatchResult 和 PhotoGuideResult 的关键字段。

    Args:
        device: 设备标识
        image_id: 图片 ID
        desc: 视觉描述
        match: 匹配结果
        guide: 导游讲解结果

    Returns:
        dict: 完整的 API 响应字典
    """
    return {
        "ok": True,
        "device": device,
        "image_id": image_id,
        "mode": guide.mode,
        "category": desc.category,
        "match_id": match.match_id,
        "match_name": match.match_name,
        "confidence": match.confidence,
        "evidence": match.evidence,
        "visual_description": desc.visual_description,
        "shape_features": desc.shape_features,
        "decoration_features": desc.decoration_features,
        "color_material": desc.color_material,
        "search_keywords": desc.search_keywords,
        "is_clear": desc.is_clear,
        "risk": desc.risk,
        "need_retake": guide.mode == RETAKE_MODE,
        "answer_text": guide.answer_text,
        "grounded": guide.grounded,
        "gate_reason": guide.gate_reason,
    }


# ---- 内部辅助 ----

def _is_valid(answer: str) -> bool:
    """检查 LLM 回答是否有效。"""
    cleaned = _clean(answer)
    if not cleaned or cleaned == FALLBACK_TEXT:
        return False
    return "知识库无相关内容" not in cleaned


def _clean(answer: str) -> str:
    """清洗回答文本。"""
    return " ".join((answer or "").strip().split())

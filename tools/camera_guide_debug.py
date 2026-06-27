"""相机导游调试模块。

提供端到端的相机导游测试功能（新架构）：
1. 调用视觉服务分析图片 → VisualDescription
2. 调用文物检索服务在知识库中匹配 → ArtifactMatchResult
3. 调用拍照导游服务生成讲解 → PhotoGuideResult

新架构流程：拍照 → 视觉描述 → KB检索 → 匹配文物 → 问答生成讲解
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from core.paths import DEFAULT_CAMERA_TEST_IMAGE
from services.artifact_search_service import ArtifactMatchResult, ArtifactSearchService
from services.bailian_app_service import FALLBACK_TEXT, BailianAppService
from services.photo_guide_service import PhotoGuideResult, PhotoGuideService
from services.vision_service import VisionService

# 默认测试提问文本
DEFAULT_CAMERA_GUIDE_TEST_TEXT = "这是什么"
logger = logging.getLogger(__name__)


async def run_camera_guide_test(
    *,
    vision_service: VisionService,
    artifact_search: ArtifactSearchService,
    photo_guide_service: PhotoGuideService,
    test_image_path: Path = DEFAULT_CAMERA_TEST_IMAGE,
    user_text: str = DEFAULT_CAMERA_GUIDE_TEST_TEXT,
) -> dict[str, Any]:
    """运行一次完整的相机导游测试（新架构）。

    流程：
    1. 检查测试图片是否存在
    2. 视觉分析 → VisualDescription
    3. 知识库检索 → ArtifactMatchResult
    4. 导游讲解 → PhotoGuideResult
    5. 返回包含所有中间结果和耗时统计的字典

    Args:
        vision_service: 视觉服务实例
        artifact_search: 文物检索服务实例
        photo_guide_service: 拍照导游服务实例
        test_image_path: 测试图片路径
        user_text: 模拟用户提问

    Returns:
        dict: 结果字典，ok=True 表示成功
    """
    total_start = time.perf_counter()
    test_image_path = Path(test_image_path)

    # 检查图片是否存在
    if not test_image_path.exists():
        return _failure(
            stage="image_not_found",
            error_type="FileNotFoundError",
            error=f"测试图片不存在：{test_image_path}",
            test_image_path=test_image_path,
            total_start=total_start,
        )

    # 第 1 步：视觉描述
    vision_start = time.perf_counter()
    try:
        desc = await asyncio.to_thread(vision_service.analyze_image, test_image_path)
    except Exception as exc:
        return _failure(
            stage="vision",
            error_type=type(exc).__name__,
            error=str(exc),
            test_image_path=test_image_path,
            total_start=total_start,
        )
    vision_elapsed_ms = _elapsed_ms(vision_start)

    # 第 2 步：知识库检索
    search_start = time.perf_counter()
    try:
        match = await artifact_search.search_async(desc)
    except Exception as exc:
        return _failure(
            stage="search",
            error_type=type(exc).__name__,
            error=str(exc),
            test_image_path=test_image_path,
            total_start=total_start,
            extra={"vision_elapsed_ms": vision_elapsed_ms, "desc": desc.to_dict()},
        )
    search_elapsed_ms = _elapsed_ms(search_start)

    # 第 3 步：生成导游讲解
    guide_start = time.perf_counter()
    try:
        guide = await photo_guide_service.build_answer_async(
            desc, match, device="debug", image_id=test_image_path.stem,
        )
    except Exception as exc:
        return _failure(
            stage="guide",
            error_type=type(exc).__name__,
            error=str(exc),
            test_image_path=test_image_path,
            total_start=total_start,
        )
    guide_elapsed_ms = _elapsed_ms(guide_start)

    # 成功完成
    total_elapsed_ms = _elapsed_ms(total_start)
    result = {
        "ok": True,
        "test_image_path": str(test_image_path),
        "user_text": user_text,
        "visual_description": desc.to_dict(),
        "match_result": match.to_dict(),
        "guide_result": {
            "mode": guide.mode,
            "grounded": guide.grounded,
            "answer_text": guide.answer_text,
            "gate_reason": guide.gate_reason,
        },
        "timing": {
            "vision_elapsed_ms": vision_elapsed_ms,
            "search_elapsed_ms": search_elapsed_ms,
            "guide_elapsed_ms": guide_elapsed_ms,
            "total_elapsed_ms": total_elapsed_ms,
        },
    }
    _log_debug(result)
    return result


def _failure(
    *,
    stage: str,
    error_type: str,
    error: str,
    test_image_path: Path,
    total_start: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建失败结果字典。"""
    data: dict[str, Any] = {
        "ok": False,
        "stage": stage,
        "error_type": error_type,
        "error": error,
        "test_image_path": str(test_image_path),
        "timing": {"total_elapsed_ms": _elapsed_ms(total_start)},
    }
    if extra:
        data.update(extra)
    return data


def _log_debug(payload: dict[str, Any]) -> None:
    """输出相机导游调试日志（JSON 格式）。"""
    text = json.dumps(payload, ensure_ascii=False)
    logger.info("[CAMERA-GUIDE-DEBUG] %s", text)
    print(f"[CAMERA-GUIDE-DEBUG] {text}", flush=True)


def _elapsed_ms(start: float) -> int:
    """计算从 start 到现在的毫秒数。"""
    return int((time.perf_counter() - start) * 1000)

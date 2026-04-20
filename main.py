import base64
import io
import json
import math
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from PIL import Image, ImageStat

APP_VERSION = "3.0.0"
WS_PATH = "/ws/sinais"


# =========================================================
# APP / LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.started_at = time.time()
    app.state.service_name = "chartlens-railway"
    yield


app = FastAPI(
    title="ChartLens Railway",
    version=APP_VERSION,
    lifespan=lifespan,
)


# =========================================================
# MODELOS / UTIL
# =========================================================

@dataclass
class ParsedContext:
    asset: str = ""
    current_price_raw: str = ""
    current_price_value: Optional[float] = None
    candle_time_remaining_raw: str = ""
    candle_time_remaining_sec: Optional[int] = None
    chart_clock: str = ""
    timeframe: str = ""
    payout_percent: Optional[int] = None
    selected_operation_sec: Optional[int] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ImageAnalysis:
    image_ok: bool
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness_proxy: float
    bullish_pressure: float
    bearish_pressure: float
    neutral_pressure: float
    dominant_bias: str
    strength_score: float
    trend_score: float
    micro_score: float
    breakout_score: float
    fake_move_risk: float
    structural_weakness: float
    reversal_pressure: float
    continuation_health: float
    detected_pattern: str
    quality_label: str
    notes: list[str]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        text = str(value).strip().replace(",", ".")
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def safe_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def parse_mmss(raw: str) -> Optional[int]:
    raw = safe_str(raw)
    if not raw:
        return None

    parts = raw.split(":")
    if len(parts) != 2:
        return None

    mm = safe_int(parts[0])
    ss = safe_int(parts[1])
    if mm is None or ss is None:
        return None
    if mm < 0 or ss < 0 or ss > 59:
        return None

    total = mm * 60 + ss
    if total > 180:
        return None
    return total


def parse_clock(raw: str) -> str:
    raw = safe_str(raw)
    if not raw:
        return ""

    parts = raw.split(":")
    if len(parts) < 2:
        return ""

    hh = safe_int(parts[0])
    mm = safe_int(parts[1])
    ss = safe_int(parts[2]) if len(parts) >= 3 else 0

    if hh is None or mm is None or ss is None:
        return ""
    if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
        return ""

    if len(parts) >= 3:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{hh:02d}:{mm:02d}"


def normalize_timeframe(raw: str) -> str:
    raw = safe_str(raw).upper()
    mapping = {
        "1M": "M1",
        "M1": "M1",
        "ML": "M1",
        "2M": "M2",
        "M2": "M2",
        "3M": "M3",
        "M3": "M3",
        "5M": "M5",
        "M5": "M5",
        "10M": "M10",
        "M10": "M10",
        "15M": "M15",
        "M15": "M15",
    }
    return mapping.get(raw, raw)


def default_operation_sec_for_timeframe(tf: str) -> int:
    return {
        "M1": 60,
        "M2": 120,
        "M3": 180,
        "M5": 300,
        "M10": 600,
        "M15": 900,
    }.get(tf, 60)


def operation_label(sec: Optional[int]) -> str:
    sec = sec or 60
    if sec % 60 == 0:
        return f"{sec // 60} min"
    return f"{sec}s"


def is_forex_like(asset: str) -> bool:
    asset = safe_str(asset).upper()
    if not asset:
        return False
    tokens = asset.replace("-", "/").replace("_", "/").split("/")
    if len(tokens) >= 2:
        left = tokens[0]
        right = tokens[1]
        forex = {"EUR", "USD", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}
        return left in forex and right in forex
    return False


def price_reliable_for_asset(asset: str, value: Optional[float], raw: str) -> bool:
    if value is None:
        return False

    clean = safe_str(raw).replace(" ", "").replace(",", ".")
    if "." not in clean:
        return False

    parts = clean.split(".")
    if len(parts) != 2:
        return False

    int_digits = len(parts[0].replace("-", ""))
    dec_digits = len(parts[1])
    asset_up = asset.upper()

    if dec_digits < 2 or dec_digits > 6:
        return False

    if "BTC" in asset_up:
        return int_digits >= 4 and 1000.0 <= value <= 200000.0
    if "ETH" in asset_up:
        return int_digits >= 3 and 100.0 <= value <= 20000.0
    if "XRP" in asset_up or "ADA" in asset_up:
        return 0.05 <= value <= 10.0
    if "DOGE" in asset_up:
        return 0.01 <= value <= 5.0
    if "SOL" in asset_up:
        return 1.0 <= value <= 1000.0
    if is_forex_like(asset_up):
        return int_digits in (1, 2) and 0.2 <= value <= 3.5
    if not asset_up:
        return 0.01 <= value <= 100000.0
    return value > 0


def normalize_outcome_action(action: str) -> str:
    value = safe_str(action).upper()
    if not value:
        return "WAIT"
    return value


# =========================================================
# CONTEXTO RECEBIDO DO APP
# =========================================================

def extract_context(payload: Dict[str, Any]) -> ParsedContext:
    warnings: list[str] = []

    asset = safe_str(
        payload.get("asset")
        or payload.get("ativo")
        or payload.get("symbol")
    ).upper()

    current_price_raw = safe_str(
        payload.get("currentPrice")
        or payload.get("price")
        or payload.get("preco")
    )
    current_price_value = safe_float(current_price_raw)

    candle_time_remaining_raw = safe_str(
        payload.get("candleTimeRemaining")
        or payload.get("candle_time_remaining")
        or payload.get("tempoVela")
    )
    candle_time_remaining_sec = parse_mmss(candle_time_remaining_raw)

    chart_clock = parse_clock(
        payload.get("chartClock")
        or payload.get("clock")
        or payload.get("hora")
    )

    timeframe = normalize_timeframe(
        payload.get("timeframe")
        or payload.get("tf")
        or payload.get("period")
    )

    payout_percent = safe_int(
        payload.get("payoutPercent")
        or payload.get("payout")
    )
    if payout_percent is not None:
        payout_percent = int(clamp(payout_percent, 0, 100))

    selected_operation_sec = safe_int(
        payload.get("selectedOperationSec")
        or payload.get("operationSec")
        or payload.get("duracao")
    )
    if selected_operation_sec is None:
        selected_operation_sec = default_operation_sec_for_timeframe(timeframe)

    if not asset:
        warnings.append("asset_missing")

    if current_price_raw and not price_reliable_for_asset(asset, current_price_value, current_price_raw):
        warnings.append("price_unreliable")

    if candle_time_remaining_raw and candle_time_remaining_sec is None:
        warnings.append("candle_time_invalid")

    if chart_clock == "":
        warnings.append("clock_missing")

    if timeframe == "":
        warnings.append("timeframe_missing")

    if payout_percent is None:
        warnings.append("payout_missing")

    return ParsedContext(
        asset=asset,
        current_price_raw=current_price_raw,
        current_price_value=current_price_value,
        candle_time_remaining_raw=candle_time_remaining_raw,
        candle_time_remaining_sec=candle_time_remaining_sec,
        chart_clock=chart_clock,
        timeframe=timeframe,
        payout_percent=payout_percent,
        selected_operation_sec=selected_operation_sec,
        warnings=warnings,
    )


# =========================================================
# IMAGEM
# =========================================================

def decode_image_from_payload(payload: Dict[str, Any]) -> Optional[Image.Image]:
    raw_b64 = (
        payload.get("imageBase64")
        or payload.get("image_base64")
        or payload.get("image")
        or payload.get("frame")
    )
    if not raw_b64:
        return None

    try:
        b64 = str(raw_b64)
        if "," in b64 and "base64" in b64[:64]:
            b64 = b64.split(",", 1)[1]
        binary = base64.b64decode(b64)
        image = Image.open(io.BytesIO(binary)).convert("RGB")
        return image
    except Exception:
        return None


def crop_box(img: Image.Image, left: float, top: float, width: float, height: float) -> Image.Image:
    w, h = img.size
    x1 = max(0, min(w - 1, int(w * left)))
    y1 = max(0, min(h - 1, int(h * top)))
    x2 = max(x1 + 1, min(w, int(w * (left + width))))
    y2 = max(y1 + 1, min(h, int(h * (top + height))))
    return img.crop((x1, y1, x2, y2))


def image_contrast(img: Image.Image) -> float:
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    if not stat.stddev:
        return 0.0
    return float(stat.stddev[0])


def image_brightness(img: Image.Image) -> float:
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    if not stat.mean:
        return 0.0
    return float(stat.mean[0])


def sharpness_proxy(img: Image.Image) -> float:
    gray = img.convert("L").resize((120, 80))
    pixels = list(gray.getdata())
    width, height = gray.size
    diffs = []

    for y in range(height - 1):
        for x in range(width - 1):
            i = y * width + x
            diffs.append(abs(pixels[i] - pixels[i + 1]))
            diffs.append(abs(pixels[i] - pixels[i + width]))

    if not diffs:
        return 0.0
    return float(sum(diffs) / len(diffs))


def classify_pixel(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    if g > r + 18 and g > b + 8:
        return "green"
    if r > g + 18 and r > b + 8:
        return "red"
    return "neutral"


def analyze_color_pressures(img: Image.Image) -> Tuple[float, float, float]:
    small = img.resize((160, 90))
    data = list(small.getdata())

    green = 0
    red = 0
    neutral = 0

    for px in data:
        label = classify_pixel(px)
        if label == "green":
            green += 1
        elif label == "red":
            red += 1
        else:
            neutral += 1

    total = max(1, len(data))
    return green / total, red / total, neutral / total


def analyze_vertical_flow(img: Image.Image) -> Tuple[float, float, float]:
    chart = crop_box(img, 0.08, 0.18, 0.76, 0.56).resize((96, 48))
    width, height = chart.size
    data = chart.load()

    left_green = left_red = right_green = right_red = 0

    for x in range(width):
        for y in range(height):
            label = classify_pixel(data[x, y])
            if x < width // 2:
                if label == "green":
                    left_green += 1
                elif label == "red":
                    left_red += 1
            else:
                if label == "green":
                    right_green += 1
                elif label == "red":
                    right_red += 1

    total_left = max(1, left_green + left_red)
    total_right = max(1, right_green + right_red)

    macro = (right_green - right_red) / total_right
    micro = ((right_green + left_green) - (right_red + left_red)) / max(1, total_left + total_right)
    trend = ((right_green - left_green) - (right_red - left_red)) / max(1, total_left + total_right)

    return macro, micro, trend


def detect_pattern(
    dominant_bias: str,
    breakout_score: float,
    fake_move_risk: float,
    reversal_pressure: float,
    continuation_health: float,
    structural_weakness: float,
) -> str:
    if breakout_score >= 0.72 and continuation_health >= 0.58 and fake_move_risk < 0.45:
        return "bullish_breakout_continuation" if dominant_bias == "bullish" else "bearish_breakout_continuation"
    if continuation_health >= 0.62 and structural_weakness < 0.42:
        return "bullish_continuation" if dominant_bias == "bullish" else "bearish_continuation"
    if reversal_pressure >= 0.64:
        return "possible_bearish_reversal" if dominant_bias == "bullish" else "possible_bullish_reversal"
    if structural_weakness >= 0.62:
        return "consolidation"
    return "unclear_pattern"


def analyze_image(image: Optional[Image.Image]) -> ImageAnalysis:
    if image is None:
        return ImageAnalysis(
            image_ok=False,
            width=0,
            height=0,
            brightness=0.0,
            contrast=0.0,
            sharpness_proxy=0.0,
            bullish_pressure=0.0,
            bearish_pressure=0.0,
            neutral_pressure=1.0,
            dominant_bias="neutral",
            strength_score=0.0,
            trend_score=0.0,
            micro_score=0.0,
            breakout_score=0.0,
            fake_move_risk=0.75,
            structural_weakness=0.75,
            reversal_pressure=0.50,
            continuation_health=0.20,
            detected_pattern="unclear_pattern",
            quality_label="no_image",
            notes=["image_missing"],
        )

    width, height = image.size
    brightness = image_brightness(image)
    contrast = image_contrast(image)
    sharpness = sharpness_proxy(image)

    bull, bear, neutral = analyze_color_pressures(image)
    macro, micro, trend = analyze_vertical_flow(image)

    raw_bias = bull - bear
    if raw_bias > 0.03:
        dominant_bias = "bullish"
    elif raw_bias < -0.03:
        dominant_bias = "bearish"
    else:
        dominant_bias = "neutral"

    strength_score = clamp(abs(raw_bias) * 2.2 + abs(macro) * 0.55, 0.0, 1.0)
    trend_score = clamp((macro + trend + 1.2) / 2.4, 0.0, 1.0)
    micro_score = clamp((micro + 1.0) / 2.0, 0.0, 1.0)

    breakout_score = clamp((abs(macro) * 0.42) + (abs(trend) * 0.28) + (contrast / 64.0) * 0.30, 0.0, 1.0)
    structural_weakness = clamp(
        (0.42 if contrast < 24 else 0.18)
        + (0.24 if sharpness < 10 else 0.08)
        + (neutral * 0.22),
        0.0,
        1.0,
    )
    fake_move_risk = clamp(
        (0.50 if abs(raw_bias) < 0.06 else 0.14)
        + neutral * 0.20
        + (0.18 if abs(trend) < 0.06 else 0.04),
        0.0,
        1.0,
    )
    reversal_pressure = clamp(
        (0.24 if dominant_bias == "neutral" else 0.08)
        + (0.28 if abs(macro) > 0.28 and abs(trend) < 0.06 else 0.0)
        + neutral * 0.16
        + structural_weakness * 0.18,
        0.0,
        1.0,
    )
    continuation_health = clamp(
        strength_score * 0.38
        + (1.0 - structural_weakness) * 0.24
        + (1.0 - fake_move_risk) * 0.20
        + (1.0 - reversal_pressure) * 0.18,
        0.0,
        1.0,
    )

    detected_pattern = detect_pattern(
        dominant_bias="bullish" if raw_bias >= 0 else "bearish",
        breakout_score=breakout_score,
        fake_move_risk=fake_move_risk,
        reversal_pressure=reversal_pressure,
        continuation_health=continuation_health,
        structural_weakness=structural_weakness,
    )

    notes: list[str] = []
    if contrast < 20:
        notes.append("low_contrast")
    if sharpness < 10:
        notes.append("low_sharpness")
    if neutral > 0.84:
        notes.append("high_neutrality")

    quality_label = "good"
    if contrast < 18 or sharpness < 8:
        quality_label = "weak"
    if width < 200 or height < 200:
        quality_label = "small"

    return ImageAnalysis(
        image_ok=True,
        width=width,
        height=height,
        brightness=brightness,
        contrast=contrast,
        sharpness_proxy=sharpness,
        bullish_pressure=bull,
        bearish_pressure=bear,
        neutral_pressure=neutral,
        dominant_bias=dominant_bias,
        strength_score=strength_score,
        trend_score=trend_score,
        micro_score=micro_score,
        breakout_score=breakout_score,
        fake_move_risk=fake_move_risk,
        structural_weakness=structural_weakness,
        reversal_pressure=reversal_pressure,
        continuation_health=continuation_health,
        detected_pattern=detected_pattern,
        quality_label=quality_label,
        notes=notes,
    )


# =========================================================
# LÓGICA DE TIMING / DECISÃO
# =========================================================

def infer_timing(candle_sec: Optional[int], selected_operation_sec: Optional[int]) -> Dict[str, Any]:
    if selected_operation_sec is None or selected_operation_sec <= 0:
        selected_operation_sec = 60

    label = operation_label(selected_operation_sec)

    if candle_sec is None:
        return {
            "timingQuality": "unknown",
            "entryTiming": "tempo_da_vela_indisponivel",
            "secondsToAction": -1,
            "waitCandles": 0,
            "recommendedOperationSec": selected_operation_sec,
            "recommendedOperationLabel": label,
            "idealEntryType": "unknown",
        }

    if candle_sec >= 42:
        return {
            "timingQuality": "early",
            "entryTiming": "cedo",
            "secondsToAction": max(1, candle_sec - 28),
            "waitCandles": 0,
            "recommendedOperationSec": selected_operation_sec,
            "recommendedOperationLabel": label,
            "idealEntryType": "wait_for_setup",
        }

    if 24 <= candle_sec < 42:
        return {
            "timingQuality": "developing",
            "entryTiming": "janela_em_formacao",
            "secondsToAction": max(1, candle_sec - 12),
            "waitCandles": 0,
            "recommendedOperationSec": selected_operation_sec,
            "recommendedOperationLabel": label,
            "idealEntryType": "setup_building",
        }

    if 8 <= candle_sec < 24:
        return {
            "timingQuality": "ideal_window",
            "entryTiming": "janela_ideal",
            "secondsToAction": 0,
            "waitCandles": 0,
            "recommendedOperationSec": selected_operation_sec,
            "recommendedOperationLabel": label,
            "idealEntryType": "immediate_or_sniper",
        }

    return {
        "timingQuality": "late",
        "entryTiming": "entrada_tardia",
        "secondsToAction": -1,
        "waitCandles": 1,
        "recommendedOperationSec": selected_operation_sec,
        "recommendedOperationLabel": label,
        "idealEntryType": "next_candle",
    }


def confidence_from_context(ctx: ParsedContext, img: ImageAnalysis) -> int:
    confidence = 48

    if img.image_ok:
        confidence += int(img.strength_score * 22)
        confidence += int((1.0 - img.structural_weakness) * 10)
        confidence += int((1.0 - img.fake_move_risk) * 8)

    if ctx.asset:
        confidence += 3
    if ctx.current_price_raw and price_reliable_for_asset(ctx.asset, ctx.current_price_value, ctx.current_price_raw):
        confidence += 5
    if ctx.chart_clock:
        confidence += 2
    if ctx.timeframe:
        confidence += 2
    if ctx.candle_time_remaining_sec is not None:
        confidence += 3
    if ctx.payout_percent is not None:
        confidence += 2

    if "price_unreliable" in ctx.warnings:
        confidence -= 8
    if "candle_time_invalid" in ctx.warnings:
        confidence -= 6
    if img.quality_label == "weak":
        confidence -= 8
    if img.dominant_bias == "neutral":
        confidence -= 6

    return int(clamp(confidence, 20, 95))


def risk_from_context(ctx: ParsedContext, img: ImageAnalysis) -> str:
    score = 0.0
    score += img.fake_move_risk * 0.34
    score += img.structural_weakness * 0.28
    score += img.reversal_pressure * 0.22
    if "price_unreliable" in ctx.warnings:
        score += 0.18
    if ctx.payout_percent is not None and ctx.payout_percent < 70:
        score += 0.12

    if score >= 0.68:
        return "high"
    if score >= 0.42:
        return "medium"
    return "low"


def infer_action(ctx: ParsedContext, img: ImageAnalysis, timing: Dict[str, Any], confidence: int, risk: str) -> Dict[str, Any]:
    payout = ctx.payout_percent or 0
    timing_quality = timing["timingQuality"]
    bias = img.dominant_bias
    continuation_prob = int(clamp(img.continuation_health * 100, 0, 100))
    reversal_prob = int(clamp(img.reversal_pressure * 100, 0, 100))

    if not ctx.asset or not ctx.timeframe or not ctx.current_price_raw:
        action = "DATA_INSUFFICIENT"
        instruction = "Dados insuficientes para entrada"
        trigger = "Confirme ativo, preço e timeframe antes de operar"
        invalidation = "Se continuar sem contexto confiável"
        reason = "contexto_minimo_insuficiente"
        main_text = "Ainda faltam dados essenciais do trade."
    elif payout < 70:
        action = "NO_ENTRY_LOW_PAYOUT"
        instruction = "Não entrar: payout baixo"
        trigger = "Espere payout melhor ou setup muito superior"
        invalidation = "Se payout continuar baixo"
        reason = "payout_baixo"
        main_text = "Payout baixo para operação curta."
    elif risk == "high" and confidence < 72:
        action = "WAIT_RISK"
        instruction = "Risco alto, aguardar"
        trigger = "Aguardar cenário mais limpo"
        invalidation = "Se o risco continuar alto"
        reason = "risco_elevado"
        main_text = "Risco acima do ideal para entrada limpa."
    elif timing_quality == "late":
        action = "WAIT_ONE_MORE_CANDLE"
        instruction = "Aguardar mais 1 vela"
        trigger = "Reavaliar na próxima vela"
        invalidation = "Se a próxima vela também vier tardia"
        reason = "janela_tardia"
        main_text = "A janela desta vela ficou tardia."
    elif img.fake_move_risk >= 0.66:
        action = "WAIT_FAKE_MOVE"
        instruction = "Movimento suspeito, aguardar"
        trigger = "Confirmar rompimento/continuação real antes de agir"
        invalidation = "Se houver pavio forte contra a direção"
        reason = "falso_movimento"
        main_text = "Há risco de armadilha ou falso rompimento."
    elif img.structural_weakness >= 0.62:
        action = "WAIT_STRUCTURE"
        instruction = "Estrutura fraca, aguardar"
        trigger = "Esperar estrutura mais limpa"
        invalidation = "Se a estrutura continuar ruído/lateral"
        reason = "estrutura_fraca"
        main_text = "A estrutura ainda está fraca."
    elif img.reversal_pressure >= 0.66:
        action = "WAIT_REVERSAL"
        instruction = "Risco de reversão, aguardar"
        trigger = "Esperar confirmação da direção dominante"
        invalidation = "Se a reversão ganhar força"
        reason = "pressao_reversao"
        main_text = "Há pressão relevante de reversão."
    elif bias == "bullish":
        if timing_quality in ("early", "developing"):
            action = "WAIT_BUY_SETUP"
            instruction = "Preparando compra"
            trigger = "Comprar somente se a continuação se mantiver"
            invalidation = "Se perder continuidade compradora"
            reason = "setup_comprador_em_formacao"
            main_text = "Compra em preparação."
        else:
            action = "BUY"
            instruction = "Comprar"
            trigger = "Comprar enquanto a direção compradora seguir firme"
            invalidation = "Se aparecer rejeição forte contra a compra"
            reason = "entrada_compradora"
            main_text = "Cenário favorável para compra."
    elif bias == "bearish":
        if timing_quality in ("early", "developing"):
            action = "WAIT_SELL_SETUP"
            instruction = "Preparando venda"
            trigger = "Vender somente se a continuação vendedora se mantiver"
            invalidation = "Se perder continuidade vendedora"
            reason = "setup_vendedor_em_formacao"
            main_text = "Venda em preparação."
        else:
            action = "SELL"
            instruction = "Vender"
            trigger = "Vender enquanto a direção vendedora seguir firme"
            invalidation = "Se aparecer rejeição forte contra a venda"
            reason = "entrada_vendedora"
            main_text = "Cenário favorável para venda."
    else:
        action = "WAIT_LATERAL"
        instruction = "Mercado lateral, aguardar"
        trigger = "Esperar direção mais clara"
        invalidation = "Se continuar sem dominância direcional"
        reason = "lateralidade_ou_neutro"
        main_text = "Mercado sem direção clara."

    primary_trend = "uptrend" if bias == "bullish" else "downtrend" if bias == "bearish" else "sideways"
    micro_trend = primary_trend
    if img.micro_score < 0.45 and bias != "neutral":
        micro_trend = "mixed"

    return {
        "action": action,
        "instruction": instruction,
        "trigger": trigger,
        "invalidation": invalidation,
        "reason": reason,
        "main_text": main_text,
        "primaryTrend": primary_trend,
        "microTrend": micro_trend,
        "continuationProbability": continuation_prob,
        "reversalProbability": reversal_prob,
    }


def build_explanation(
    ctx: ParsedContext,
    img: ImageAnalysis,
    decision: Dict[str, Any],
    timing: Dict[str, Any],
    confidence: int,
    risk: str,
) -> str:
    parts = []

    parts.append(decision["main_text"])

    if ctx.asset:
        parts.append(f"Ativo {ctx.asset}.")
    if ctx.timeframe:
        parts.append(f"Timeframe {ctx.timeframe}.")
    if ctx.chart_clock:
        parts.append(f"Hora {ctx.chart_clock}.")
    if ctx.candle_time_remaining_sec is not None:
        parts.append(f"Vela {ctx.candle_time_remaining_sec}s.")

    parts.append(
        f"Viés {img.dominant_bias}. "
        f"Força {int(img.strength_score * 100)}%. "
        f"Continuação {int(img.continuation_health * 100)}%. "
        f"Reversão {int(img.reversal_pressure * 100)}%."
    )

    if img.detected_pattern != "unclear_pattern":
        parts.append(f"Padrão {img.detected_pattern}.")

    parts.append(
        f"Timing {timing['entryTiming']}. "
        f"Janela {timing['timingQuality']}. "
        f"Tempo recomendado {timing['recommendedOperationLabel']}."
    )

    parts.append(
        f"Risco {risk}. "
        f"Confiança {confidence}%. "
        f"Fake move {int(img.fake_move_risk * 100)}%. "
        f"Fraqueza estrutural {int(img.structural_weakness * 100)}%."
    )

    if ctx.warnings:
        parts.append("Alertas: " + ", ".join(ctx.warnings) + ".")

    return " ".join(parts)


def build_short_message(decision: Dict[str, Any], img: ImageAnalysis) -> str:
    action = decision["action"]
    if action == "BUY":
        return "Compra com contexto favorável"
    if action == "SELL":
        return "Venda com contexto favorável"
    if action == "WAIT_BUY_SETUP":
        return "Viés comprador em formação"
    if action == "WAIT_SELL_SETUP":
        return "Viés vendedor em formação"
    if action == "WAIT_FAKE_MOVE":
        return "Possível armadilha no movimento"
    if action == "WAIT_REVERSAL":
        return "Reversão em observação"
    if action == "WAIT_STRUCTURE":
        return "Estrutura ainda fraca"
    if action == "WAIT_ONE_MORE_CANDLE":
        return "Aguardar próxima vela"
    if action == "NO_ENTRY_LOW_PAYOUT":
        return "Payout abaixo do ideal"
    if action == "DATA_INSUFFICIENT":
        return "Sem dados mínimos"
    if img.dominant_bias == "neutral":
        return "Mercado sem dominância clara"
    return "Aguardando confirmação"


def strength_label(img: ImageAnalysis) -> str:
    s = img.strength_score
    if s >= 0.76:
        return "strong"
    if s >= 0.56:
        return "moderate"
    if s >= 0.38:
        return "light"
    return "weak"


def next_move_prediction(img: ImageAnalysis) -> str:
    if img.dominant_bias == "bullish":
        return "bullish_continuation_likely" if img.continuation_health >= 0.58 else "bullish_but_fragile"
    if img.dominant_bias == "bearish":
        return "bearish_continuation_likely" if img.continuation_health >= 0.58 else "bearish_but_fragile"
    return "sideways_or_unclear"


def market_state(img: ImageAnalysis) -> str:
    if img.structural_weakness >= 0.62:
        return "fragile_structure"
    if img.fake_move_risk >= 0.66:
        return "fake_move_risk"
    if img.reversal_pressure >= 0.66:
        return "reversal_pressure"
    if img.dominant_bias == "bullish":
        return "bullish"
    if img.dominant_bias == "bearish":
        return "bearish"
    return "sideways"


# =========================================================
# RESPOSTA
# =========================================================

def build_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    ctx = extract_context(payload)
    image = decode_image_from_payload(payload)
    img = analyze_image(image)

    timing = infer_timing(
        candle_sec=ctx.candle_time_remaining_sec,
        selected_operation_sec=ctx.selected_operation_sec,
    )
    confidence = confidence_from_context(ctx, img)
    risk = risk_from_context(ctx, img)
    decision = infer_action(ctx, img, timing, confidence, risk)

    explanation = build_explanation(ctx, img, decision, timing, confidence, risk)
    short_message = build_short_message(decision, img)

    return {
        "ok": True,
        "source": "railway",
        "serverTimestamp": now_iso(),

        "asset": ctx.asset,
        "currentPrice": ctx.current_price_raw,
        "candleTimeRemaining": ctx.candle_time_remaining_raw,
        "chartClock": ctx.chart_clock,
        "timeframe": ctx.timeframe,
        "payoutPercent": ctx.payout_percent or 0,
        "selectedOperationSec": ctx.selected_operation_sec or 60,

        "marketState": market_state(img),
        "nextMovePrediction": next_move_prediction(img),

        "action": normalize_outcome_action(decision["action"]),
        "instruction": decision["instruction"],
        "entryTiming": timing["entryTiming"],
        "secondsToAction": timing["secondsToAction"],
        "confidence": confidence,
        "risk": risk,

        "continuationProbability": decision["continuationProbability"],
        "reversalProbability": decision["reversalProbability"],

        "primaryTrend": decision["primaryTrend"],
        "microTrend": decision["microTrend"],
        "detectedPattern": img.detected_pattern,
        "trigger": decision["trigger"],
        "invalidation": decision["invalidation"],
        "waitCandles": timing["waitCandles"],

        "recommendedOperationSec": timing["recommendedOperationSec"],
        "recommendedOperationLabel": timing["recommendedOperationLabel"],

        "explanation": explanation,
        "shortMessage": short_message,
        "reason": decision["reason"],
        "strengthLabel": strength_label(img),
        "timingQuality": timing["timingQuality"],
        "idealEntryType": timing["idealEntryType"],

        "continuationHealth": round(img.continuation_health, 4),
        "reversalPressure": round(img.reversal_pressure, 4),
        "structuralWeakness": round(img.structural_weakness, 4),
        "fakeMoveRisk": round(img.fake_move_risk, 4),

        "imageQuality": img.quality_label,
        "imageNotes": img.notes,
        "contextWarnings": ctx.warnings,

        "debug": {
            "imageOk": img.image_ok,
            "imageWidth": img.width,
            "imageHeight": img.height,
            "brightness": round(img.brightness, 2),
            "contrast": round(img.contrast, 2),
            "sharpnessProxy": round(img.sharpness_proxy, 2),
            "bullishPressure": round(img.bullish_pressure, 4),
            "bearishPressure": round(img.bearish_pressure, 4),
            "neutralPressure": round(img.neutral_pressure, 4),
            "trendScore": round(img.trend_score, 4),
            "microScore": round(img.micro_score, 4),
            "breakoutScore": round(img.breakout_score, 4),
        },
    }


# =========================================================
# ROTAS
# =========================================================

@app.get("/")
async def root():
    uptime_sec = int(time.time() - app.state.started_at)
    return JSONResponse({
        "ok": True,
        "service": app.state.service_name,
        "version": APP_VERSION,
        "status": "online",
        "timestamp": now_iso(),
        "uptimeSec": uptime_sec,
        "ws": WS_PATH,
    })


@app.get("/health")
async def health():
    uptime_sec = int(time.time() - app.state.started_at)
    return {
        "ok": True,
        "status": "healthy",
        "service": app.state.service_name,
        "version": APP_VERSION,
        "timestamp": now_iso(),
        "uptimeSec": uptime_sec,
    }


@app.websocket(WS_PATH)
async def ws_sinais(websocket: WebSocket):
    await websocket.accept()

    await websocket.send_text(json.dumps({
        "ok": True,
        "type": "connected",
        "message": f"Cliente conectado no {WS_PATH}",
        "serverTimestamp": now_iso(),
    }))

    try:
        while True:
            raw = await websocket.receive_text()
            started = time.time()

            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("JSON precisa ser objeto")
            except Exception:
                await websocket.send_text(json.dumps({
                    "ok": False,
                    "type": "error",
                    "message": "Payload inválido: JSON malformado",
                    "serverTimestamp": now_iso(),
                }))
                continue

            command = safe_str(payload.get("command") or payload.get("comando") or "analisar_grafico")
            if command and command not in ("analisar_grafico", "analyze_chart", "analyze"):
                await websocket.send_text(json.dumps({
                    "ok": False,
                    "type": "error",
                    "message": f"Comando não suportado: {command}",
                    "serverTimestamp": now_iso(),
                }))
                continue

            try:
                result = build_response(payload)
                result["latencyMs"] = int((time.time() - started) * 1000)
                await websocket.send_text(json.dumps(result))
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "ok": False,
                    "type": "error",
                    "message": f"Erro interno ao analisar: {str(e)}",
                    "serverTimestamp": now_iso(),
                }))

    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass

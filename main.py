import base64
import io
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from PIL import Image, ImageStat

APP_VERSION = "5.0.0-tradingview-smart-context-safe"
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
# MODELOS
# =========================================================

@dataclass
class ParsedContext:
    asset: str = ""
    current_price_raw: str = ""
    current_price_normalized: str = ""
    current_price_value: Optional[float] = None
    candle_time_remaining_raw: str = ""
    candle_time_remaining_sec: Optional[int] = None
    chart_clock: str = ""
    timeframe: str = ""
    selected_operation_sec: Optional[int] = None
    chart_base_sec: int = 300

    platform: str = "TRADINGVIEW"
    broker_profile: str = "TRADINGVIEW_BINGX_COMPATIBLE"
    operation_mode: str = "manual_tradingview_analysis"

    chart_readable: bool = False
    order_panel_open: bool = False
    blocking_popup: bool = False
    paper_trading: bool = False
    market_view_mode: str = ""
    screen_status: str = ""

    asset_reliable: bool = False
    timeframe_reliable: bool = False
    price_reliable: bool = False
    timer_reliable: bool = False
    context_reliable: bool = False

    price_confidence: int = 0
    price_source: str = ""
    price_status: str = ""

    payout_percent: Optional[int] = None
    payout_ignored: bool = True
    binary_broker_mode: bool = False

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


# =========================================================
# UTIL
# =========================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


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


def parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "sim", "on"):
        return True
    if text in ("false", "0", "no", "nao", "não", "off"):
        return False

    return None


def parse_bool_any(payload: Dict[str, Any], keys: list[str]) -> Optional[bool]:
    for key in keys:
        if key in payload:
            parsed = parse_bool(payload.get(key))
            if parsed is not None:
                return parsed
    return None


def contains_any(text: str, terms: list[str]) -> bool:
    upper = text.upper()
    return any(term.upper() in upper for term in terms)


def raw_payload_text(payload: Dict[str, Any]) -> str:
    try:
        limited = {
            key: value
            for key, value in payload.items()
            if key not in ("imageBase64", "image_base64", "image", "frame")
        }
        return json.dumps(limited, ensure_ascii=False)
    except Exception:
        return str(payload)


# =========================================================
# NORMALIZAÇÃO DE ATIVO / PREÇO / TEMPO
# =========================================================

def normalize_asset(value: Any) -> str:
    raw = safe_str(value).upper()
    if not raw:
        return ""

    compact = (
        raw.replace(" ", "")
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
        .replace(":", "")
    )

    if "BITCOIN" in raw or "BTCUSDT" in compact:
        return "BTCUSDT"
    if "ETHEREUM" in raw or "ETHER" in raw or "ETHUSDT" in compact:
        return "ETHUSDT"
    if "SOLANA" in raw or "SOLUSDT" in compact:
        return "SOLUSDT"
    if "XRPUSDT" in compact or "RIPPLE" in raw:
        return "XRPUSDT"
    if "DOGEUSDT" in compact or "DOGECOIN" in raw:
        return "DOGEUSDT"
    if "BNBUSDT" in compact:
        return "BNBUSDT"
    if "XAUUSD" in compact or "GOLD" in raw or "OURO" in raw:
        return "XAUUSD"
    if "EURUSD" in compact:
        return "EURUSD"
    if "GBPUSD" in compact:
        return "GBPUSD"
    if "USDJPY" in compact:
        return "USDJPY"
    if "NAS100" in compact or "US100" in compact:
        return "NAS100"
    if "SPX" in compact or "US500" in compact or "SP500" in compact:
        return "SPX"

    allowed = "".join(ch for ch in compact if ch.isalnum() or ch == ".")
    if 3 <= len(allowed) <= 14:
        return allowed

    return ""


def normalize_timeframe(raw: Any) -> str:
    value = safe_str(raw).upper().replace(" ", "")
    value = (
        value.replace("MINUTOS", "M")
        .replace("MINUTO", "M")
        .replace("MIN", "M")
    )

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
        "30M": "M30",
        "M30": "M30",
        "45M": "M45",
        "M45": "M45",
        "1H": "H1",
        "H1": "H1",
        "2H": "H2",
        "H2": "H2",
        "4H": "H4",
        "H4": "H4",
        "1D": "D1",
        "D1": "D1",
    }

    return mapping.get(value, value if value else "")


def timeframe_to_seconds(tf: str) -> Optional[int]:
    tf = normalize_timeframe(tf)
    return {
        "M1": 60,
        "M2": 120,
        "M3": 180,
        "M5": 300,
        "M10": 600,
        "M15": 900,
        "M30": 1800,
        "M45": 2700,
        "H1": 3600,
        "H2": 7200,
        "H4": 14400,
        "D1": 86400,
    }.get(tf)


def resolve_chart_base_sec(timeframe: str, selected_operation_sec: Optional[int]) -> int:
    from_tf = timeframe_to_seconds(timeframe)
    if from_tf is not None:
        return from_tf

    if selected_operation_sec is not None and 1 <= selected_operation_sec <= 86400:
        return selected_operation_sec

    return 300


def chart_label(sec: Optional[int], timeframe: str = "") -> str:
    safe_sec = sec if sec and sec > 0 else 300
    tf = normalize_timeframe(timeframe)

    if tf:
        return f"gráfico {tf}"

    if safe_sec >= 3600 and safe_sec % 3600 == 0:
        return f"gráfico {safe_sec // 3600}h"

    if safe_sec >= 60 and safe_sec % 60 == 0:
        return f"gráfico {safe_sec // 60}m"

    return f"gráfico {safe_sec}s"


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

    return asset in {
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "NZDUSD",
        "USDCAD",
        "USDCHF",
        "EURJPY",
        "GBPJPY",
        "USDBRL",
    }


def normalize_single_separator_number(value: str, separator: str, asset: str) -> str:
    parts = value.split(separator)
    if len(parts) != 2:
        return value.replace(separator, ".")

    left, right = parts
    decimal_interpretation = f"{left}.{right}"
    thousand_interpretation = f"{left}{right}"

    right_looks_thousands = len(right) == 3 and 1 <= len(left) <= 3
    asset_up = asset.upper()

    if right_looks_thousands:
        thousand_value = safe_float(thousand_interpretation)
        decimal_value = safe_float(decimal_interpretation)

        thousand_valid = price_reliable_for_asset(asset_up, thousand_value, thousand_interpretation)
        decimal_valid = price_reliable_for_asset(asset_up, decimal_value, decimal_interpretation)

        if thousand_valid and not decimal_valid:
            return thousand_interpretation

        if thousand_valid and decimal_valid:
            if any(key in asset_up for key in ("BTC", "ETH", "SOL", "XAU", "GOLD", "NAS", "SPX")):
                return thousand_interpretation

    return decimal_interpretation


def trim_leading_zeros_if_needed(value: str, asset: str) -> str:
    integer = value.split(".")[0]
    decimal = value.split(".", 1)[1] if "." in value else ""

    if len(integer) <= 1:
        return value

    asset_up = asset.upper()
    should_trim = (
        integer.startswith("00")
        or any(key in asset_up for key in ("BTC", "ETH", "SOL", "XAU", "GOLD", "NAS", "SPX"))
    )

    if not should_trim:
        return value

    trimmed_integer = integer.lstrip("0") or "0"
    return f"{trimmed_integer}.{decimal}" if decimal else trimmed_integer


def normalize_price_text(raw: Any, asset: str = "") -> Optional[str]:
    cleaned = (
        safe_str(raw)
        .replace("O", "0")
        .replace("o", "0")
        .replace("I", "1")
        .replace("l", "1")
        .replace("|", "1")
        .replace("R$", "")
        .replace("$", "")
        .replace("%", "")
        .replace(" ", "")
    )

    if not cleaned:
        return None

    has_comma = "," in cleaned
    has_dot = "." in cleaned

    if has_comma and has_dot:
        last_comma = cleaned.rfind(",")
        last_dot = cleaned.rfind(".")
        decimal_separator = "," if last_comma > last_dot else "."
        thousands_separator = "." if decimal_separator == "," else ","
        cleaned = cleaned.replace(thousands_separator, "").replace(decimal_separator, ".")
    elif has_comma:
        cleaned = normalize_single_separator_number(cleaned, ",", asset)
    elif has_dot:
        cleaned = normalize_single_separator_number(cleaned, ".", asset)

    final_value = "".join(ch for ch in cleaned if ch.isdigit() or ch == ".")

    if not final_value:
        return None
    if final_value.count(".") > 1:
        return None

    final_value = trim_leading_zeros_if_needed(final_value, asset)

    value = safe_float(final_value)
    if value is None or value <= 0:
        return None

    return final_value


def parse_price_value(raw: Any, asset: str = "") -> Optional[float]:
    normalized = normalize_price_text(raw, asset)
    if not normalized:
        return None
    return safe_float(normalized)


def price_reliable_for_asset(asset: str, value: Optional[float], raw: Any) -> bool:
    if value is None:
        return False

    normalized = normalize_price_text(raw, asset)
    if not normalized:
        return False

    integer_part = normalized.split(".")[0].replace("-", "")
    decimal_part = normalized.split(".")[1] if "." in normalized else ""

    int_digits = len(integer_part)
    dec_digits = len(decimal_part)
    asset_up = asset.upper()

    if value <= 0:
        return False

    if "BTC" in asset_up:
        return 4 <= int_digits <= 7 and dec_digits <= 5 and 1000.0 <= value <= 1_000_000.0
    if "ETH" in asset_up:
        return 3 <= int_digits <= 6 and dec_digits <= 5 and 100.0 <= value <= 100_000.0
    if "SOL" in asset_up:
        return 1 <= int_digits <= 5 and dec_digits <= 5 and 1.0 <= value <= 10_000.0
    if "XRP" in asset_up or "ADA" in asset_up:
        return 0.01 <= value <= 100.0 and dec_digits <= 6
    if "DOGE" in asset_up:
        return 0.001 <= value <= 100.0 and dec_digits <= 6
    if "BNB" in asset_up:
        return 1.0 <= value <= 50_000.0
    if "LTC" in asset_up:
        return 1.0 <= value <= 50_000.0
    if "XAU" in asset_up or "GOLD" in asset_up:
        return 3 <= int_digits <= 5 and dec_digits <= 5 and 100.0 <= value <= 20_000.0
    if is_forex_like(asset_up):
        if "JPY" in asset_up:
            return 2 <= int_digits <= 4 and 10.0 <= value <= 500.0 and dec_digits <= 5
        if "BRL" in asset_up:
            return 1 <= int_digits <= 3 and 1.0 <= value <= 50.0 and dec_digits <= 5
        return 1 <= int_digits <= 2 and 0.2 <= value <= 5.0 and dec_digits <= 6
    if any(key in asset_up for key in ("NAS", "US100", "SPX", "US500")):
        return 2 <= int_digits <= 7 and 10.0 <= value <= 500_000.0

    if not asset_up:
        return 0.001 <= value <= 1_000_000.0

    return 0.000001 <= value <= 1_000_000.0


def parse_mmss(raw: Any, max_seconds: Optional[int] = None) -> Optional[int]:
    text = safe_str(raw)
    if not text:
        return None

    if text.lower().endswith("s"):
        value = safe_int(text[:-1])
        if value is None:
            return None
        limit = max_seconds + 5 if max_seconds else 86400
        return value if 0 <= value <= limit else None

    import re

    match = re.search(r"\b(\d{1,2}):([0-5]\d)\b", text)
    if not match:
        return None

    mm = safe_int(match.group(1))
    ss = safe_int(match.group(2))

    if mm is None or ss is None:
        return None

    total = mm * 60 + ss
    limit = max_seconds + 5 if max_seconds else 86400

    if total > limit:
        return None

    return total


def normalize_timer_text(raw: Any, max_seconds: Optional[int] = None) -> str:
    text = safe_str(raw)
    if not text:
        return ""

    import re

    mmss = re.search(r"\b(\d{1,2}):([0-5]\d)\b", text)
    if mmss:
        return mmss.group(0)

    seconds = safe_int(text.replace("s", "").replace("S", ""))
    if seconds is not None:
        limit = max_seconds + 5 if max_seconds else 86400
        if 0 <= seconds <= limit:
            return f"{seconds}s"

    return ""


def parse_clock(raw: Any) -> str:
    text = safe_str(raw)
    if not text:
        return ""

    import re

    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b", text)
    if not match:
        return ""

    hh = safe_int(match.group(1))
    mm = safe_int(match.group(2))
    ss = safe_int(match.group(3)) if match.group(3) is not None else None

    if hh is None or mm is None:
        return ""

    if ss is None:
        return f"{hh:02d}:{mm:02d}"

    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def normalize_action(action: Any) -> str:
    value = safe_str(action).upper().replace("-", "_").replace(" ", "_")

    if not value:
        return "WAIT"

    if "DATA_INSUFFICIENT" in value or "INSUFFICIENT" in value:
        return "DATA_INSUFFICIENT"
    if "WAIT_ORDER_PANEL" in value:
        return "WAIT_ORDER_PANEL"
    if "WAIT_ONE_MORE_CANDLE" in value or "WAIT_NEXT_CANDLE" in value:
        return "WAIT_ONE_MORE_CANDLE"
    if "WAIT_THEN_BUY" in value:
        return "WAIT_BUY_SETUP"
    if "WAIT_THEN_SELL" in value:
        return "WAIT_SELL_SETUP"
    if "WAIT_BUY_SETUP" in value:
        return "WAIT_BUY_SETUP"
    if "WAIT_SELL_SETUP" in value:
        return "WAIT_SELL_SETUP"
    if "WAIT_STRUCTURE" in value:
        return "WAIT_STRUCTURE"
    if "WAIT_RISK" in value:
        return "WAIT_RISK"

    if value in ("CALL", "PUT") or "CALL/PUT" in value:
        return "WAIT_RISK"

    if value in ("BUY", "BUY_NOW", "LONG") or "COMPRAR" in value or "COMPRA" in value:
        return "BUY"

    if value in ("SELL", "SELL_NOW", "SHORT") or "VENDER" in value or "VENDA" in value:
        return "SELL"

    if "FAKE" in value:
        return "WAIT_RISK"
    if "REVERSAL" in value or "REVERSAO" in value or "REVERSÃO" in value:
        return "WAIT_RISK"
    if "CAUTION" in value:
        return "WAIT_RISK"
    if "WAIT" in value or "AGUARD" in value:
        return "WAIT"

    return "WAIT"


def normalize_risk(value: Any) -> str:
    upper = safe_str(value).upper()

    if "HIGH" in upper or "ALTO" in upper or "ALTA" in upper:
        return "high"
    if "LOW" in upper or "BAIXO" in upper or "BAIXA" in upper:
        return "low"

    return "medium"


def safe_seconds_to_action(action: str, seconds: Any) -> int:
    normalized = normalize_action(action)
    parsed = safe_int(seconds)

    if normalized in ("BUY", "SELL"):
        if parsed is None:
            return 0
        return int(clamp(parsed, 0, 600))

    return -1


def has_binary_legacy_text(text: str) -> bool:
    upper = text.upper()
    terms = [
        "NO_ENTRY_LOW_PAYOUT",
        "LOW_PAYOUT",
        "PAYOUT OBRIGATORIO",
        "PAYOUT OBRIGATÓRIO",
        "BINARY_BROKER",
        "BINARIA",
        "BINÁRIA",
        "EXPIRACAO",
        "EXPIRAÇÃO",
        "CALL/PUT",
        "CALL PUT",
    ]

    if any(term in upper for term in terms):
        return True

    import re
    return bool(re.search(r"\b(CALL|PUT)\b", upper))


# =========================================================
# CONTEXTO RECEBIDO DO APP
# =========================================================

def extract_context(payload: Dict[str, Any]) -> ParsedContext:
    warnings: list[str] = []

    platform = safe_str(payload.get("platform") or "TRADINGVIEW").upper() or "TRADINGVIEW"
    broker_profile = safe_str(payload.get("brokerProfile") or "TRADINGVIEW_BINGX_COMPATIBLE").upper() or "TRADINGVIEW_BINGX_COMPATIBLE"
    operation_mode = safe_str(payload.get("operationMode") or "manual_tradingview_analysis") or "manual_tradingview_analysis"

    asset = normalize_asset(
        payload.get("asset")
        or payload.get("ativo")
        or payload.get("symbol")
        or payload.get("ticker")
    )

    current_price_raw = safe_str(
        payload.get("currentPrice")
        or payload.get("price")
        or payload.get("preco")
    )

    current_price_normalized = normalize_price_text(current_price_raw, asset) or ""
    current_price_value = safe_float(current_price_normalized) if current_price_normalized else None

    timeframe = normalize_timeframe(
        payload.get("timeframe")
        or payload.get("tf")
        or payload.get("period")
        or payload.get("chartTimeframe")
    )

    selected_operation_sec = safe_int(
        payload.get("selectedOperationSec")
        or payload.get("chartBaseSec")
        or payload.get("timeframeSec")
        or payload.get("recommendedOperationSec")
        or payload.get("operationSec")
        or payload.get("duracao")
    )

    chart_base_sec = resolve_chart_base_sec(timeframe, selected_operation_sec)
    selected_operation_sec = chart_base_sec

    candle_time_remaining_raw = safe_str(
        payload.get("candleTimeRemaining")
        or payload.get("candle_time_remaining")
        or payload.get("tempoVela")
        or payload.get("timer")
    )
    candle_time_remaining_sec = parse_mmss(candle_time_remaining_raw, chart_base_sec)
    candle_time_remaining_raw = normalize_timer_text(candle_time_remaining_raw, chart_base_sec) or candle_time_remaining_raw

    chart_clock = parse_clock(
        payload.get("chartClock")
        or payload.get("clock")
        or payload.get("hora")
    )

    payout_percent = safe_int(payload.get("payoutPercent") or payload.get("payout"))
    if payout_percent is not None:
        payout_percent = int(clamp(payout_percent, -1, 100))

    text = raw_payload_text(payload)

    order_panel_open_payload = parse_bool_any(payload, ["orderPanelOpen", "order_panel_open"])
    blocking_popup_payload = parse_bool_any(payload, ["blockingPopup", "screenBlocked", "blocking_popup"])
    paper_trading_payload = parse_bool_any(payload, ["paperTrading", "paperTradingMode", "paper_trading"])
    chart_readable_payload = parse_bool_any(payload, ["chartReadable", "chart_readable"])

    order_panel_open = order_panel_open_payload
    if order_panel_open is None:
        order_panel_open = contains_any(
            text,
            [
                "TAKE PROFIT",
                "STOP LOSS",
                "TEMPO EM VIGOR",
                "UNIDADES",
                "ORDEM DE MERCADO",
                "PAINEL DE ORDEM",
            ],
        )

    blocking_popup = blocking_popup_payload
    if blocking_popup is None:
        blocking_popup = contains_any(
            text,
            [
                "PEDIDO REJEITADO",
                "CONCLUA A ATUALIZACAO",
                "CONCLUA A ATUALIZAÇÃO",
                "ATUALIZACAO DO KYC",
                "ATUALIZAÇÃO DO KYC",
                "VERIFICACAO DE IDENTIDADE",
                "VERIFICAÇÃO DE IDENTIDADE",
                "DOCUMENTO",
                "SEM CONEXAO",
                "SEM CONEXÃO",
                "SUPORTE AO CLIENTE",
            ],
        )

    paper_trading = paper_trading_payload
    if paper_trading is None:
        paper_trading = contains_any(text, ["PAPER TRADING", "PAPER", "SIMULADOR", "CONTA DEMO"])

    market_view_mode = safe_str(payload.get("marketViewMode") or payload.get("viewMode"))

    price_confidence = safe_int(
        payload.get("priceConfidence")
        or payload.get("detectedPriceConfidence")
        or payload.get("detectedPriceConfidenceFinal")
    )

    if price_confidence is None:
        price_confidence = 70 if current_price_normalized else 0

    price_confidence = int(clamp(price_confidence, 0, 100))

    price_source = safe_str(
        payload.get("priceSource")
        or payload.get("detectedPriceSource")
        or payload.get("detectedPriceSourceFinal")
    )

    price_status = safe_str(payload.get("priceStatus"))

    asset_reliable = bool(asset)
    timeframe_reliable = bool(timeframe)
    price_reliable = bool(
        current_price_normalized
        and price_confidence >= 55
        and price_reliable_for_asset(asset, current_price_value, current_price_normalized)
    )
    timer_reliable = candle_time_remaining_sec is not None

    if not asset_reliable:
        warnings.append("asset_missing")

    if not current_price_raw:
        warnings.append("price_missing")
    elif not price_reliable:
        warnings.append("price_unreliable")

    if price_source and ("axis" in price_source.lower() or "scale" in price_source.lower()) and price_confidence < 70:
        warnings.append("price_axis_weak")

    if price_status and any(word in price_status.lower() for word in ("rejected", "invalid", "weak")):
        warnings.append("price_status_weak")

    if candle_time_remaining_raw and candle_time_remaining_sec is None:
        warnings.append("candle_time_invalid")

    if chart_clock == "":
        warnings.append("clock_missing")

    if not timeframe_reliable:
        warnings.append("timeframe_missing")

    if order_panel_open:
        warnings.append("order_panel_open")

    if blocking_popup:
        warnings.append("blocking_popup")

    context_reliable = asset_reliable and timeframe_reliable and price_reliable
    chart_readable = context_reliable

    if chart_readable_payload is True and asset_reliable and timeframe_reliable:
        chart_readable = True

    if chart_readable_payload is False:
        warnings.append("chart_readable_false_from_client")

    if not chart_readable:
        warnings.append("chart_not_readable")

    screen_status = safe_str(payload.get("screenStatus") or payload.get("status"))
    if not screen_status:
        if blocking_popup:
            screen_status = "blocked_popup"
        elif order_panel_open:
            screen_status = "order_panel_open"
        elif context_reliable:
            screen_status = "chart_readable"
        elif asset_reliable and timeframe_reliable:
            screen_status = "partial_context"
        else:
            screen_status = "insufficient_context"

    return ParsedContext(
        asset=asset,
        current_price_raw=current_price_raw,
        current_price_normalized=current_price_normalized,
        current_price_value=current_price_value,
        candle_time_remaining_raw=candle_time_remaining_raw,
        candle_time_remaining_sec=candle_time_remaining_sec,
        chart_clock=chart_clock,
        timeframe=timeframe,
        payout_percent=payout_percent,
        selected_operation_sec=selected_operation_sec,
        chart_base_sec=chart_base_sec,
        warnings=warnings,
        platform=platform,
        broker_profile=broker_profile,
        operation_mode=operation_mode,
        chart_readable=chart_readable,
        order_panel_open=order_panel_open,
        blocking_popup=blocking_popup,
        paper_trading=paper_trading,
        market_view_mode=market_view_mode,
        screen_status=screen_status,
        asset_reliable=asset_reliable,
        timeframe_reliable=timeframe_reliable,
        price_reliable=price_reliable,
        timer_reliable=timer_reliable,
        context_reliable=context_reliable,
        price_confidence=price_confidence,
        price_source=price_source,
        price_status=price_status,
        payout_ignored=True,
        binary_broker_mode=False,
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
    max_ch = max(r, g, b)
    min_ch = min(r, g, b)
    sat = max_ch - min_ch
    lum = (r * 299 + g * 587 + b * 114) / 1000

    if lum < 28 or sat < 28:
        return "neutral"

    green = (
        g >= 75
        and g > r + 18
        and g >= b - 22
    ) or (
        g >= 90
        and b >= 65
        and r <= 135
        and g >= r + 20
    )

    red = (
        r >= 88
        and r > g + 20
        and r > b + 2
    ) or (
        r >= 112
        and b >= 45
        and r > g + 24
    )

    if green:
        return "green"
    if red:
        return "red"

    return "neutral"


def analyze_color_pressures(img: Image.Image) -> Tuple[float, float, float]:
    chart = crop_box(img, 0.02, 0.10, 0.84, 0.66).resize((180, 105))
    data = list(chart.getdata())

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
    chart = crop_box(img, 0.08, 0.14, 0.76, 0.56).resize((112, 56))
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
    if structural_weakness >= 0.70:
        return "consolidation"

    if fake_move_risk >= 0.74:
        return "fake_move_risk_context"

    if reversal_pressure >= 0.68:
        return "possible_bearish_reversal" if dominant_bias == "bullish" else "possible_bullish_reversal"

    if breakout_score >= 0.72 and continuation_health >= 0.58 and fake_move_risk < 0.45:
        return "bullish_breakout_continuation" if dominant_bias == "bullish" else "bearish_breakout_continuation"

    if continuation_health >= 0.62 and structural_weakness < 0.42:
        return "bullish_continuation" if dominant_bias == "bullish" else "bearish_continuation"

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
            fake_move_risk=0.65,
            structural_weakness=0.70,
            reversal_pressure=0.40,
            continuation_health=0.15,
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

    if raw_bias > 0.025:
        dominant_bias = "bullish"
    elif raw_bias < -0.025:
        dominant_bias = "bearish"
    else:
        dominant_bias = "neutral"

    strength_score = clamp(abs(raw_bias) * 2.35 + abs(macro) * 0.55, 0.0, 1.0)
    trend_score = clamp((macro + trend + 1.2) / 2.4, 0.0, 1.0)
    micro_score = clamp((micro + 1.0) / 2.0, 0.0, 1.0)

    breakout_score = clamp(
        (abs(macro) * 0.42)
        + (abs(trend) * 0.28)
        + (contrast / 64.0) * 0.30,
        0.0,
        1.0,
    )

    structural_weakness = clamp(
        (0.42 if contrast < 24 else 0.18)
        + (0.24 if sharpness < 10 else 0.08)
        + (neutral * 0.22),
        0.0,
        1.0,
    )

    fake_move_risk = clamp(
        (0.52 if abs(raw_bias) < 0.06 else 0.14)
        + neutral * 0.20
        + (0.18 if abs(trend) < 0.06 else 0.04),
        0.0,
        1.0,
    )

    reversal_pressure = clamp(
        (0.25 if dominant_bias == "neutral" else 0.08)
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

    pattern_bias = "bullish" if raw_bias >= 0 else "bearish"
    detected_pattern = detect_pattern(
        dominant_bias=pattern_bias,
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
    if neutral > 0.88:
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
# TIMING / DECISÃO
# =========================================================

def infer_timing(candle_sec: Optional[int], chart_base_sec: int, timeframe: str) -> Dict[str, Any]:
    duration = chart_base_sec if chart_base_sec > 0 else 300
    label = chart_label(duration, timeframe)

    if candle_sec is None:
        return {
            "timingQuality": "unknown",
            "entryTiming": "timer_indisponivel",
            "secondsToAction": -1,
            "waitCandles": 0,
            "recommendedOperationSec": duration,
            "recommendedOperationLabel": label,
            "idealEntryType": "wait_confirmation",
        }

    remaining = max(0, min(duration, candle_sec))
    elapsed = duration - remaining
    progress = clamp(elapsed / max(1, duration), 0.0, 1.0)

    def seconds_until(target_progress: float) -> int:
        target_elapsed = int(duration * target_progress)
        return max(1, target_elapsed - elapsed)

    if duration <= 60:
        early_limit = 0.08
        build_limit = 0.20
        ideal_limit = 0.62
        late_limit = 0.78
        very_late_limit = 0.90
    elif duration <= 300:
        early_limit = 0.08
        build_limit = 0.22
        ideal_limit = 0.70
        late_limit = 0.84
        very_late_limit = 0.93
    else:
        early_limit = 0.07
        build_limit = 0.20
        ideal_limit = 0.68
        late_limit = 0.84
        very_late_limit = 0.93

    if progress < early_limit:
        return {
            "timingQuality": "too_early",
            "entryTiming": "vela_nova",
            "secondsToAction": seconds_until(build_limit),
            "waitCandles": 0,
            "recommendedOperationSec": duration,
            "recommendedOperationLabel": label,
            "idealEntryType": "wait_for_setup",
        }

    if progress < build_limit:
        return {
            "timingQuality": "building",
            "entryTiming": "janela_em_formacao",
            "secondsToAction": seconds_until(min(0.30, ideal_limit)),
            "waitCandles": 0,
            "recommendedOperationSec": duration,
            "recommendedOperationLabel": label,
            "idealEntryType": "setup_building",
        }

    if progress <= ideal_limit:
        return {
            "timingQuality": "ideal_window",
            "entryTiming": "janela_operacional",
            "secondsToAction": 0,
            "waitCandles": 0,
            "recommendedOperationSec": duration,
            "recommendedOperationLabel": label,
            "idealEntryType": "immediate_or_sniper",
        }

    if progress <= late_limit:
        return {
            "timingQuality": "late_warning",
            "entryTiming": "final_de_vela_chegando",
            "secondsToAction": -1,
            "waitCandles": 1,
            "recommendedOperationSec": duration,
            "recommendedOperationLabel": label,
            "idealEntryType": "only_if_strong",
        }

    return {
        "timingQuality": "late",
        "entryTiming": "aguardar_proxima_vela",
        "secondsToAction": -1,
        "waitCandles": 1,
        "recommendedOperationSec": duration,
        "recommendedOperationLabel": label,
        "idealEntryType": "next_candle",
    }


def confidence_from_context(ctx: ParsedContext, img: ImageAnalysis) -> int:
    confidence = 46

    if img.image_ok:
        confidence += int(img.strength_score * 20)
        confidence += int((1.0 - img.structural_weakness) * 10)
        confidence += int((1.0 - img.fake_move_risk) * 8)

    if ctx.asset_reliable:
        confidence += 5
    else:
        confidence -= 15

    if ctx.timeframe_reliable:
        confidence += 5
    else:
        confidence -= 15

    if ctx.price_reliable:
        confidence += 8
    else:
        confidence -= 14

    if ctx.timer_reliable:
        confidence += 3

    if ctx.chart_clock:
        confidence += 2

    if ctx.paper_trading:
        confidence += 1

    if ctx.order_panel_open:
        confidence -= 12

    if ctx.blocking_popup:
        confidence -= 15

    if not ctx.chart_readable:
        confidence -= 10

    if img.quality_label == "weak":
        confidence -= 6

    if img.dominant_bias == "neutral":
        confidence -= 5

    return int(clamp(confidence, 8, 95))


def risk_from_context(ctx: ParsedContext, img: ImageAnalysis) -> str:
    score = 0.0
    score += img.fake_move_risk * 0.30
    score += img.structural_weakness * 0.26
    score += img.reversal_pressure * 0.20

    if not ctx.context_reliable:
        score += 0.20

    if not ctx.price_reliable:
        score += 0.16

    if ctx.order_panel_open:
        score += 0.20

    if ctx.blocking_popup:
        score += 0.25

    if not ctx.chart_readable:
        score += 0.12

    if score >= 0.70:
        return "high"

    if score >= 0.44:
        return "medium"

    return "low"


def infer_action(ctx: ParsedContext, img: ImageAnalysis, timing: Dict[str, Any], confidence: int, risk: str) -> Dict[str, Any]:
    timing_quality = timing["timingQuality"]
    bias = img.dominant_bias

    continuation_prob = int(clamp(img.continuation_health * 100, 0, 100))
    reversal_prob = int(clamp(img.reversal_pressure * 100, 0, 100))

    if ctx.blocking_popup:
        action = "DATA_INSUFFICIENT"
        instruction = "Tela bloqueada • feche aviso/popup"
        trigger = "Fechar popup/aviso antes de analisar"
        invalidation = "Enquanto houver aviso cobrindo a tela"
        reason = "popup_bloqueando_contexto"
        main_text = "A tela está bloqueada por aviso ou popup."
    elif ctx.order_panel_open:
        action = "WAIT_ORDER_PANEL"
        instruction = "Painel aberto • feche para analisar"
        trigger = "Fechar o painel de ordem para a IA ler candles"
        invalidation = "Enquanto o painel cobrir o gráfico"
        reason = "painel_de_ordem_aberto"
        main_text = "O painel de ordem está aberto e pode cobrir candles."
    elif not ctx.asset_reliable or not ctx.timeframe_reliable:
        action = "DATA_INSUFFICIENT"
        instruction = "Dados mínimos ausentes"
        trigger = "Confirmar ativo e timeframe no TradingView"
        invalidation = "Enquanto ativo/timeframe não forem detectados"
        reason = "contexto_minimo_insuficiente"
        main_text = "Falta ativo ou timeframe para analisar com segurança."
    elif not ctx.price_reliable:
        action = "DATA_INSUFFICIENT"
        instruction = "Preço suspeito • aguardar"
        trigger = "Confirmar preço atual na etiqueta do TradingView"
        invalidation = "Enquanto o preço vier ausente, quebrado ou do eixo lateral"
        reason = "preco_nao_confiavel"
        main_text = "O preço atual não está confiável para liberar entrada."
    elif not img.image_ok:
        action = "DATA_INSUFFICIENT"
        instruction = "Imagem ausente • aguardar"
        trigger = "Manter TradingView aberto com candles visíveis"
        invalidation = "Se a captura não enviar frame"
        reason = "imagem_indisponivel"
        main_text = "O contexto existe, mas a imagem do gráfico não chegou."
    elif risk == "high":
        action = "WAIT_RISK"
        instruction = "Risco alto • aguardar"
        trigger = "Aguardar cenário mais limpo"
        invalidation = "Se o risco continuar alto"
        reason = "risco_elevado"
        main_text = "Risco acima do ideal para entrada limpa."
    elif timing_quality in ("late", "late_warning", "very_late"):
        action = "WAIT_ONE_MORE_CANDLE"
        instruction = "Final de vela • aguardar próxima"
        trigger = "Reavaliar na próxima vela"
        invalidation = "Se a próxima vela também vier sem confirmação"
        reason = "janela_tardia"
        main_text = "A janela desta vela ficou tardia."
    elif img.fake_move_risk >= 0.74:
        action = "WAIT_RISK"
        instruction = "Movimento suspeito • aguardar"
        trigger = "Confirmar rompimento/continuação real antes de agir"
        invalidation = "Se houver pavio forte contra a direção"
        reason = "falso_movimento"
        main_text = "Há risco de armadilha ou falso rompimento."
    elif img.structural_weakness >= 0.72:
        action = "WAIT_STRUCTURE"
        instruction = "Estrutura fraca • aguardar"
        trigger = "Esperar estrutura mais limpa"
        invalidation = "Se a estrutura continuar ruidosa ou lateral"
        reason = "estrutura_fraca"
        main_text = "A estrutura ainda está fraca."
    elif img.reversal_pressure >= 0.74:
        action = "WAIT_RISK"
        instruction = "Risco de reversão • aguardar"
        trigger = "Esperar confirmação da direção dominante"
        invalidation = "Se a reversão ganhar força"
        reason = "pressao_reversao"
        main_text = "Há pressão relevante de reversão."
    elif confidence < 68:
        if bias == "bullish":
            action = "WAIT_BUY_SETUP"
            instruction = "Armar compra • aguardar confirmação"
            trigger = "Comprar somente se a continuação compradora se mantiver"
            invalidation = "Se perder força compradora"
            reason = "confianca_insuficiente_para_buy"
            main_text = "Existe viés comprador, mas ainda falta confirmação."
        elif bias == "bearish":
            action = "WAIT_SELL_SETUP"
            instruction = "Armar venda • aguardar confirmação"
            trigger = "Vender somente se a continuação vendedora se mantiver"
            invalidation = "Se perder força vendedora"
            reason = "confianca_insuficiente_para_sell"
            main_text = "Existe viés vendedor, mas ainda falta confirmação."
        else:
            action = "WAIT"
            instruction = "Mercado lateral • aguardar"
            trigger = "Esperar direção mais clara"
            invalidation = "Se continuar sem dominância direcional"
            reason = "lateralidade_ou_neutro"
            main_text = "Mercado sem direção clara."
    elif bias == "bullish":
        if timing_quality in ("too_early", "building", "unknown"):
            action = "WAIT_BUY_SETUP"
            instruction = "Armar compra • aguardar confirmação"
            trigger = "Comprar somente se a continuação compradora se mantiver"
            invalidation = "Se perder força compradora"
            reason = "setup_comprador_em_formacao"
            main_text = "Compra em preparação."
        else:
            action = "BUY"
            instruction = "Comprar • TradingView/BingX manual"
            trigger = "Comprar enquanto a direção compradora seguir firme"
            invalidation = "Se aparecer rejeição forte contra a compra"
            reason = "entrada_compradora"
            main_text = "Cenário favorável para compra."
    elif bias == "bearish":
        if timing_quality in ("too_early", "building", "unknown"):
            action = "WAIT_SELL_SETUP"
            instruction = "Armar venda • aguardar confirmação"
            trigger = "Vender somente se a continuação vendedora se mantiver"
            invalidation = "Se perder força vendedora"
            reason = "setup_vendedor_em_formacao"
            main_text = "Venda em preparação."
        else:
            action = "SELL"
            instruction = "Vender • TradingView/BingX manual"
            trigger = "Vender enquanto a direção vendedora seguir firme"
            invalidation = "Se aparecer rejeição forte contra a venda"
            reason = "entrada_vendedora"
            main_text = "Cenário favorável para venda."
    else:
        action = "WAIT"
        instruction = "Mercado lateral • aguardar"
        trigger = "Esperar direção mais clara"
        invalidation = "Se continuar sem dominância direcional"
        reason = "lateralidade_ou_neutro"
        main_text = "Mercado sem direção clara."

    primary_trend = "bullish" if bias == "bullish" else "bearish" if bias == "bearish" else "unclear"
    micro_trend = primary_trend

    if img.micro_score < 0.45 and bias != "neutral":
        micro_trend = "unclear"

    return {
        "action": normalize_action(action),
        "instruction": instruction,
        "trigger": trigger,
        "invalidation": invalidation,
        "reason": reason,
        "main_text": main_text,
        "primaryTrend": primary_trend,
        "microTrend": micro_trend,
        "continuationProbability": continuation_prob if ctx.context_reliable else 0,
        "reversalProbability": reversal_prob if ctx.asset_reliable and ctx.timeframe_reliable else 0,
    }


def enforce_final_safety(
    action: str,
    timing: Dict[str, Any],
    ctx: ParsedContext,
    confidence: int,
    risk: str,
) -> Tuple[str, int, int, str, str]:
    safe_action = normalize_action(action)
    safe_seconds = safe_seconds_to_action(safe_action, timing.get("secondsToAction", -1))
    safe_confidence = int(clamp(confidence, 0, 100))
    safe_risk = normalize_risk(risk)
    safety_reason = ""

    if safe_action in ("BUY", "SELL"):
        if not ctx.context_reliable:
            safety_reason = "entrada_bloqueada_por_contexto_nao_confiavel"
            safe_action = "DATA_INSUFFICIENT"
            safe_seconds = -1
            safe_confidence = min(safe_confidence, 55)
            safe_risk = "high"
        elif safe_confidence < 68:
            safety_reason = "entrada_bloqueada_por_confianca_baixa"
            safe_action = "WAIT_BUY_SETUP" if safe_action == "BUY" else "WAIT_SELL_SETUP"
            safe_seconds = -1
        elif safe_risk == "high":
            safety_reason = "entrada_bloqueada_por_risco_alto"
            safe_action = "WAIT_RISK"
            safe_seconds = -1

    if safe_action not in ("BUY", "SELL"):
        safe_seconds = -1

    return safe_action, safe_seconds, safe_confidence, safe_risk, safety_reason


def build_explanation(
    ctx: ParsedContext,
    img: ImageAnalysis,
    decision: Dict[str, Any],
    timing: Dict[str, Any],
    confidence: int,
    risk: str,
    safety_reason: str,
) -> str:
    parts = []

    parts.append(decision["main_text"])
    parts.append("Modo TradingView/BingX com payout ignorado e sem lógica de binária.")

    if ctx.asset:
        parts.append(f"Ativo {ctx.asset}.")
    else:
        parts.append("Ativo ausente.")

    if ctx.current_price_normalized:
        parts.append(f"Preço {ctx.current_price_normalized}.")
    else:
        parts.append("Preço ausente ou suspeito.")

    if ctx.timeframe:
        parts.append(f"Timeframe {ctx.timeframe}.")
    else:
        parts.append("Timeframe ausente.")

    if ctx.chart_clock:
        parts.append(f"Hora {ctx.chart_clock}.")

    if ctx.candle_time_remaining_sec is not None:
        parts.append(f"Vela {ctx.candle_time_remaining_sec}s.")
    else:
        parts.append("Timer da vela não confirmado.")

    parts.append(
        f"Viés visual {img.dominant_bias}. "
        f"Força {int(img.strength_score * 100)}%. "
        f"Continuação {int(img.continuation_health * 100)}%. "
        f"Reversão {int(img.reversal_pressure * 100)}%."
    )

    if img.detected_pattern != "unclear_pattern":
        parts.append(f"Padrão {img.detected_pattern}.")

    parts.append(
        f"Timing {timing['entryTiming']}. "
        f"Janela {timing['timingQuality']}. "
        f"Base {timing['recommendedOperationLabel']}."
    )

    parts.append(
        f"Risco {risk}. "
        f"Confiança {confidence}%. "
        f"Fake move {int(img.fake_move_risk * 100)}%. "
        f"Fraqueza estrutural {int(img.structural_weakness * 100)}%."
    )

    parts.append(
        f"Tela {ctx.screen_status}. "
        f"Chart readable {ctx.chart_readable}. "
        f"Context reliable {ctx.context_reliable}. "
        f"Preço confiável {ctx.price_reliable}. "
        f"Fonte preço {ctx.price_source or 'não informada'}. "
        f"Confiança preço {ctx.price_confidence}%."
    )

    if safety_reason:
        parts.append(f"Trava aplicada: {safety_reason}.")

    if ctx.warnings:
        parts.append("Alertas: " + ", ".join(ctx.warnings) + ".")

    return " ".join(parts)


def build_short_message(action: str, img: ImageAnalysis, ctx: ParsedContext) -> str:
    if action == "BUY":
        return "Compra com contexto favorável"
    if action == "SELL":
        return "Venda com contexto favorável"
    if action == "WAIT_BUY_SETUP":
        return "Viés comprador em formação"
    if action == "WAIT_SELL_SETUP":
        return "Viés vendedor em formação"
    if action == "WAIT_RISK":
        return "Risco elevado, aguardar"
    if action == "WAIT_STRUCTURE":
        return "Estrutura ainda fraca"
    if action == "WAIT_ONE_MORE_CANDLE":
        return "Aguardar próxima vela"
    if action == "WAIT_ORDER_PANEL":
        return "Feche o painel de ordem"
    if action == "DATA_INSUFFICIENT":
        if not ctx.price_reliable and ctx.asset and ctx.timeframe:
            return "Preço não confiável"
        return "Dados insuficientes"

    if img.dominant_bias == "neutral":
        return "Mercado sem dominância clara"

    return "Aguardando confirmação"


def strength_label(img: ImageAnalysis, action: str) -> str:
    if action == "DATA_INSUFFICIENT":
        return "insufficient"

    s = img.strength_score

    if s >= 0.76:
        return "strong"
    if s >= 0.56:
        return "moderate"
    if s >= 0.38:
        return "weak"

    return "fragile"


def next_move_prediction(ctx: ParsedContext, img: ImageAnalysis) -> str:
    if not ctx.context_reliable:
        return "insufficient_context_no_prediction"

    if img.dominant_bias == "bullish":
        if img.continuation_health >= 0.58:
            return "bullish_continuation_likely"
        return "bullish_but_fragile"

    if img.dominant_bias == "bearish":
        if img.continuation_health >= 0.58:
            return "bearish_continuation_likely"
        return "bearish_but_fragile"

    return "sideways_or_unclear"


def market_state(ctx: ParsedContext, img: ImageAnalysis) -> str:
    if not ctx.context_reliable:
        return "insufficient_context"

    if img.structural_weakness >= 0.74:
        return "fragile_structure"

    if img.fake_move_risk >= 0.76:
        return "fake_move_risk"

    if img.reversal_pressure >= 0.78:
        return "reversal_pressure"

    if img.dominant_bias == "bullish":
        return "bullish"

    if img.dominant_bias == "bearish":
        return "bearish"

    return "sideways"


def screen_quality(ctx: ParsedContext) -> float:
    score = 1.0

    if not ctx.chart_readable:
        score -= 0.25

    if ctx.order_panel_open:
        score -= 0.30

    if ctx.blocking_popup:
        score -= 0.35

    if not ctx.asset_reliable:
        score -= 0.25

    if not ctx.price_reliable:
        score -= 0.25

    if not ctx.timeframe_reliable:
        score -= 0.25

    if not ctx.timer_reliable:
        score -= 0.06

    return clamp(score, 0.0, 1.0)


def context_reliability_score(ctx: ParsedContext) -> float:
    score = 0.0

    if ctx.asset_reliable:
        score += 0.25

    if ctx.price_reliable:
        score += 0.30

    if ctx.timeframe_reliable:
        score += 0.25

    if ctx.timer_reliable:
        score += 0.12

    if ctx.chart_clock:
        score += 0.08

    if ctx.order_panel_open:
        score -= 0.20

    if ctx.blocking_popup:
        score -= 0.25

    return clamp(score, 0.0, 1.0)


# =========================================================
# RESPOSTA
# =========================================================

def build_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    if has_binary_legacy_text(raw_payload_text(payload)):
        return {
            "ok": False,
            "source": "railway",
            "serverTimestamp": now_iso(),
            "platform": "TRADINGVIEW",
            "brokerProfile": "TRADINGVIEW_BINGX_COMPATIBLE",
            "operationMode": "manual_tradingview_analysis",
            "binaryBrokerMode": False,
            "payoutIgnored": True,
            "action": "WAIT_RISK",
            "instruction": "Payload rejeitado: lógica antiga de binária/payout",
            "entryTiming": "payload_incompativel",
            "secondsToAction": -1,
            "confidence": 0,
            "risk": "high",
            "shortMessage": "Payload incompatível",
            "reason": "payload_mencionou_binaria_payout_call_put",
        }

    ctx = extract_context(payload)
    image = decode_image_from_payload(payload)
    img = analyze_image(image)

    timing = infer_timing(
        candle_sec=ctx.candle_time_remaining_sec,
        chart_base_sec=ctx.chart_base_sec,
        timeframe=ctx.timeframe,
    )

    confidence = confidence_from_context(ctx, img)
    risk = risk_from_context(ctx, img)
    decision = infer_action(ctx, img, timing, confidence, risk)

    final_action, final_seconds, confidence, risk, safety_reason = enforce_final_safety(
        action=decision["action"],
        timing=timing,
        ctx=ctx,
        confidence=confidence,
        risk=risk,
    )

    if final_action == "DATA_INSUFFICIENT":
        confidence = min(confidence, 35)
        risk = "high"

    explanation = build_explanation(ctx, img, decision, timing, confidence, risk, safety_reason)
    short_message = build_short_message(final_action, img, ctx)

    sq = screen_quality(ctx)
    cr = context_reliability_score(ctx)

    continuation_probability = decision["continuationProbability"] if ctx.context_reliable else 0
    reversal_probability = decision["reversalProbability"] if ctx.asset_reliable and ctx.timeframe_reliable else 0

    if continuation_probability > 0 and reversal_probability > 0:
        if abs(continuation_probability - reversal_probability) < 12:
            continuation_probability = min(continuation_probability, 60)
            reversal_probability = min(reversal_probability, 60)

    return {
        "ok": True,
        "source": "railway",
        "serverTimestamp": now_iso(),

        "platform": "TRADINGVIEW",
        "brokerProfile": "TRADINGVIEW_BINGX_COMPATIBLE",
        "operationMode": "manual_tradingview_analysis",
        "binaryBrokerMode": False,
        "payoutIgnored": True,

        "screenStatus": ctx.screen_status,
        "marketViewMode": ctx.market_view_mode,
        "chartReadable": ctx.chart_readable,
        "orderPanelOpen": ctx.order_panel_open,
        "blockingPopup": ctx.blocking_popup,
        "paperTrading": ctx.paper_trading,
        "screenQuality": round(sq, 4),
        "contextReliability": round(cr, 4),

        "asset": ctx.asset,
        "currentPrice": ctx.current_price_normalized or ctx.current_price_raw,
        "candleTimeRemaining": ctx.candle_time_remaining_raw,
        "chartClock": ctx.chart_clock,
        "timeframe": ctx.timeframe,

        "assetReliable": ctx.asset_reliable,
        "timeframeReliable": ctx.timeframe_reliable,
        "priceReliable": ctx.price_reliable,
        "timerReliable": ctx.timer_reliable,
        "contextReliable": ctx.context_reliable,
        "priceConfidence": ctx.price_confidence,
        "priceSource": ctx.price_source,
        "priceStatus": ctx.price_status,

        "payoutPercent": -1,
        "selectedOperationSec": ctx.chart_base_sec,
        "chartBaseSec": ctx.chart_base_sec,
        "timeframeSec": ctx.chart_base_sec,

        "marketState": market_state(ctx, img),
        "nextMovePrediction": next_move_prediction(ctx, img),

        "action": final_action,
        "instruction": decision["instruction"],
        "entryTiming": timing["entryTiming"] if final_action in ("BUY", "SELL") else decision.get("entryTiming", timing["entryTiming"]),
        "secondsToAction": final_seconds,
        "confidence": confidence,
        "risk": risk,

        "continuationProbability": continuation_probability,
        "reversalProbability": reversal_probability,

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
        "reason": safety_reason or decision["reason"],
        "strengthLabel": strength_label(img, final_action),
        "timingQuality": timing["timingQuality"],
        "idealEntryType": timing["idealEntryType"],

        "continuationHealth": round(img.continuation_health if ctx.context_reliable else 0.0, 4),
        "reversalPressure": round(img.reversal_pressure, 4),
        "structuralWeakness": round(img.structural_weakness, 4),
        "fakeMoveRisk": round(img.fake_move_risk, 4),

        "imageQuality": img.quality_label,
        "imageNotes": img.notes,
        "contextWarnings": ctx.warnings,

        "debug": {
            "appVersion": APP_VERSION,
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
            "payoutIgnored": True,
            "binaryBrokerMode": False,
            "chartBaseSec": ctx.chart_base_sec,
            "safetyMode": "TRADINGVIEW_SMART_CONTEXT",
            "waitMeansNoEntry": True,
            "buySellOnlyWithReliableContext": True,
            "priceMustBeReliable": True,
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
        "mode": "TRADINGVIEW_SMART_CONTEXT",
        "payoutIgnored": True,
        "binaryBrokerMode": False,
        "waitMeansNoEntry": True,
        "buySellOnlyWithReliableContext": True,
        "priceMustBeReliable": True,
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
        "mode": "TRADINGVIEW_SMART_CONTEXT",
        "payoutIgnored": True,
        "binaryBrokerMode": False,
        "waitMeansNoEntry": True,
        "buySellOnlyWithReliableContext": True,
        "priceMustBeReliable": True,
    }


@app.websocket(WS_PATH)
async def ws_sinais(websocket: WebSocket):
    await websocket.accept()

    await websocket.send_text(json.dumps({
        "ok": True,
        "type": "connected",
        "message": f"Cliente conectado no {WS_PATH}",
        "serverTimestamp": now_iso(),
        "version": APP_VERSION,
        "mode": "TRADINGVIEW_SMART_CONTEXT",
        "payoutIgnored": True,
        "binaryBrokerMode": False,
        "waitMeansNoEntry": True,
        "buySellOnlyWithReliableContext": True,
        "priceMustBeReliable": True,
    }, ensure_ascii=False))

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
                }, ensure_ascii=False))
                continue

            command = safe_str(
                payload.get("command")
                or payload.get("comando")
                or "analisar_grafico"
            )

            if command and command not in ("analisar_grafico", "analyze_chart", "analyze"):
                await websocket.send_text(json.dumps({
                    "ok": False,
                    "type": "error",
                    "message": f"Comando não suportado: {command}",
                    "serverTimestamp": now_iso(),
                }, ensure_ascii=False))
                continue

            try:
                result = build_response(payload)
                result["latencyMs"] = int((time.time() - started) * 1000)
                await websocket.send_text(json.dumps(result, ensure_ascii=False))
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "ok": False,
                    "type": "error",
                    "message": f"Erro interno ao analisar: {str(e)}",
                    "serverTimestamp": now_iso(),
                    "version": APP_VERSION,
                }, ensure_ascii=False))

    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import json
import os
import time
import base64
import io
from typing import Optional, Dict, Any

try:
    from PIL import Image, ImageStat
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

app = FastAPI(title="ChartLens Live Backend")

connections = set()


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "ChartLens Live backend online",
        "websocket": "/ws/sinais"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "connections": len(connections),
        "pillow": PIL_AVAILABLE
    }


@app.websocket("/ws/sinais")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.add(websocket)
    print("Cliente conectado no /ws/sinais")

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                payload = json.loads(raw)
            except Exception:
                await websocket.send_text(json.dumps(build_error_response("JSON inválido")))
                continue

            response = process_payload(payload)
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("Cliente desconectado")
    except Exception as e:
        print(f"Erro no websocket: {e}")
    finally:
        if websocket in connections:
            connections.remove(websocket)


def process_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    comando = str(payload.get("comando", "")).strip().lower()

    if comando != "analisar_grafico":
        return build_error_response("Comando inválido ou ausente")

    image_base64 = payload.get("imageBase64")
    width = to_int(payload.get("width"), 0)
    height = to_int(payload.get("height"), 0)

    image_info = analyze_image_base64(image_base64, width, height)
    context = extract_context(payload)
    timing = infer_timing(context)
    market = infer_market_state(image_info, context)
    action_pack = infer_action(market, timing, context)

    return {
        "sinal": action_pack["signal"],
        "status": action_pack["status"],
        "confidence": action_pack["confidence"],
        "risk": action_pack["risk"],
        "explanation": action_pack["explanation"],

        "asset": context["asset"],
        "currentPrice": context["currentPrice"],
        "candleTimeRemaining": context["candleTimeRemaining"],
        "chartClock": context["chartClock"],
        "timeframe": context["timeframe"],

        "payoutPercent": context["payoutPercent"],
        "selectedOperationSec": context["selectedOperationSec"],

        "marketState": market["marketState"],
        "nextMovePrediction": market["nextMovePrediction"],

        "action": action_pack["action"],
        "instruction": action_pack["instruction"],
        "entryTiming": timing["entryTiming"],
        "secondsToAction": timing["secondsToAction"],

        "continuationProbability": action_pack["continuationProbability"],
        "reversalProbability": action_pack["reversalProbability"],

        "primaryTrend": market["primaryTrend"],
        "microTrend": market["microTrend"],
        "detectedPattern": market["detectedPattern"],
        "trigger": market["trigger"],
        "invalidation": market["invalidation"],
        "waitCandles": market["waitCandles"],

        "recommendedOperationSec": timing["recommendedOperationSec"],
        "recommendedOperationLabel": timing["recommendedOperationLabel"],

        "serverTimestamp": int(time.time())
    }


def extract_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    asset = clean_str(payload.get("asset")) or "ATIVO"
    current_price = clean_str(payload.get("currentPrice")) or ""
    candle_time_remaining = clean_str(payload.get("candleTimeRemaining")) or ""
    chart_clock = clean_str(payload.get("chartClock")) or ""
    timeframe = (clean_str(payload.get("timeframe")) or "M1").upper()

    payout_percent = optional_int(payload.get("payoutPercent"))
    selected_operation_sec = optional_int(payload.get("selectedOperationSec"))

    return {
        "asset": asset,
        "currentPrice": current_price,
        "candleTimeRemaining": candle_time_remaining,
        "chartClock": chart_clock,
        "timeframe": timeframe,
        "payoutPercent": payout_percent,
        "selectedOperationSec": selected_operation_sec
    }


def analyze_image_base64(image_base64: Optional[str], width: int, height: int) -> Dict[str, Any]:
    result = {
        "ok": False,
        "brightness": 0.0,
        "contrast": 0.0,
        "dominantBias": "neutral",
        "greenRatio": 0.0,
        "redRatio": 0.0,
        "imageQuality": "unknown"
    }

    if not image_base64:
        return result

    if not PIL_AVAILABLE:
        return result

    try:
        image_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        stat = ImageStat.Stat(img)
        avg_r, avg_g, avg_b = stat.mean
        contrast = sum(stat.stddev) / 3.0

        sample = img.resize((84, 84))
        pixels = list(sample.getdata())

        green = 0
        red = 0
        active = 0
        total = len(pixels)

        for r, g, b in pixels:
            brightness = (r + g + b) / 3
            if brightness < 30:
                continue

            if g > r + 18 and g > b + 10:
                green += 1
                active += 1
            elif r > g + 18 and r > b + 10:
                red += 1
                active += 1

        green_ratio = green / active if active else 0.0
        red_ratio = red / active if active else 0.0

        dominant = "neutral"
        if green_ratio > red_ratio + 0.06:
            dominant = "bullish"
        elif red_ratio > green_ratio + 0.06:
            dominant = "bearish"

        if contrast >= 28:
            image_quality = "strong"
        elif contrast >= 18:
            image_quality = "good"
        elif contrast >= 10:
            image_quality = "medium"
        else:
            image_quality = "weak"

        result.update({
            "ok": True,
            "brightness": round((avg_r + avg_g + avg_b) / 3.0, 2),
            "contrast": round(contrast, 2),
            "dominantBias": dominant,
            "greenRatio": round(green_ratio, 4),
            "redRatio": round(red_ratio, 4),
            "imageQuality": image_quality
        })
        return result

    except Exception:
        return result


def infer_market_state(image_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    if not image_info.get("ok"):
        return {
            "marketState": "unclear",
            "nextMovePrediction": "unclear_next_move",
            "primaryTrend": "unclear",
            "microTrend": "unclear",
            "detectedPattern": "unclear_pattern",
            "trigger": "aguardar confirmação estrutural",
            "invalidation": "dados insuficientes",
            "waitCandles": 1
        }

    bias = image_info["dominantBias"]
    contrast = image_info["contrast"]
    quality = image_info["imageQuality"]
    payout = context["payoutPercent"] or 0

    if bias == "bullish":
        wait_candles = 0 if contrast >= 18 else 1
        if payout in range(1, 70):
            wait_candles = max(wait_candles, 1)
        return {
            "marketState": "bullish",
            "nextMovePrediction": "bullish_continuation",
            "primaryTrend": "bullish",
            "microTrend": "bullish",
            "detectedPattern": "bullish_continuation" if quality != "weak" else "unclear_pattern",
            "trigger": "comprar apenas se a pressão compradora continuar limpa",
            "invalidation": "perder o fundo curto ou enfraquecer a continuação",
            "waitCandles": wait_candles
        }

    if bias == "bearish":
        wait_candles = 0 if contrast >= 18 else 1
        if payout in range(1, 70):
            wait_candles = max(wait_candles, 1)
        return {
            "marketState": "bearish",
            "nextMovePrediction": "bearish_continuation",
            "primaryTrend": "bearish",
            "microTrend": "bearish",
            "detectedPattern": "bearish_continuation" if quality != "weak" else "unclear_pattern",
            "trigger": "vender apenas se a pressão vendedora continuar limpa",
            "invalidation": "romper a máxima curta ou enfraquecer a continuação",
            "waitCandles": wait_candles
        }

    return {
        "marketState": "sideways",
        "nextMovePrediction": "sideways_probable",
        "primaryTrend": "unclear",
        "microTrend": "unclear",
        "detectedPattern": "consolidation",
        "trigger": "aguardar rompimento limpo",
        "invalidation": "seguir lateral sem confirmação",
        "waitCandles": 1
    }


def infer_timing(context: Dict[str, Any]) -> Dict[str, Any]:
    candle_raw = context["candleTimeRemaining"]
    timeframe = context["timeframe"]
    selected_operation = context["selectedOperationSec"]

    recommended_operation_sec = selected_operation or timeframe_to_seconds(timeframe)
    recommended_operation_label = format_duration_label(recommended_operation_sec)

    seconds_remaining = parse_mmss_to_seconds(candle_raw)

    if seconds_remaining is None:
        return {
            "entryTiming": "tempo indefinido",
            "secondsToAction": None,
            "recommendedOperationSec": recommended_operation_sec,
            "recommendedOperationLabel": recommended_operation_label
        }

    if recommended_operation_sec <= 60:
        if seconds_remaining > 42:
            return {
                "entryTiming": "cedo para operação curta",
                "secondsToAction": max(seconds_remaining - 16, 0),
                "recommendedOperationSec": recommended_operation_sec,
                "recommendedOperationLabel": recommended_operation_label
            }
        if 22 <= seconds_remaining <= 42:
            return {
                "entryTiming": "janela em formação",
                "secondsToAction": max(seconds_remaining - 9, 0),
                "recommendedOperationSec": recommended_operation_sec,
                "recommendedOperationLabel": recommended_operation_label
            }
        if 8 <= seconds_remaining <= 21:
            return {
                "entryTiming": "janela ideal",
                "secondsToAction": 0,
                "recommendedOperationSec": recommended_operation_sec,
                "recommendedOperationLabel": recommended_operation_label
            }
        if 3 <= seconds_remaining <= 7:
            return {
                "entryTiming": "entrada tardia",
                "secondsToAction": None,
                "recommendedOperationSec": recommended_operation_sec,
                "recommendedOperationLabel": recommended_operation_label
            }
        return {
            "entryTiming": "próxima vela",
            "secondsToAction": seconds_remaining + 2,
            "recommendedOperationSec": recommended_operation_sec,
            "recommendedOperationLabel": recommended_operation_label
        }

    if seconds_remaining > 36:
        return {
            "entryTiming": "janela em formação",
            "secondsToAction": max(seconds_remaining - 12, 0),
            "recommendedOperationSec": recommended_operation_sec,
            "recommendedOperationLabel": recommended_operation_label
        }

    if 12 <= seconds_remaining <= 36:
        return {
            "entryTiming": "janela ideal",
            "secondsToAction": 0,
            "recommendedOperationSec": recommended_operation_sec,
            "recommendedOperationLabel": recommended_operation_label
        }

    return {
        "entryTiming": "próxima vela",
        "secondsToAction": seconds_remaining + 2,
        "recommendedOperationSec": recommended_operation_sec,
        "recommendedOperationLabel": recommended_operation_label
    }


def infer_action(market: Dict[str, Any], timing: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    state = market["marketState"]
    entry_timing = timing["entryTiming"]
    payout = context["payoutPercent"] or 0

    payout_penalty = 0
    payout_risk = "low"
    if payout in range(1, 60):
        payout_penalty = 18
        payout_risk = "high"
    elif payout in range(60, 70):
        payout_penalty = 10
        payout_risk = "medium"

    if state == "bullish":
        if entry_timing == "janela ideal":
            confidence = max(52, 80 - payout_penalty)
            risk = "low" if payout_risk == "low" else payout_risk
            if payout in range(1, 60):
                return build_action(
                    signal="WAIT",
                    action="NO_ENTRY_LOW_PAYOUT",
                    status="Payout baixo para compra",
                    instruction="Não entrar",
                    confidence=confidence,
                    risk="high",
                    continuation=max(45, 78 - payout_penalty),
                    reversal=min(55, 22 + payout_penalty),
                    explanation="A estrutura compradora existe, mas o payout está baixo para entrada saudável."
                )
            return build_action(
                signal="BUY",
                action="BUY",
                status="Compra com força",
                instruction="Comprar",
                confidence=confidence,
                risk=risk,
                continuation=max(50, 78 - payout_penalty),
                reversal=min(50, 22 + payout_penalty),
                explanation="Estrutura bullish com continuidade e timing favorável."
            )

        if entry_timing in ("janela em formação", "cedo para operação curta"):
            return build_action(
                signal="WAIT",
                action="WAIT_BUY_SETUP",
                status="Aguardando confirmação buy",
                instruction="Preparando compra",
                confidence=max(45, 66 - payout_penalty),
                risk="medium" if payout_risk != "high" else "high",
                continuation=max(42, 70 - payout_penalty),
                reversal=min(58, 30 + payout_penalty),
                explanation="Há viés comprador, mas a melhor janela ainda está formando."
            )

        if entry_timing == "entrada tardia":
            return build_action(
                signal="WAIT",
                action="WAIT_LATE",
                status="Entrada tardia",
                instruction="Aguardar",
                confidence=max(40, 58 - payout_penalty),
                risk="high",
                continuation=max(35, 62 - payout_penalty),
                reversal=min(65, 38 + payout_penalty),
                explanation="O cenário comprador existe, mas a entrada ficou tardia."
            )

        return build_action(
            signal="WAIT",
            action="WAIT_ONE_MORE_CANDLE",
            status="Aguardar próxima vela",
            instruction="Aguardar mais 1 vela",
            confidence=max(42, 60 - payout_penalty),
            risk="medium" if payout_risk != "high" else "high",
            continuation=max(38, 65 - payout_penalty),
            reversal=min(62, 35 + payout_penalty),
            explanation="Compra provável, mas a próxima vela tende a dar entrada mais limpa."
        )

    if state == "bearish":
        if entry_timing == "janela ideal":
            confidence = max(52, 80 - payout_penalty)
            risk = "low" if payout_risk == "low" else payout_risk
            if payout in range(1, 60):
                return build_action(
                    signal="WAIT",
                    action="NO_ENTRY_LOW_PAYOUT",
                    status="Payout baixo para venda",
                    instruction="Não entrar",
                    confidence=confidence,
                    risk="high",
                    continuation=max(45, 78 - payout_penalty),
                    reversal=min(55, 22 + payout_penalty),
                    explanation="A estrutura vendedora existe, mas o payout está baixo para entrada saudável."
                )
            return build_action(
                signal="SELL",
                action="SELL",
                status="Venda com força",
                instruction="Vender",
                confidence=confidence,
                risk=risk,
                continuation=max(50, 78 - payout_penalty),
                reversal=min(50, 22 + payout_penalty),
                explanation="Estrutura bearish com continuidade e timing favorável."
            )

        if entry_timing in ("janela em formação", "cedo para operação curta"):
            return build_action(
                signal="WAIT",
                action="WAIT_SELL_SETUP",
                status="Aguardando confirmação sell",
                instruction="Preparando venda",
                confidence=max(45, 66 - payout_penalty),
                risk="medium" if payout_risk != "high" else "high",
                continuation=max(42, 70 - payout_penalty),
                reversal=min(58, 30 + payout_penalty),
                explanation="Há viés vendedor, mas a melhor janela ainda está formando."
            )

        if entry_timing == "entrada tardia":
            return build_action(
                signal="WAIT",
                action="WAIT_LATE",
                status="Entrada tardia",
                instruction="Aguardar",
                confidence=max(40, 58 - payout_penalty),
                risk="high",
                continuation=max(35, 62 - payout_penalty),
                reversal=min(65, 38 + payout_penalty),
                explanation="O cenário vendedor existe, mas a entrada ficou tardia."
            )

        return build_action(
            signal="WAIT",
            action="WAIT_ONE_MORE_CANDLE",
            status="Aguardar próxima vela",
            instruction="Aguardar mais 1 vela",
            confidence=max(42, 60 - payout_penalty),
            risk="medium" if payout_risk != "high" else "high",
            continuation=max(38, 65 - payout_penalty),
            reversal=min(62, 35 + payout_penalty),
            explanation="Venda provável, mas a próxima vela tende a dar entrada mais limpa."
        )

    return build_action(
        signal="WAIT",
        action="WAIT_LATERAL",
        status="Mercado lateral",
        instruction="Aguardar",
        confidence=48,
        risk="high",
        continuation=40,
        reversal=35,
        explanation="Ainda não há direção limpa suficiente para entrada."
    )


def build_action(
    signal: str,
    action: str,
    status: str,
    instruction: str,
    confidence: int,
    risk: str,
    continuation: int,
    reversal: int,
    explanation: str
) -> Dict[str, Any]:
    return {
        "signal": signal,
        "action": action,
        "status": status,
        "instruction": instruction,
        "confidence": max(0, min(100, confidence)),
        "risk": risk,
        "continuationProbability": max(0, min(100, continuation)),
        "reversalProbability": max(0, min(100, reversal)),
        "explanation": explanation
    }


def build_error_response(message: str) -> Dict[str, Any]:
    return {
        "sinal": "WAIT",
        "status": message,
        "confidence": 0,
        "risk": "high",
        "explanation": message,
        "asset": "",
        "currentPrice": "",
        "candleTimeRemaining": "",
        "chartClock": "",
        "timeframe": "",
        "payoutPercent": None,
        "selectedOperationSec": None,
        "marketState": "unclear",
        "nextMovePrediction": "unclear_next_move",
        "action": "DATA_INSUFFICIENT",
        "instruction": "Dados insuficientes para entrada",
        "entryTiming": "tempo indefinido",
        "secondsToAction": None,
        "continuationProbability": 0,
        "reversalProbability": 0,
        "primaryTrend": "unclear",
        "microTrend": "unclear",
        "detectedPattern": "unclear_pattern",
        "trigger": "aguardar confirmação estrutural",
        "invalidation": "dados insuficientes",
        "waitCandles": 1,
        "recommendedOperationSec": 60,
        "recommendedOperationLabel": "1 min",
        "serverTimestamp": int(time.time())
    }


def parse_mmss_to_seconds(value: str) -> Optional[int]:
    if not value or ":" not in value:
        return None
    parts = value.split(":")
    if len(parts) != 2:
        return None
    mm = to_int(parts[0], -1)
    ss = to_int(parts[1], -1)
    if mm < 0 or ss < 0:
        return None
    return mm * 60 + ss


def timeframe_to_seconds(timeframe: str) -> int:
    mapping = {
        "M1": 60,
        "M2": 120,
        "M3": 180,
        "M5": 300,
        "M10": 600,
        "M15": 900
    }
    return mapping.get(timeframe.upper(), 60)


def format_duration_label(seconds: int) -> str:
    if seconds % 60 == 0:
        return f"{seconds // 60} min"
    return f"{seconds}s"


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def optional_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

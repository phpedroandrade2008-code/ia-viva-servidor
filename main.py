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

    market = infer_market_state(image_info)
    timing = infer_timing(payload)
    action_pack = infer_action(market, timing)

    return {
        "sinal": action_pack["signal"],
        "status": action_pack["status"],
        "confidence": action_pack["confidence"],
        "risk": action_pack["risk"],
        "explanation": action_pack["explanation"],

        "asset": str(payload.get("asset", "ATIVO")),
        "currentPrice": str(payload.get("currentPrice", "")),
        "candleTimeRemaining": str(payload.get("candleTimeRemaining", "")),
        "chartClock": str(payload.get("chartClock", "")),
        "timeframe": str(payload.get("timeframe", "M1")),

        "payoutPercent": optional_int(payload.get("payoutPercent")),
        "selectedOperationSec": optional_int(payload.get("selectedOperationSec")),

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


def analyze_image_base64(image_base64: Optional[str], width: int, height: int) -> Dict[str, Any]:
    result = {
        "ok": False,
        "brightness": 0.0,
        "contrast": 0.0,
        "dominantBias": "neutral",
        "greenRatio": 0.0,
        "redRatio": 0.0
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

        sample = img.resize((96, 96))
        pixels = list(sample.getdata())

        green = 0
        red = 0
        total = len(pixels)

        for r, g, b in pixels:
            if g > r + 18 and g > b + 10:
                green += 1
            elif r > g + 18 and r > b + 10:
                red += 1

        green_ratio = green / total if total else 0.0
        red_ratio = red / total if total else 0.0

        dominant = "neutral"
        if green_ratio > red_ratio + 0.03:
            dominant = "bullish"
        elif red_ratio > green_ratio + 0.03:
            dominant = "bearish"

        result.update({
            "ok": True,
            "brightness": round((avg_r + avg_g + avg_b) / 3.0, 2),
            "contrast": round(contrast, 2),
            "dominantBias": dominant,
            "greenRatio": round(green_ratio, 4),
            "redRatio": round(red_ratio, 4),
        })
        return result

    except Exception:
        return result


def infer_market_state(image_info: Dict[str, Any]) -> Dict[str, Any]:
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

    if bias == "bullish":
        return {
            "marketState": "bullish",
            "nextMovePrediction": "bullish_continuation",
            "primaryTrend": "bullish",
            "microTrend": "bullish",
            "detectedPattern": "bullish_continuation",
            "trigger": "comprar apenas se a pressão compradora continuar limpa",
            "invalidation": "perder o fundo curto ou enfraquecer a continuação",
            "waitCandles": 0 if contrast >= 18 else 1
        }

    if bias == "bearish":
        return {
            "marketState": "bearish",
            "nextMovePrediction": "bearish_continuation",
            "primaryTrend": "bearish",
            "microTrend": "bearish",
            "detectedPattern": "bearish_continuation",
            "trigger": "vender apenas se a pressão vendedora continuar limpa",
            "invalidation": "romper a máxima curta ou enfraquecer a continuação",
            "waitCandles": 0 if contrast >= 18 else 1
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


def infer_timing(payload: Dict[str, Any]) -> Dict[str, Any]:
    candle_raw = str(payload.get("candleTimeRemaining", "")).strip()
    timeframe = str(payload.get("timeframe", "M1")).strip().upper()
    selected_operation = optional_int(payload.get("selectedOperationSec"))

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
        if seconds_remaining > 38:
            return {
                "entryTiming": "cedo para operação curta",
                "secondsToAction": max(seconds_remaining - 15, 0),
                "recommendedOperationSec": recommended_operation_sec,
                "recommendedOperationLabel": recommended_operation_label
            }
        if 21 <= seconds_remaining <= 38:
            return {
                "entryTiming": "janela em formação",
                "secondsToAction": max(seconds_remaining - 8, 0),
                "recommendedOperationSec": recommended_operation_sec,
                "recommendedOperationLabel": recommended_operation_label
            }
        if 8 <= seconds_remaining <= 20:
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

    if seconds_remaining > 35:
        return {
            "entryTiming": "janela em formação",
            "secondsToAction": max(seconds_remaining - 12, 0),
            "recommendedOperationSec": recommended_operation_sec,
            "recommendedOperationLabel": recommended_operation_label
        }

    if 12 <= seconds_remaining <= 35:
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


def infer_action(market: Dict[str, Any], timing: Dict[str, Any]) -> Dict[str, Any]:
    state = market["marketState"]
    entry_timing = timing["entryTiming"]

    if state == "bullish":
        if entry_timing == "janela ideal":
            return build_action(
                signal="BUY",
                action="BUY",
                status="Compra com força",
                instruction="Comprar",
                confidence=78,
                risk="low",
                continuation=78,
                reversal=22,
                explanation="Estrutura bullish com continuidade e timing favorável."
            )
        if entry_timing in ("janela em formação", "cedo para operação curta"):
            return build_action(
                signal="WAIT",
                action="WAIT_BUY_SETUP",
                status="Aguardando confirmação buy",
                instruction="Preparando compra",
                confidence=66,
                risk="medium",
                continuation=70,
                reversal=30,
                explanation="Há viés comprador, mas a melhor janela ainda está formando."
            )
        if entry_timing == "entrada tardia":
            return build_action(
                signal="WAIT",
                action="WAIT_LATE",
                status="Entrada tardia",
                instruction="Aguardar",
                confidence=58,
                risk="high",
                continuation=62,
                reversal=38,
                explanation="O cenário comprador existe, mas a entrada ficou tardia."
            )
        return build_action(
            signal="WAIT",
            action="WAIT_ONE_MORE_CANDLE",
            status="Aguardar próxima vela",
            instruction="Aguardar mais 1 vela",
            confidence=60,
            risk="medium",
            continuation=65,
            reversal=35,
            explanation="Compra provável, mas a próxima vela tende a dar entrada mais limpa."
        )

    if state == "bearish":
        if entry_timing == "janela ideal":
            return build_action(
                signal="SELL",
                action="SELL",
                status="Venda com força",
                instruction="Vender",
                confidence=78,
                risk="low",
                continuation=78,
                reversal=22,
                explanation="Estrutura bearish com continuidade e timing favorável."
            )
        if entry_timing in ("janela em formação", "cedo para operação curta"):
            return build_action(
                signal="WAIT",
                action="WAIT_SELL_SETUP",
                status="Aguardando confirmação sell",
                instruction="Preparando venda",
                confidence=66,
                risk="medium",
                continuation=70,
                reversal=30,
                explanation="Há viés vendedor, mas a melhor janela ainda está formando."
            )
        if entry_timing == "entrada tardia":
            return build_action(
                signal="WAIT",
                action="WAIT_LATE",
                status="Entrada tardia",
                instruction="Aguardar",
                confidence=58,
                risk="high",
                continuation=62,
                reversal=38,
                explanation="O cenário vendedor existe, mas a entrada ficou tardia."
            )
        return build_action(
            signal="WAIT",
            action="WAIT_ONE_MORE_CANDLE",
            status="Aguardar próxima vela",
            instruction="Aguardar mais 1 vela",
            confidence=60,
            risk="medium",
            continuation=65,
            reversal=35,
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
        "confidence": confidence,
        "risk": risk,
        "continuationProbability": continuation,
        "reversalProbability": reversal,
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

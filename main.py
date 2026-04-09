from fastapi import FastAPI, WebSocket
import uvicorn
import asyncio
import json

app = FastAPI()

# Lista de conexões ativas (celulares conectados)
connections = []

@app.get("/")
async def root():
    return {"status": "IA Operando em Tempo Real"}

@app.websocket("/ws/sinais")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    print("Celular conectado para receber sinais!")
    try:
        while True:
            # Aqui a IA recebe os dados do gráfico
            data = await websocket.receive_text()
            dados_recebidos = json.loads(data)
            
            # SIMULAÇÃO DA LÓGICA DA IA (Aqui entrará sua lógica de análise)
            # Em vez de esperar 4.5 segundos, aqui processamos em milissegundos
            resposta = {
                "sinal": "AGUARDAR",
                "paridade": dados_recebidos.get("asset", "EURUSD"),
                "status": "Analisando fluxo vivo..."
            }
            
            # Se a IA identificar uma entrada, ela "grita" pro Flutter na hora
            await websocket.send_text(json.dumps(resposta))
            
    except Exception as e:
        print(f"Conexão encerrada: {e}")
    finally:
        connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
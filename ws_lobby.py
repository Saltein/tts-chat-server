import asyncio
import websockets
import random
import string
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

rooms = {}  # {код: {'host': websocket, 'clients': set(websocket)}}

def generate_room_code(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

async def create_room(websocket):
    while True:
        code = generate_room_code()
        if code not in rooms:
            rooms[code] = {'host': websocket, 'clients': set()}
            logging.info(f"Создана комната {code}")
            return code

async def handler(websocket):
    room_code = None
    is_host = False
    
    try:
        # Ждём команду "create" или "join:<код>"
        first_msg = await websocket.recv()

        if first_msg.strip().lower() == "create":
            code = await create_room(websocket)
            room_code = code
            is_host = True
            # Отправляем JSON вместо plain text
            await websocket.send(json.dumps({'type': 'room_created', 'code': code}))
            logging.info(f"Создана комната {code}, хост подключён")

        elif first_msg.startswith("join:"):
            code = first_msg.split("join:")[1].strip().upper()
            if code not in rooms:
                await websocket.send(json.dumps({'type': 'error', 'message': 'room_not_found'}))
                return
            
            rooms[code]['clients'].add(websocket)
            room_code = code
            is_host = False
            # Отправляем JSON вместо plain text
            await websocket.send(json.dumps({'type': 'joined', 'code': code}))
            
            # Уведомляем хост о новом подключении
            host_websocket = rooms[code]['host']
            if host_websocket.open:
                await host_websocket.send(json.dumps({
                    'type': 'client_connected', 
                    'clients_count': len(rooms[code]['clients'])
                }))
            
            logging.info(f"Клиент подключился к комнате {code}")

        else:
            await websocket.send(json.dumps({'type': 'error', 'message': 'unknown_command'}))
            return

        # Основной цикл общения в комнате
        async for message in websocket:
            if room_code not in rooms:
                break
            
                # Если пользователь отправил команду "leave"
            if message.strip().lower() == "leave":
                if is_host:
                    # Удаляем комнату и уведомляем клиентов
                    if room_code in rooms:
                        room = rooms.pop(room_code)
                        for client in room['clients']:
                            if client.open:
                                await client.send(json.dumps({
                                    'type': 'room_closed',
                                    'message': 'host_left'
                                }))
                        logging.info(f"Хост покинул комнату {room_code}, комната удалена")
                else:
                    # Клиент просто уходит
                    if room_code in rooms:
                        room = rooms[room_code]
                        room['clients'].discard(websocket)
                        logging.info(f"Клиент покинул комнату {room_code}")
                        host_websocket = room['host']
                        if host_websocket.open:
                            await host_websocket.send(json.dumps({
                                'type': 'client_disconnected',
                                'clients_count': len(room['clients'])
                            }))
                break
                
            # Пытаемся распарсить JSON от хоста
            try:
                data = json.loads(message)
                logging.info(f"[{room_code}] получено: {data}")
                
                # Если это хост - рассылаем данные всем клиентам
                if is_host:
                    room = rooms[room_code]
                    disconnected_clients = []
                    
                    for client in list(room['clients']):
                        if client.open:
                            try:
                                await client.send(json.dumps({
                                    'type': 'data',
                                    'payload': data
                                }))
                            except websockets.ConnectionClosed:
                                disconnected_clients.append(client)
                        else:
                            disconnected_clients.append(client)
                    
                    # Удаляем отключившихся клиентов
                    for client in disconnected_clients:
                        room['clients'].discard(client)
                        
            except json.JSONDecodeError:
                logging.warning(f"Получено невалидное JSON сообщение: {message}")
                await websocket.send(json.dumps({'type': 'error', 'message': 'invalid_json_format'}))

    except websockets.ConnectionClosed:
        logging.info(f"Соединение закрыто для комнаты {room_code}")
    except Exception as e:
        logging.error(f"Ошибка: {e}")
    finally:
        # Убираем отключившегося клиента
        if is_host:
            room = rooms[room_code]
            disconnected_clients = []
            logging.info(f"[{room_code}] Рассылаем {len(room['clients'])} клиентам: {data}")
            
            for client in list(room['clients']):
                logging.info(f"  ➜ Проверяем клиента: open={client.open}")
                if client.open:
                    try:
                        await client.send(json.dumps({
                            'type': 'data',
                            'payload': data
                        }))
                        logging.info(f"  ✅ Отправлено клиенту")
                    except websockets.ConnectionClosed:
                        logging.warning(f"  ⚠️ Клиент отключён (ConnectionClosed)")
                        disconnected_clients.append(client)
                else:
                    logging.warning(f"  ⚠️ client.open = False")
                    disconnected_clients.append(client)

                # Клиент отключился
                if websocket in rooms[room_code]['clients']:
                    rooms[room_code]['clients'].remove(websocket)
                    logging.info(f"Клиент вышел из комнаты {room_code}")
                    
                    # Уведомляем хост об отключении клиента
                    host_websocket = rooms[room_code]['host']
                    if host_websocket.open:
                        await host_websocket.send(json.dumps({
                            'type': 'client_disconnected',
                            'clients_count': len(rooms[room_code]['clients'])
                        }))

async def main():
    async with websockets.serve(handler, "0.0.0.0", 6789, ping_interval=None):
        logging.info("✅ WebSocket сервер запущен на ws://0.0.0.0:6789")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
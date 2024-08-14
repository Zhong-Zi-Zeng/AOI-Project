import socketio

sio = socketio.Client()


@sio.event
def connect():
    print('Connected to server')


@sio.on('status_update')
def on_status_update(data):
    print('Status update received:', data)


@sio.event
def disconnect():
    print('Disconnected from server')


sio.connect('http://localhost:5000')
sio.wait()

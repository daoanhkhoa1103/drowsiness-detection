from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)
@app.route('/')
def index():
    return "Hello Flask-SocketIO!"
if __name__ == '__main__':
    import eventlet
    eventlet.monkey_patch()
    socketio.run(app, host="0.0.0.0", port=8000)

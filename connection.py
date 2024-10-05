import socket
import threading

def loop(listen, host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Bind the socket to the host and port
        server_socket.bind((host, port))
        # Listen for incoming connections
        server_socket.listen()

        while True:
            # Accept a new connection
            client_socket, client_address = server_socket.accept()
            with client_socket:
                while True:
                    print('ch')
                    # Receive data from the client
                    data = client_socket.recv(1024)
                    if not data:
                        break  # Connection closed
                    listen(data.decode())

def start_server(listen, host='0.0.0.0', port=12345):
    # Create a TCP socket
    thr = threading.Thread(target=lambda:loop(listen, host, port))
    thr.daemon = True
    thr.start()

class Pressures:
    def __init__(self):
        start_server(lambda x: self.add_pressure(x))
        self.pressures = []

    def loop(self):
        while True:
            print(self.pressures)
            pass

    def get_pressure(self):
        if len(self.pressures) == 0:
            return 'E'
        else:
            return self.pressures[len(self.pressures) - 1]

    def add_pressure(self, x):
        self.pressures.append(x)

if __name__ == '__main__':
    p = Pressures()
    p.loop()


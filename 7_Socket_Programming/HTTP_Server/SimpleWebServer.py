#!/usr/bin/env python3

from socket import *

def SimpleHTTPServer():
    HOST = 'localhost'
    PORT = 80
    BUFFER = 1024

    #Setup port to serve on, host name, socket, socket optioins, socket binding
    server_socket = socket(AF_INET, SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    
    #Listen for 1 connection
    server_socket.listen(1)

    #Print server ready message
    print(f'Local HTTP Server Ready. Listening on Port 80. Visit http://localhost/tester.html ...')

    while True:
        connection, address = server_socket.accept()
        try:
            data = connection.recv(BUFFER)
            print(data)
            fl = data.split()[1]

            f = open(fl[1:])
            out = f.read()
            print(out)
            OK = 'HTTP/1.1 200 OK\r\n'
            connection.sendall(OK.encode())
            connection.close()
        
        except IOError:
            err = "HTTP/1.1 404 Not Found\r\n"
            connection.sendall(err.encode())
            connection.close()
        
if __name__ == "__main__":
    SimpleHTTPServer()
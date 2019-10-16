#!/usr/bin/env python3
import base64, getpass, socket, ssl, sys 

# Set variables
SERVER = 'smtp.gmail.com'
PORT = 465

# Connect to mail server
mail_client = ssl.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM), ssl_version=ssl.PROTOCOL_SSLv23)
mail_client.connect((SERVER, PORT))

#Print response
response = mail_client.recv(1024)
print(str(response))

# Init HELO to mail server
helo = 'HELO Bryan\r\n'
print(helo)
mail_client.send(helo.encode('utf-8'))
response = mail_client.recv(1024)
print(str(response))

# authenticate
username = '***********'
password = '***********'
authMesg = 'AUTH LOGIN\r\n'
crlfMesg = '\r\n'
print(authMesg)
mail_client.send(authMesg.encode('utf-8'))
response = mail_client.recv(1024)
print(str(response))
user64 = base64.b64encode(username.encode('utf-8'))
pass64 = base64.b64encode(password.encode('utf-8'))
print(user64)
mail_client.send(user64)
mail_client.send(crlfMesg.encode('utf-8'))
response = mail_client.recv(1024)
print(str(response))
print(pass64)
mail_client.send(pass64)
mail_client.send(crlfMesg.encode('utf-8'))
response = mail_client.recv(1024)
print(str(response))

# Provide server with sender info
sender = 'MAIL FROM: <bryanwnonni@gmail.com>\r\n'
print(sender)
mail_client.send(sender.encode('utf-8'))
response = mail_client.recv(1024)
print(str(response))

# Provide server with recipient info
recipient = 'RCPT TO: <me@bryanwnonni.com>\r\n'
print(recipient)
mail_client.send(recipient.encode('utf-8'))
response = mail_client.recv(1024)
print(str(response))

# Send mail message to server + subject
data_msg = 'DATA\r\n'
print(data_msg)
mail_client.send(data_msg.encode('utf-8'))
response = mail_client.recv(1024)
print(str(response))
subject = 'Subject: $3nt via #python#\r\n'
print(subject)
body = '#3ll0 $3lf^! (*)!\r\n'
print(body)
mail_client.send(subject.encode('utf-8'))
mail_client.send(body.encode('utf-8'))

# Send STOP msg
STOP = '\r\n.\r\n'
print(STOP)
mail_client.send(STOP.encode('utf-8'))
response = mail_client.recv(1024)
print(str(response))

# Send quit signal to server
QUIT = 'QUIT\r\n'
print(QUIT)
mail_client.send(QUIT.encode('utf-8'))
response = mail_client.recv(1024)
print(str(response))

# Close socket
mail_client.close()
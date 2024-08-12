import socket

try:
    local_hostname = socket.gethostname()
    print(local_hostname)

except:
    local_hostname = None


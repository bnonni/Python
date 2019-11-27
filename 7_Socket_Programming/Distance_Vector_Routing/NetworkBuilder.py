import os
import sys
import subprocess
import time


def buildNetwork(args):
    if len(args) < 2:
        return exit("ERR: Please pass filepath to the router .dat files.")

    router_dat_path = args[1]
    if not os.path.exists(router_dat_path):
        return exit("ERR: Directory does not exist.")

    router_dats = os.listdir(router_dat_path)
    network_size = len(router_dats)
    ports = []
    nodes = ""

    print("Starting network with " + network_size.__str__() + " routers.")

    for i in range(network_size):
        tmp = router_dats[i][0]
        print("Select port in range 1025-65535 for router " + tmp + ":")
        dats = True

        while dats:
            try:
                port = int(input())
                ports.append(port)
                dats = False
            except Exception as e:
                return exit(str(e))
        nodes = nodes + tmp + ":" + str(ports[i])
        if i != len(router_dats) - 1:
            nodes = nodes + "-"

    for i in range(network_size):
        cwd = os.getcwd()
        path = router_dat_path + router_dats[i]
        subprocess.call(['./Terminal.scpt', f'{cwd}/Router.py ' + str(i+1) + ' ' + cwd + '/' + path + ' ' +
                         str(network_size) + ' ' + nodes])
        # os.system('sudo python3 Router.py ' + str(i+1) + ' ' + path + ' ' +
        #           str(network_size) + ' ' + nodes)
        time.sleep(2)


if __name__ == "__main__":
    buildNetwork(sys.argv)

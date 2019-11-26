#!/usr/bin/env python3
import os
import sys
import subprocess
import time


def buildNetwork():
    if len(sys.argv) < 2:
        return exit("ERR: Please pass filepath to router data files.")

    router_conf_path = sys.argv[1]
    if not os.path.exists(router_conf_path):
        return exit("ERR: Directory does not exist.")

    router_confs = os.listdir(router_conf_path)
    network_size = len(router_confs)
    ports = []
    nodes = ""

    print("Starting network with " + network_size.__str__() + " routers.")

    for i in range(network_size):
        tmp = router_confs[i]
        print("Select port in range 1025-65535 for router " + tmp[0] + ":")
        flag = False

        while not flag:
            try:
                p = int(input())
                ports.append(p)
                flag = True
            except Exception as e:
                return exit(str(e))
        nodes = nodes + tmp[0] + ":" + str(ports[i])
        if i + 1 != len(router_confs):
            nodes = nodes + "-"

    print(nodes)

    for i in range(network_size):
        path = router_conf_path + "/" + router_confs[i]
        print(path)
        print(subprocess.call(['./Terminal', path, nodes], shell=True))
        time.sleep(2)


if __name__ == "__main__":
    buildNetwork()

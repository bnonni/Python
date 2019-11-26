#!/usr/bin/env python3
from LinkedList import *


def reverse(list):

    # Initializing values
    prev = None
    curr = list.head
    nex = curr.getNextNode()

    # looping
    while curr:
        # reversing the link
        curr.setNextNode(prev)

        # moving to next node
        prev = curr
        curr = nex
        if nex:
            nex = nex.getNextNode()

    # initializing head
    list.head = prev


def init():
    LinkList = LinkedList()
    print("Inserting")
    print(LinkList.addNode(5))
    print(LinkList.addNode(15))
    print(LinkList.addNode(25))
    print("Printing list: ")
    print(LinkList.printNode())
    print("Size", LinkList.getSize())
    print("Reverse list: ")
    print(reverse(LinkList))


init()

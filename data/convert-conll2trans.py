import sys
import os
action = []
buffer = []
for line in sys.stdin:
    if line.strip() != "":
        line = line.strip().split(" ")
        word = line[0] + "$" + line[1]
        for i in range(4,len(line)):
            word = word + "$" + line[i]
        buffer.append(word)
        if line[3] == "O":
            action.append("OUT")
        else:
            action.append("SHIFT")

    else:
        assert(len(buffer) == len(action))
        if len(action) != 0:
            print " ".join(buffer).strip() + " ||| " + " ".join(action).strip()
        action = []
        buffer = []



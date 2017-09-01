import sys
out = ""
for line in sys.stdin:
    if line.strip() != "":
        line = line.strip().split(" ")
        out = out + line[0] + "/" + line[1] + "/" + line[3] + " "
    else:
        if out !="":
            print out.strip()
        out = ""


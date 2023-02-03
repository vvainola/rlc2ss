import sys

lines = open(sys.argv[1], "r").readlines()
lines.sort()
f = open(sys.argv[1], "w").writelines(lines)

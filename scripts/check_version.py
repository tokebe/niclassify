import sys

if sys.version_info[0] != 3:
    sys.exit(-1)
elif sys.version_info[1] > 8 or sys.version_info[1] < 6:
    sys.exit(-1)
else:
    sys.exit(0)

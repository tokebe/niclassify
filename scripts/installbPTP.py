#! python
import subprocess
from os import path

subprocess.run(
    '"{}" "{}" install'.format(
        "python",
        path.join(path.dirname(path.abspath(__file__)),
                  "niclassify/bin/PTP-master/setup.py")
    ),
    cwd=path.join(path.dirname(
        path.abspath((__file__))), "niclassify/bin/PTP-master"),
    shell=True
)

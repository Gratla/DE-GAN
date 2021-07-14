import subprocess
import sys
import os
import glob

mode = sys.argv[1]
inputPath = sys.argv[2]
outputPath = sys.argv[3]


def enhanceAll(inputPath, outputPath):
    for f in glob.glob(inputPath + '/*.png'):
        print("Start enhancement of " + f)
        subprocess.call(["py", "enhance.py", mode, f, outputPath + '/' + mode + '_' + os.path.basename(f)],
                                   stdout=subprocess.PIPE)

enhanceAll(inputPath, outputPath)

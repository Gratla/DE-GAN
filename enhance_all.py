import subprocess
import sys
import os
import glob

# This script calls the enhance.py script to enhances all images with the configured weights and settings (DE-GAN script).
# usage: py enhance_all.py <mode> ./input/folder/path ./output/folder/path
# example: py enhance_all.py S;T;epoch100batchsize32_msi_bin ./input/folder/path ./output/folder/path

mode = sys.argv[1]
inputPath = sys.argv[2]
outputPath = sys.argv[3]

def enhanceAll(inputPath, outputPath):
    for f in glob.glob(inputPath + '/*.png'):
        print("Start enhancement of " + f)
        subprocess.call(["py", "enhance.py", mode, f, outputPath + '/' + mode + '_' + os.path.basename(f)],
                                   stdout=subprocess.PIPE)

enhanceAll(inputPath, outputPath)

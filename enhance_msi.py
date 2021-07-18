import glob
import os
import subprocess
from pathlib import Path
from time import time

import click

from invert_images import invertImage
from msi_pca import executePCA

pcaFolder = "/pca"
pcaFirstComponentExtension = "pca00.png"
pcaInvertedFolder = "/pcaInverted"
pcaFirstFolder = "/pcaFirstDeganSecond"
deganFolder = "/degan"
deganFirstFolder = "/deganFirstPcaSecond"
deganFirstInvertedFolder = "/deganFirstPcaSecondInverted"
deganMode = "epoch114batchsize64_msi_bin"

@click.command()
@click.argument('mode')
@click.argument('msiname')
@click.argument('msipath')
@click.argument('outputpath')
def enhanceMSI(mode, msiname, msipath, outputpath):
    print("Mode: " + mode)
    outputPath = outputpath + '/' + msiname
    Path(outputPath).mkdir(parents=True, exist_ok=True)

    t0 = time()
    if mode == "pca":
        pca(msiname, msipath, outputPath)
    elif mode == "pcaFirst":
        enhancePCAFirst(msiname, msipath, outputPath)
    elif mode == "deganFirst":
        enhanceDEGANFirst(msiname, msipath, outputPath)

    timeNeeded = (time() - t0)
    print("Done in %0.3fs" % timeNeeded)


# Executes a PCA on the given msi which needs to have 12 images
def pca(msiName, msiPath, outputPath):
    executePCA(msiName, msiPath, outputPath, 12)


# Executes a PCA on the input msi, inverts the first component of the result
# and feeds the first component into the DE-GAN
def enhancePCAFirst(msiName, msiPath, outputPath):
    # Execute PCA
    executePCA(msiName, msiPath, outputPath + pcaFolder, 12)

    # Invert first Component
    Path(outputPath + pcaInvertedFolder).mkdir(parents=True, exist_ok=True)
    invertImage(outputPath + pcaFolder + '/' + msiName + '_' + pcaFirstComponentExtension,
                outputPath + pcaInvertedFolder + '/' + msiName + '_' + pcaFirstComponentExtension)

    # Enhance inverted first Component with DE-GAN
    Path(outputPath + pcaFirstFolder).mkdir(parents=True, exist_ok=True)
    subprocess.call(["py", "enhance.py", deganMode,
                     outputPath + pcaInvertedFolder + '/' + msiName + '_' + pcaFirstComponentExtension,
                     outputPath + pcaFirstFolder + '/pcaFirstDEGANSecond_' + msiName + '.png'],
                    stdout=subprocess.PIPE)

# Sends the input image through the DE-GAN and then execute the PCA on the outputs
def enhanceDEGANFirst(msiName, msiPath, outputPath):
    # Enhance inverted first Component with DE-GAN
    Path(outputPath + deganFolder).mkdir(parents=True, exist_ok=True)
    files = glob.glob(msiPath + '/' + msiName + '_*.png')
    print("Found MSIs: " + str(files))

    # Do enhancement for every file of the msi
    for f in files:
        print("Start DE-GAN enhancement of " + f)
        subprocess.call(["py", "enhance.py", deganMode,
                         f,
                         outputPath + deganFolder + '/degan_' + os.path.basename(f)],
                        stdout=subprocess.PIPE)

    # Execute PCA
    executePCA('degan_' + msiName, outputPath + deganFolder, outputPath + deganFirstFolder, 12)

    # Invert first Component
    Path(outputPath + deganFirstInvertedFolder).mkdir(parents=True, exist_ok=True)
    invertImage(outputPath + deganFirstFolder + '/degan_' + msiName + '_' + pcaFirstComponentExtension,
                outputPath + deganFirstInvertedFolder + '/degan_' + msiName + '_' + pcaFirstComponentExtension)


if __name__ == '__main__':
    enhanceMSI()

import glob
import os
import subprocess
from pathlib import Path
from time import time

import click
from PIL import ImageOps
from PIL import Image

from invert_images import invertImage
from msi_pca import executePCA, invertIfNeed

pcaFolder = "/pca"
pcaFirstComponentExtension = "pca00.png"
pcaInvertedFolder = "/pcaInverted"
pcaFirstFolder = "/pcaFirstDeganSecond"
deganFolder = "/degan"
deganFirstFolder = "/deganFirstPcaSecond"
deganFirstInvertedFolder = "/deganFirstPcaSecondInverted"
deganMode = "S;T;epoch79_original"
reduceMSI = True

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

    if mode == "binarize":
        binarize(msipath)
    elif mode == "invert":
        invert(msipath + "/" + msiname, outputPath)
    elif (msiname == "_"):
        names = getAllMSINames(glob.glob(msipath + '/*'))

        print("Found Names of MSI: " + str(names))

        for name in names:
            if mode == "degan":
                degan(name, msipath, outputPath)
            elif mode == "pca":
                pca(name, msipath, outputPath)
            elif mode == "pcaFirst":
                enhancePCAFirst(name, msipath, outputPath)
            elif mode == "deganFirst":
                enhanceDEGANFirst(name, msipath, outputPath)
    else:
        if mode == "degan":
            degan(msiname, msipath, outputPath)
        elif mode == "pca":
            pca(msiname, msipath, outputPath)
        elif mode == "pcaFirst":
            enhancePCAFirst(msiname, msipath, outputPath)
        elif mode == "deganFirst":
            enhanceDEGANFirst(msiname, msipath, outputPath)

    if mode == "pca":
        postProcessPCA(outputPath)

    timeNeeded = (time() - t0)
    print("Done in %0.3fs" % timeNeeded)

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def getAllMSINames(files):
    files = [f for f in files if f.endswith(".png")]
    print(str(files))
    names = []
    for f in files:
        splittedPath = os.path.basename(f).split('_')

        if len(splittedPath) < 2:
            splittedPath = os.path.basename(f).split('.')

        if splittedPath[-2] not in names:
            names.append(splittedPath[-2])
    return names



def postProcessPCA(path):
    invertIfNeed(path)

def binarize(path):
    subprocess.call(["py", "binarize_images.py",
                     path],
                    stdout=subprocess.PIPE)

def invert(file, outputPath):
    im = Image.open(file)
    im_invert = ImageOps.invert(im)
    im_invert.save(outputPath)

# Takes the first image from the msi for enhancement
def degan(msiName, msiPath, outputPath):
    files = glob.glob(msiPath + '/' + msiName + '*.png')
    Path(outputPath + deganFolder).mkdir(parents=True, exist_ok=True)
    subprocess.call(["py", "enhance.py", deganMode,
                     files[0],
                     outputPath + deganFolder + '/degan_' + msiName + '.png'],
                    stdout=subprocess.PIPE)


# Executes a PCA on the given msi which needs to have 12 images
def pca(msiName, msiPath, outputPath):
    files = glob.glob(msiPath + '/' + msiName + '_*.png')
    executePCA(msiName, msiPath, outputPath, 1)



# Executes a PCA on the input msi, inverts the first component of the result
# and feeds the first component into the DE-GAN
def enhancePCAFirst(msiName, msiPath, outputPath):
    files = reduceMSIs(glob.glob(msiPath + '/' + msiName + '_*.png'))

    # Execute PCA
    executePCA(msiName, msiPath, outputPath + pcaFolder, 1)

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
    files = reduceMSIs(glob.glob(msiPath + '/' + msiName + '_*.png'))
    print("Found MSIs: " + str(files))

    # Do enhancement for every file of the msi
    for f in files:
        print("Start DE-GAN enhancement of " + f)
        subprocess.call(["py", "enhance.py", deganMode,
                         f,
                         outputPath + deganFolder + '/degan_' + os.path.basename(f)],
                        stdout=subprocess.PIPE)

    # Execute PCA
    executePCA('degan_' + msiName, outputPath + deganFolder, outputPath + deganFirstFolder, 1)

    # Invert first Component
    Path(outputPath + deganFirstInvertedFolder).mkdir(parents=True, exist_ok=True)
    invertImage(outputPath + deganFirstFolder + '/degan_' + msiName + '_' + pcaFirstComponentExtension,
                outputPath + deganFirstInvertedFolder + '/degan_' + msiName + '_' + pcaFirstComponentExtension)

# Reduces the number of files to the configured ones.
def reduceMSIs(files):
    if reduceMSI:
        return [f for f in files if os.path.basename(f).endswith(("_0.png", "_1.png", "_3.png", "_5.png"))]
    else:
        return files


if __name__ == '__main__':
    enhanceMSI()

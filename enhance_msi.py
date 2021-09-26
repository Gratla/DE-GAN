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

# This script implements the pipelines for DFPS and PFDS as well as the possibility to call the stand-alone DE-GAN
# and PCA. Note, that the Pipelines do not work with every MSI due to the need of manuell inversion. Also, the scripts
# does a lot at once which may cause some unexpected behaviour, so always double-check the results.
# (In the degan-function, the filename-ending might needs to be changed! Else, the files will not be found!)
# The weights and settings for the DE-GAN are configured in the deganMode variable.
#
# usage: py enhance_msi.py <mode> <msiname> <msipath> <outputpath>
#
# mode: There are multiple modes which can be configured.
#   binarize:       Calls the binarize_images.py script and uses your <msipath> as folder (<outputpath> unused)
#   invert:         Inverts a MSI (all channels) with the name <msiname> in the <msipath> folder and saves them with
#                   the same name (<msiname>) in the <outputpath> directory.
#   invertAll:      ONLY WITH <msiname> == "_". Inverts all MSIs stored in <msipath> folder.
#                   Same behaviour as invert-mode.
#   scaleAll:       ONLY WITH <msiname> == "_". Scales all images in the <msipath> folder
#                   and saves them in the <outputpath> directory. (factor is HARDCODED!!!)
#   degan:          Calls the evaluate.py script to enhance one image (filename-extension HARDCODED!!!).
#                   The file with the <msiname> and its extension (example _0.png or _pca00.png) in the <msipath> folder
#                   will be enhanced and saved to the <outputpath> directory. If <msiname> == "_", all MSIs in the
#                   <msipath> folder will be enhanced (WARNING: This may take a long time. i.e. several hours)
#   pca:            Computes the first principal component for the file with name <msiname> in the <msipath> folder.
#                   The result is saved in the <outputpath> folder. Therefore, the files and logic of the multispectral
#                   are used. If <msiname> == "_", all MSIs in the <msipath> folder will be used.
#   pcaFirst:       The given <msiname> file will be enhanced with the PFDS - pipeline (experimental)
#                   If <msiname> == "_", all MSIs in the <msipath> folder will be enhanced.
#                   (WARNING: This may take a long time. i.e. several hours)
#   deganFirst:     The given <msiname> file will be enhanced with the DFPS - pipeline (experimental)
#                   If <msiname> == "_", all MSIs in the <msipath> folder will be enhanced.
#                   (WARNING: This may take a long time. i.e. several hours)


# Configuration for the script
pcaFolder = "/pca"
pcaFirstComponentExtension = "pca00.png"
pcaInvertedFolder = "/pcaInverted"
pcaFirstFolder = "/pcaFirstDeganSecond"
deganFolder = "/degan"
deganFirstFolder = "/deganFirstPcaSecond"
deganFirstInvertedFolder = "/deganFirstPcaSecondInverted"
deganMode = "S;epoch100batchsize32_msi_bin"
#deganMode = "S;T;epoch130batchsize64_msi_bin_wrong_GT"
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
        invert(msipath + "/" + msiname + ".png", outputPath + "/" + msiname + ".png")
    elif (msiname == "_"):
        files = glob.glob(msipath + '/*')
        names = getAllMSINames(files)

        print("Found Names of MSI: " + str(names))

        if mode == "invertAll":
            for file in files:
                invert(file, outputPath + '/' + os.path.basename(file))
        elif mode == "scaleAll":
            for file in files:
                scale(file, outputPath + '/' + os.path.basename(file), 0.5)
        else:
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

def scale(file, outputPath, factor):
    im = Image.open(file)
    im = im.resize((round(im.size[0] * factor), round(im.size[1] * factor)), Image.ANTIALIAS)
    im.save(outputPath)

# Takes the first image from the msi for enhancement
def degan(msiName, msiPath, outputPath):
    files = glob.glob(msiPath + '/' + msiName + '_pca00.png')
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
        #return [f for f in files if os.path.basename(f).endswith(("_0.png", "_1.png", "_3.png", "_5.png"))] # For MSBin
        return [f for f in files if os.path.basename(f).endswith(("_F1s.png", "_F2s.png", "_F3s.png", "_F5s.png"))] # For MS-TEx
    else:
        return files


if __name__ == '__main__':
    enhanceMSI()

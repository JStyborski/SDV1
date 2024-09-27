import clip
import numpy as np
import os
from PIL import Image
import random
import shutil
import torch

def run_main():

    # srcDir = r'D:\Art_Styles\Rayonism_Natalia_Goncharova\Orig_Imgs'
    # mstDir = r'D:\Art_Styles\Rayonism_Natalia_Goncharova\Misted_Imgs\test'

    srcDir = r'D:\Art_Styles\Fish_Doll\Orig_Imgs'
    jpg_to_png(srcDir)

    # CLIP Searcher
    # srcImg = r'C:\Users\jeremy\Python_Projects\Art_Styles\images\Rayonism_Natalia_Goncharova\Generated_Imgs\00000-4004092763.png'
    # searchDir = r'C:\Users\jeremy\Python_Projects\Art_Styles\images\Rayonism_Natalia_Goncharova\Generated_Imgs'
    # nSamples = 1024
    # batchSize = 128
    # clip_embedding_searcher(srcImg, searchDir, nSamples, batchSize)

    # SubImg Saver
    # imgDir = r'C:\Users\jeremy\Python_Projects\Art_Styles\images\Rayonism_Natalia_Goncharova\Target_Imgs'
    # for fileName in os.listdir(imgDir):
    #     imgFile = os.path.join(imgDir, fileName)
    #     save_subimage(imgFile, resizeSize=256, subImgSize=256)

    # JPG to PNG Converter
    # imgDir = r'C:\Users\jeremy\Python_Projects\Art_Styles\images\Landscapes_JuliaSPowell\Orig_Imgs'
    # jpg_to_png(imgDir)

    # File Renamer
    # fileCaptDict = r'D:\Art_Styles\Rayonism_Natalia_Goncharova\Orig_Imgs\BLIP_Captions\file_caption_dict.npy'
    # file_caption_renamer(imgsDir, fileCaptDict, file2Capt=True)

    # textDir = r'D:\Art_Styles\Rayonism_Natalia_Goncharova\Orig_Imgs\BLIP_Captions'
    # create_file_capt_dict(imgsDir, textDir)


def clip_embedding_searcher(srcImg, searchDir, nSamples, batchSize, findNearest=True):
    """
    :param srcImg: [str] - Source image filepath
    :param searchDir: [str] - Directory containing images to encode/search
    :param nSamples: [int] - Total number of samples to search
    :param batchSize: [int] - Batch size to search in one iteration
    :param findNearest: [Bool] - Find closest embedding (True) or furthest embedding (False)
    :return:
    """

    # Set device and load CLIP
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/16', device=device)

    # Load source image and encode
    imgTens = preprocess(Image.open(srcImg)).unsqueeze(0).to(device)
    with torch.no_grad():
        srcEnc = model.encode_image(imgTens).to(device)

    # Initialize searched files list and encodings
    fileList = os.listdir(searchDir)
    searchFiles = []
    searchEncs = torch.zeros(nSamples, 512).to(device)

    # Loop through nSamples in batches and preprocess/encode images from searchDir
    for iBatch in range(int(nSamples / batchSize)):
        batchTens = torch.zeros(batchSize, 3, 224, 224).to(device)
        for i in range(batchSize):
            imgFile = random.choice(fileList)
            searchFiles.append(imgFile)
            batchTens[i, :, :, :] = preprocess(Image.open(os.path.join(searchDir, imgFile))).to(device)
        with torch.no_grad():
            searchEncs[iBatch*batchSize:(iBatch+1)*batchSize, :] = model.encode_image(batchTens)

    # Calculate cosine similarity between source and searched image embeddings
    cosSim = torch.nn.functional.cosine_similarity(srcEnc, searchEncs)

    # Get nearest/furthest file, copy it to source directory, and print results
    topImg = cosSim.topk(k=1, largest=findNearest)
    topFile = searchFiles[topImg.indices[0]]
    shutil.copy(os.path.join(searchDir, topFile), os.path.splitext(srcImg)[0] + '_Nearest_' + os.path.basename(topFile))
    print('Nearest file: ' + topFile + ' (cossim = ' + str(topImg.values[0]) + ')')

def target_image_creator(imgFile, imgSize=256):
    """
    :param imgFile: [str] - Image filepath to save as
    :param imgSize: [int] - Square image size
    :return: None, saves image file
    """

    imgArr = np.random.randint(0, 255, size=(imgSize, imgSize, 3), dtype=np.uint8)
    #imgArr = 255 * np.ones((imgSize, imgSize, 3), dtype=np.uint8)
    imgOut = Image.fromarray(imgArr)
    imgOut.save(imgFile)

def copy_files(srcDir, trgDir, fileNameList=None):
    """
    :param srcDir: [str] - Source directory containing files to copy
    :param trgDir: [str] - Target directory to copy files into
    :param fileNameList: [list] - List of filenames to copy from srcDir to newDir
    :return: None, copies files
    """

    # Generate list of all filenames in srcDir if no fileNameList given
    if fileNameList is None:
        fileNameList = os.listdir(srcDir)

    # Make new directory to copy files to
    os.makedirs(trgDir, exist_ok=True)

    # Copy each file from fileNameList to newDir
    for fileName in fileNameList:
        shutil.copy(os.path.join(srcDir, fileName), os.path.join(trgDir, fileName))

def jpg_to_png(srcDir):
    """
    :param srcDir: Source directory with jpgs to convert into pngs
    :return: None, converts jpg to png
    """

    for file in os.listdir(srcDir):
        fileName, fileExt = os.path.splitext(file)  # Get just the name of the file
        if not os.path.isfile(os.path.join(srcDir, file)) or fileExt not in ['.jpg', '.jpeg']:
            continue
        img = Image.open(os.path.join(srcDir, file)).convert('RGB')  # Open jpg image
        img.save(os.path.join(srcDir, fileName + '.png'), format='PNG')  # Save as png
        os.remove(os.path.join(srcDir, file))  # Remove original jpg image

def save_subimage(imgFile, resizeSize=256, subImgSize=128):
    """
    :param imgFile: [str] - Filepath to image file
    :param resizeSize: [int] - Size to resize image to
    :param subImgSize: [int] - Size to center crop image to
    :return: None, saves subimage file
    """

    # Get image abspath
    absPath = os.path.abspath(imgFile)
    absPath = os.path.join('\\\\?\\' + absPath)

    # Get image and subimage
    imgArr = np.asarray(Image.open(absPath).convert('RGB').resize((resizeSize, resizeSize), resample=Image.BICUBIC))
    minLim = int((resizeSize - subImgSize) / 2)
    maxLim = int((resizeSize + subImgSize) / 2)
    subImgArr = imgArr[minLim:maxLim, minLim:maxLim, :]  # Take center crop of image

    # Save subimg array for viewing
    outImg = Image.fromarray(np.uint8(subImgArr))
    outImg.save(os.path.splitext(imgFile)[0] + '_Subimage.png')

def create_file_capt_dict(imgsDir, textDir):
    """
    :param imgsDir: [str] - Directory with images where filenames will serve as dictionary keys
    :param textDir: [str] - Directory with text files with corresponding filenames that contain image caption text
    :return: None, saves dictionary file
    """

    fileCaptDict = {}
    for file in os.listdir(imgsDir):

        # Only process .png files
        fileName, fileExt = os.path.splitext(file)
        if not os.path.isfile(os.path.join(imgsDir, file)) or fileExt != '.png':
            continue

        # Read image description contained in corresponding text file
        with open(os.path.join(textDir, fileName + '_Cleaned.txt'), 'r', encoding='utf-8', errors='ignore') as f:
            line = f.readlines()

        # Remove problematic characters
        # Chained .replace() shown to be fastest: https://stackoverflow.com/questions/3411771/best-way-to-replace-multiple-characters-in-a-string
        newLine = line[0].replace('<', '').replace('>', '').replace(':', '').replace('"', '').replace('/', '').replace('\\', '').replace('|', '').replace('?', '').replace('*', '')
        fileCaptDict[fileName] = newLine

    np.save(os.path.join(textDir, 'file_caption_dict.npy'), fileCaptDict)

def file_caption_renamer(imgsDir, fileCaptDict, file2Capt=True):
    """
    :param imgsDir: [str] - Directory with source images to rename, filenames correspond to fileCaptDict keys
    :param fileCaptDict: [str] - Filepath to file-caption dictionary
    :param file2Capt: [Bool] - Boolean to do file-to-caption (True) or caption-to-file (False) renaming
    :return: None, renames all matching files according to dictionary
    """

    # Load dictionary
    fileCaptDict = np.load(fileCaptDict, allow_pickle=True).item()

    # Reverse dictionary keys/values if desired
    if not (file2Capt):
        captFileDict = {}
        for key in fileCaptDict.keys():
            val = fileCaptDict[key]
            captFileDict[val] = key
        fileCaptDict = captFileDict

    # Create abspath for long filenames
    abspath = os.path.abspath(imgsDir)
    abspath = '\\\\?\\' + abspath

    # Loop through ALL files in all directories/subdirectories in the given folder
    # Change filenames according to the dictionary
    for root, dirs, files in os.walk(abspath):
        for file in files:
            # Check file extension
            fileName, fileExt = os.path.splitext(os.path.basename(file))
            if fileExt != '.png' or fileName not in fileCaptDict.keys():
                continue
            os.rename(os.path.join(root, file), os.path.join(root, fileCaptDict[fileName] + '.png'))

def scan_for_substring(srcDir):
    """
    :param srcDir: [str] - Source directory with text files to search
    :return: Unique substrings that exist in directory files and match criteria
    """

    allList = []
    for file in os.listdir(srcDir):

        # Only process .txt files
        fileName, fileExt = os.path.splitext(file)
        if not os.path.isfile(file) or fileExt != '.txt':
            continue

        # Read the text in file
        with open(os.path.join(srcDir, file), 'r', encoding='utf-8', errors='ignore') as f:
            line = f.readlines()

        # Split the line by whitespace and check for words with capital letters
        lineParts = line[0].split(' ')
        for i, part in enumerate(lineParts):
            if part[0].isupper():
                allList.append(part)

    return set(allList)


def remove_substring(srcDir, newFileSuffix):
    """
    :param srcDir: [str] - Source directory with text files to search
    :param newFileSuffix: [str] - Suffix to add to new files with removed substring
    :return: None, saves files
    """

    for file in os.listdir(srcDir):

        # Only process .txt files
        fileName, fileExt = os.path.splitext(file)
        if not os.path.isfile(file) or fileExt != '.txt':
            continue

        # Read the text in file
        with open(os.path.join(srcDir, file), 'r', encoding='utf-8', errors='ignore') as f:
            line = f.readlines()

        # Split the line by whitespace and check for words with capital letters
        lineParts = line[0].split(' ')
        removeList = []
        for i, part in enumerate(lineParts):
            if part[0].isupper():
                removeList.append(i)

        # Remove parts of the line corresponding to words with capital letters
        for idx in removeList[::-1]:
            lineParts.remove(lineParts[idx])

        # Create new line and write to new file
        newLine = ' '.join(lineParts)
        newFile = fileName + newFileSuffix
        with open(os.path.join(srcDir, newFile), 'w', encoding='utf-8', errors='ignore') as f:
            f.write(newLine)

if __name__ == "__main__":
    run_main()
import clip
from io import BytesIO
import numpy as np
import os
from PIL import Image, ImageFilter, ImageOps
import random
import shutil
from sklearn.cluster import KMeans
import torch

def run_main():

    imgDir = r'D:\Art_Styles\Fish_Doll\Orig_Imgs'

    # SubImg Saver
    # imgDir = r'C:\Users\jeremy\Python_Projects\Art_Styles\images\Rayonism_Natalia_Goncharova\Target_Imgs'
    # for fileName in os.listdir(imgDir):
    #     imgFile = os.path.join(imgDir, fileName)
    #     save_subimage(imgFile, resizeSize=256, subImgSize=256)

    # JPG to PNG Converter
    # imgDir = r'D:\Art_Styles\rand_test'
    # jpg_to_png(imgDir)

    # File Renamer
    # fileCaptDict = r'D:\Art_Styles\Rayonism_Natalia_Goncharova\Orig_Imgs\BLIP_Captions\file_caption_dict.npy'
    # file_caption_renamer(imgsDir, fileCaptDict, file2Capt=True)

    # textDir = r'D:\Art_Styles\Rayonism_Natalia_Goncharova\Orig_Imgs\BLIP_Captions'
    # create_file_capt_dict(imgsDir, textDir)

    # Image Resizer
    # outDir = r'D:\Art_Styles\Fish_Doll\Orig_Imgs\resize512'
    # image_resizer(srcDir, outDir, 512)

    # outDir = r'D:\Art_Styles\Fish_Doll\Misted_Imgs\Orig_M2\BDR6'
    # image_transformer(srcDir, outDir, bdr_transform, n_bits=2)

    # imgDir = r'D:\MSCOCO\train2017'
    # sig = img_pixel_variance(imgDir, queriesPerSample=10, sampleLimit=1000, x_range=12, y_range=12)
    # print(sig)

    # from JS_img2img import load_img
    # img1 = r'D:\Art_Styles\Fish_Doll\Orig_Imgs\a stuffed animal is sitting on a ledge in a room with a white wall.png'
    # img2 = r'D:\Art_Styles\Fish_Doll\Misted_Imgs\Orig_M2\a stuffed animal is sitting on a ledge in a room with a white wall.png'
    # img1Tens = load_img(img1, 512)
    # img2Tens = load_img(img2, 512)

    # CLIP Searcher
    # srcImg = r'C:\Users\jeremy\Python_Projects\Art_Styles\images\Rayonism_Natalia_Goncharova\Generated_Imgs\00000-4004092763.png'
    # searchDir = r'C:\Users\jeremy\Python_Projects\Art_Styles\images\Rayonism_Natalia_Goncharova\Generated_Imgs'
    # nSamples = 1024
    # batchSize = 128
    # clip_embedding_searcher(srcImg, searchDir, nSamples, batchSize)

    # Set device and load CLIP
    imgDir2 = r'D:\Art_Styles\rand_test'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-L/14', device=device)

    # Get image embeddings
    allEmbs = get_clip_embs(imgDir, preprocess, model, device)
    allEmbs2 = get_clip_embs(imgDir2, preprocess, model, device)
    sims = torch.nn.functional.cosine_similarity(allEmbs, allEmbs2)
    print(sims)

    # Get zero shot classification
    # IN_dict = torch.load(r'D:\ImageNet1k\ImageNet_Class_Dict.pt')
    # for i in range(len(allEmbs)):
    #     print(clip_zero_shot_cls(allEmbs[i], IN_dict, model, device, topk=3))
    # print('')
    # for i in range(len(allEmbs2)):
    #     print(clip_zero_shot_cls(allEmbs2[i], IN_dict, model, device, topk=3))

def grayscale_transform(img):
    img = ImageOps.grayscale(img)
    return img

def blur_transform(img, sigma=2.):
    img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def jpeg_transform(img, quality=10):
    buffer = BytesIO()  # Create buffer object
    img.save(buffer, 'JPEG', quality=quality)  # Save image as a JPEG in memory but not on disk
    imgData = BytesIO(buffer.getbuffer())  # Load the image data contained in the buffer
    img = Image.open(imgData)  # Load the image data
    return img

def bdr_transform(img, n_bits=4):
    # Follows the BDR algorithm from https://arxiv.org/abs/1910.04397
    bit_power = 2 ** n_bits - 1
    img = np.array(img)
    img = img / 255. * bit_power  # Scale image based on bit power
    img = np.round(img) / bit_power * 255.  # Quantize image and scale to uint8
    img = Image.fromarray(img.astype(np.uint8))
    return img

def kmeans_transform(img, k=16):
    # Copied from Mingzhi's code, not sure if he got this from somewhere.
    img = np.array(img)
    pixel_values = img.flatten().reshape(-1, 1)  # Reshape image into a 1D array of pixel values
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')  # Perform k-means clustering
    kmeans.fit(pixel_values)
    bin_centers = kmeans.cluster_centers_  # Get the cluster centroids
    quantized_pixels = bin_centers[kmeans.labels_].astype(np.uint8)  # Assign each pixel value to the nearest bin center
    img = quantized_pixels.reshape(img.shape)  # Reshape the quantized pixels back to the original image shape
    img = Image.fromarray(img)
    return img

def noise_transform(img, noise_type='uniform', strength=16):
    # Apply uniform or normal noise to image
    img = np.array(img).astype(float)
    if noise_type == 'uniform':
        img = img + np.random.uniform(low=-strength, high=strength, size=img.shape)
    elif noise_type == 'normal':
        img = img + np.random.normal(scale=strength)
    img = np.clip(img, a_min=0., a_max=255.)
    img = Image.fromarray(img.astype(np.uint8))
    return img

def image_transformer(imgDir, outDir, transform, **kwargs):
    """
    Applies given transform to PIL images
    :param imgDir: [str] - Image directory to gather images from
    :param outDir: [str] - Directory to save images at
    :param transform: [torchvision.transforms or equivalent function] - Function to process PIL images
    :return:
    """

    # Create output directory if it doesn't exist
    os.makedirs(outDir, exist_ok=True)

    for imgFile in os.listdir(imgDir):

        # Only process .png files
        fileName, fileExt = os.path.splitext(imgFile)
        if not os.path.isfile(os.path.join(imgDir, imgFile)) or fileExt != '.png':
            continue

        img = Image.open(os.path.join(imgDir, imgFile)).convert('RGB')
        img = transform(img, **kwargs)
        img.save(os.path.join(outDir, imgFile))

def get_clip_embs(imgDir, clipPreprocess, clipModel, device):

    encList = []
    for imgFile in os.listdir(imgDir):

        # Only process .png files
        fileName, fileExt = os.path.splitext(imgFile)
        if not os.path.isfile(os.path.join(imgDir, imgFile)) or fileExt != '.png':
            continue

        # Load image and encode
        imgTens = clipPreprocess(Image.open(os.path.join(imgDir, imgFile))).unsqueeze(0).to(device)
        with torch.no_grad():
            encList.append(clipModel.encode_image(imgTens).to(device))

    return torch.cat(encList, dim=0).float()

def clip_zero_shot_cls(imgEmb, clsDict, clipModel, device, topk=5):

    # Get text embeddings
    text_inputs = torch.cat([clip.tokenize(f'A photo of a {cls}') for cls in clsDict.values()]).to(device)
    with torch.no_grad():
        text_features = clipModel.encode_text(text_inputs).to(device).float()

    sims = torch.nn.functional.cosine_similarity(imgEmb, text_features)
    _, indices = sims.topk(topk)
    return [clsDict[idx.item()] for idx in indices.data]

def clip_embedding_searcher(srcImg, searchDir, nSamples, batchSize, findNearest=True):
    """
    Find the image from a given directory that is closest to a given image, as measured by CLIP score
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
    Creates an image of specified size using a certain method (custom define)
    :param imgFile: [str] - Image filepath to save on
    :param imgSize: [int] - Square image size
    :return: None, saves image file
    """

    imgArr = np.random.randint(0, 255, size=(imgSize, imgSize, 3), dtype=np.uint8)
    #imgArr = 255 * np.ones((imgSize, imgSize, 3), dtype=np.uint8)
    imgOut = Image.fromarray(imgArr)
    imgOut.save(imgFile)

def jpg_to_png(imgDir):
    """
    Convert all images in directory to PNG
    :param imgDir: Source directory with jpgs to convert into pngs
    :return: None, converts jpg to png
    """

    for file in os.listdir(imgDir):
        fileName, fileExt = os.path.splitext(file)  # Get just the name of the file
        if not os.path.isfile(os.path.join(imgDir, file)) or fileExt not in ['.jpg', '.jpeg']:
            continue
        img = Image.open(os.path.join(imgDir, file)).convert('RGB')  # Open jpg image
        img.save(os.path.join(imgDir, fileName + '.png'), format='PNG')  # Save as png
        os.remove(os.path.join(imgDir, file))  # Remove original jpg image

def create_file_capt_dict(imgsDir, textDir):
    """
    Create and save a dictionary of filename-caption values
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
    Rename all files in a directory according to their captions from a dictionary
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
    Search through text file strings and classify which files contain the string
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
    Sort through text file strings and remove words that satisfy a certain criteria
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

# def img_pixel_variance(imgDir, queriesPerSample, sampleLimit, x_range, y_range):
#
#     diffList = []
#     listDir = os.listdir(imgDir)
#     for i in range(len(listDir)):
#
#         imgFile = np.random.choice(listDir)
#
#         # Only process .png files
#         fileName, fileExt = os.path.splitext(imgFile)
#         if not os.path.isfile(os.path.join(imgDir, imgFile)) or fileExt not in ['.png', '.jpg']:
#             continue
#
#         img = np.array(Image.open(os.path.join(imgDir, imgFile)).convert('RGB')).astype(float)
#         h, w, c = img.shape
#
#         for _ in range(queriesPerSample):
#             rand_h = np.random.randint(y_range, h - y_range)
#             rand_w = np.random.randint(x_range, w - x_range)
#             rand_c = np.random.randint(c)
#             h_del = rand_h + np.random.choice([-1, 1]) * y_range
#             w_del = rand_w + np.random.choice([-1, 1]) * x_range
#
#             source_pixel = img[rand_h, rand_w, rand_c]
#             query_pixel = img[h_del, w_del, rand_c]
#             diffList.append(query_pixel - source_pixel)
#
#         if i > sampleLimit:
#             break
#
#     print(diffList)
#     return np.std(diffList)

if __name__ == "__main__":
    run_main()
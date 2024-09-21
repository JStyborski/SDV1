import clip
import numpy as np
import os
import PIL
import piq
import torch
import torchvision

import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

def run_main():

    inputSize = 256
    t = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(int(round(256 / 224 * 224))),  # CIFAR: 1.1428 * 28 = 32, IN: 1.1428 * 224 = 256
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor()
    ])

    # CLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clipModel, _ = clip.load('ViT-B/16', device=device)

    # Source images
    srcImgRoot = r'C:\Users\jeremy\Python_Projects\Art_Styles\images\Rayonism_Natalia_Goncharova\Orig_Imgs'
    # srcImgDS = No_Labels_Images(srcImgRoot, t, inputSize=inputSize)
    # srcImgDL = torch.utils.data.DataLoader(srcImgDS, batch_size=1, shuffle=False)
    # srcMeanRecDS = No_Labels_Images(os.path.join(srcImgRoot, 'Mean_Reconstr_Imgs'), t, inputSize=inputSize)
    # srcMeanRecDL = torch.utils.data.DataLoader(srcMeanRecDS, batch_size=1, shuffle=False)
    # srcDiffRecDS = No_Labels_Images(os.path.join(srcImgRoot, 'Diff_Reconstr_Imgs'), t, inputSize=inputSize)
    # srcDiffRecDL = torch.utils.data.DataLoader(srcDiffRecDS, batch_size=1, shuffle=False)
    # srcGenDS = No_Labels_Images(os.path.join(srcImgRoot, 'Generated_Imgs'), t, inputSize=inputSize)
    # srcGenDL = torch.utils.data.DataLoader(srcGenDS, batch_size=1, shuffle=False)

    # Target images
    # trgImgRoot = r'C:\Users\jeremy\Python_Projects\Art_Styles\images\Rayonism_Natalia_Goncharova\Target_Imgs'
    # trgImgDS = No_Labels_Images(trgImgRoot, t, inputSize=inputSize)
    # trgImgDL = torch.utils.data.DataLoader(trgImgDS, batch_size=1, shuffle=False)

    # Misted images
    mstImgRoot = r'C:\Users\jeremy\Python_Projects\Art_Styles\images\Rayonism_Natalia_Goncharova\Misted_Imgs\MIST_Target_Mode-0_16px'
    # mstImgDS = No_Labels_Images(mstImgRoot, t, inputSize=inputSize)
    # mstImgDL = torch.utils.data.DataLoader(mstImgDS, batch_size=1, shuffle=False)
    # mstMeanRecDS = No_Labels_Images(os.path.join(mstImgRoot, 'Mean_Reconstr_Imgs'), t, inputSize=inputSize)
    # mstMeanRecDL = torch.utils.data.DataLoader(mstMeanRecDS, batch_size=1, shuffle=False)
    # mstDiffRecDS = No_Labels_Images(os.path.join(mstImgRoot, 'Diff_Reconstr_Imgs'), t, inputSize=inputSize)
    # mstDiffRecDL = torch.utils.data.DataLoader(mstDiffRecDS, batch_size=1, shuffle=False)
    # mstGenDS = No_Labels_Images(os.path.join(mstImgRoot, 'Generated_Imgs'), t, inputSize=inputSize)
    # mstGenDL = torch.utils.data.DataLoader(mstGenDS, batch_size=1, shuffle=False)

    # No-ref Metric
    # metric = brisque_metric
    # mstImg_brisque = noref_metric_calc(metric, mstImgDL, device, [0])
    # mstMeanRec_brisque = noref_metric_calc(metric, mstMeanRecDL, device, [0])
    # mstDiffRec_brisque = noref_metric_calc(metric, mstDiffRecDL, device, [0])
    # mstGen_brisque = noref_metric_calc(metric, mstGenDL, device, [0])

    # Ref Metric
    # metricList = clip_metric
    # srcsrc_metric = ref_metric_calc(metric, srcImgDL, srcImgDL, clipModel, device, 'perm', [0], True)
    # srcsrc2_metric = ref_metric_calc(metric, srcImgDL, srcImgDL, clipModel, device, 'perm', [0, 1], True)
    # srcsrcGen_metric = ref_metric_calc(metric, srcImgDL, srcGenDL, clipModel, device, 'perm', [0, 1])
    # srctrg_metric = ref_metric_calc(metric, srcImgDL, trgImgDL, clipModel, device, 'perm', [0])
    # srcmst_metric = ref_metric_calc(metric, srcImgDL, mstImgDL, clipModel, device, 'corr', [0])
    # srcmstMR_metric = ref_metric_calc(metric, srcImgDL, mstMeanRecDL, clipModel, device, 'corr', [0])
    # srcmstDR_metric = ref_metric_calc(metric, srcImgDL, mstDiffRecDL, clipModel, device, 'corr', [0])
    # srcmstGen_metric = ref_metric_calc(metric, srcImgDL, mstGenDL, clipModel, device, 'perm', [0, 1])
    # trgmst_metric = ref_metric_calc(metric, trgImgDL, mstImgDL, clipModel, device, 'perm', [0, 1])
    # trgmstMR_metric = ref_metric_calc(metric, trgImgDL, mstMeanRecDL, clipModel, device, 'perm', [0, 1])
    # trgmstDR_metric = ref_metric_calc(metric, trgImgDL, mstDiffRecDL, clipModel, device, 'perm', [0, 1])
    # trgmstGen_metric = ref_metric_calc(metric, trgImgDL, mstGenDL, clipModel, device, 'perm', [0, 1])
    # mstmstMR_metric = ref_metric_calc(metric, mstImgDL, mstMeanRecDL, clipModel, device, 'corr', [0])
    # mstmstDR_metric = ref_metric_calc(metric, mstImgDL, mstDiffRecDL, clipModel, device, 'corr', [0])
    # mstmstGen_metric = ref_metric_calc(metric, mstImgDL, mstGenDL, clipModel, device, 'perm', [0, 1])
    # srcGenMstGen_metric = ref_metric_calc(metric, srcGenDL, mstGenDL, clipModel, device, 'perm', [0, 1])

    # Image L2
    # srcsrcMR_l2 = ref_metric_calc(l2_metric, srcImgDL, srcMeanRecDL, None, device, 'corr', [0])
    # srcsrcDR_l2 = ref_metric_calc(l2_metric, srcImgDL, srcDiffRecDL, None, device, 'corr', [0])
    # srctrg_l2 = ref_metric_calc(l2_metric, srcImgDL, trgImgDL, None, device, 'perm', [0])
    # srcmst_l2 = ref_metric_calc(l2_metric, srcImgDL, mstImgDL, None, device, 'corr', [0])
    # srcmstMR_l2 = ref_metric_calc(l2_metric, srcImgDL, mstMeanRecDL, None, device, 'corr', [0])
    # srcmstDR_l2 = ref_metric_calc(l2_metric, srcImgDL, mstDiffRecDL, None, device, 'corr', [0])
    # mstmstMR_l2 = ref_metric_calc(l2_metric, mstImgDL, mstMeanRecDL, None, device, 'corr', [0])
    # mstmstDR_l2 = ref_metric_calc(l2_metric, mstImgDL, mstDiffRecDL, None, device, 'corr', [0])

    # Image Encodings
    ckptSuffix = '_AWA0887_LR2en2_Accum1_EMA0p0'
    srcMeanEncDS = No_Labels(os.path.join(srcImgRoot, 'Lat_Mean' + ckptSuffix), None, torch.load)
    srcMeanEncDL = torch.utils.data.DataLoader(srcMeanEncDS, batch_size=1, shuffle=False)
    # srcDiffEncDS = No_Labels(os.path.join(srcImgRoot, 'Lat_N2I'), None, torch.load, -1)
    # srcDiffEncDL = torch.utils.data.DataLoader(srcDiffEncDS, batch_size=1, shuffle=False)
    # trgMeanEncDS = No_Labels(os.path.join(trgImgRoot, 'Lat_Mean'), None, torch.load)
    # trgMeanEncDL = torch.utils.data.DataLoader(trgMeanEncDS, batch_size=1, shuffle=False)
    mstMeanEncDS = No_Labels(os.path.join(mstImgRoot, 'Lat_Mean' + ckptSuffix), None, torch.load)
    mstMeanEncDL = torch.utils.data.DataLoader(mstMeanEncDS, batch_size=1, shuffle=False)
    # mstDiffEncDS = No_Labels(os.path.join(mstImgRoot, 'Lat_N2I'), None, torch.load, -1)
    # mstDiffEncDL = torch.utils.data.DataLoader(mstDiffEncDS, batch_size=1, shuffle=False)

    metrics = [l2_metric, cosine_metric]
    for metric in metrics:
        # srcMsrcM_metric = ref_metric_calc(metric, srcMeanEncDL, srcMeanEncDL, None, device, 'perm', [0, 1], True)
        # srcMsrcD_metric = ref_metric_calc(metric, srcMeanEncDL, srcDiffEncDL, None, device, 'corr', [0])
        # srcMtrgM_metric = ref_metric_calc(metric, srcMeanEncDL, trgMeanEncDL, None, device, 'perm', [0])
        srcMmstM_metric = ref_metric_calc(metric, srcMeanEncDL, mstMeanEncDL, None, device, 'corr', [0])
        # srcMmstD_metric = ref_metric_calc(metric, srcMeanEncDL, mstDiffEncDL, None, device, 'corr', [0])
        # trgMmstM_metric = ref_metric_calc(metric, trgMeanEncDL, mstMeanEncDL, None, device, 'perm', [0, 1])
        # trgMmstD_metric = ref_metric_calc(metric, trgMeanEncDL, mstDiffEncDL, None, device, 'perm', [0, 1])
        # mstMmstM_metric = ref_metric_calc(metric, mstMeanEncDL, mstMeanEncDL, None, device, 'perm', [0, 1], True)
        # mstMmstD_metric = ref_metric_calc(metric, mstMeanEncDL, mstDiffEncDL, None, device, 'corr', [0])
        print(srcMmstM_metric)

    # # Source images
    # srcImgRoot = r'C:\Users\jeremy\Python_Projects\Art_Styles\images\Rayonism_Natalia_Goncharova\Orig_Imgs'
    # srcFwdDEncDS = No_Labels(os.path.join(srcImgRoot, 'Lat_I2N'), None, torch.load)
    # srcFwdDEncDL = torch.utils.data.DataLoader(srcFwdDEncDS, batch_size=1, shuffle=False)
    # srcRevDEncDS = No_Labels(os.path.join(srcImgRoot, 'Lat_N2I'), None, torch.load)
    # srcRevDEncDL = torch.utils.data.DataLoader(srcRevDEncDS, batch_size=1, shuffle=False)
    #
    # # Mist images
    # mstImgRoot = r'C:\Users\jeremy\Python_Projects\Art_Styles\images\Rayonism_Natalia_Goncharova\Misted_Imgs\MIST_Target_Mode-0_16px'
    # mstFwdDEncDS = No_Labels(os.path.join(mstImgRoot, 'Lat_I2N'), None, torch.load)
    # mstFwdDEncDL = torch.utils.data.DataLoader(mstFwdDEncDS, batch_size=1, shuffle=False)
    # mstRevDEncDS = No_Labels(os.path.join(mstImgRoot, 'Lat_N2I'), None, torch.load)
    # mstRevDEncDL = torch.utils.data.DataLoader(mstRevDEncDS, batch_size=1, shuffle=False)
    #
    # for i, ((srcFwdSL, _), (srcRevSL, _), (mstFwdSL, _), (mstRevSL, _)) in enumerate(zip(srcFwdDEncDL, srcRevDEncDL, mstFwdDEncDL, mstRevDEncDL)):
    #     srcFwdSL, srcRevSL, mstFwdSL, mstRevSL = srcFwdSL[0], srcRevSL[0], mstFwdSL[0], mstRevSL[0]
    #     assert torch.all(srcFwdSL[-1] == srcRevSL[0])
    #     assert torch.all(mstFwdSL[-1] == mstRevSL[0])
    #     srcAdjDist = torch.linalg.vector_norm(torch.flip(srcRevSL, [0]) - srcFwdSL, ord=2, dim=(1, 2, 3, 4))
    #     mstAdjDist = torch.linalg.vector_norm(torch.flip(mstRevSL, [0]) - mstFwdSL, ord=2, dim=(1, 2, 3, 4))
    #     srcStartPt, srcMidPt, srcEndPt = srcFwdSL[0], srcFwdSL[-1], srcRevSL[-1]
    #     #mstStartPt, mstMidPt, mstEndPt = mstFwdSL[0], mstFwdSL[-1], mstRevSL[-1]
    #     srcSL = torch.cat((srcFwdSL, srcRevSL[1:]), dim=0)
    #     mstSL = torch.cat((mstFwdSL, mstRevSL[1:]), dim=0)
    #     srcDist2Start = torch.linalg.vector_norm(srcSL - srcStartPt, ord=2, dim=(1, 2, 3, 4))
    #     srcDist2Mid = torch.linalg.vector_norm(srcSL - srcMidPt, ord=2, dim=(1, 2, 3, 4))
    #     srcDist2End = torch.linalg.vector_norm(srcSL - srcEndPt, ord=2, dim=(1, 2, 3, 4))
    #     mstDist2Start = torch.linalg.vector_norm(mstSL - srcStartPt, ord=2, dim=(1, 2, 3, 4))
    #     mstDist2Mid = torch.linalg.vector_norm(mstSL - srcMidPt, ord=2, dim=(1, 2, 3, 4))
    #     mstDist2End = torch.linalg.vector_norm(mstSL - srcEndPt, ord=2, dim=(1, 2, 3, 4))
    #     srcVecFromStart = torch.flatten(srcSL - srcStartPt, start_dim=1)
    #     mstVecFromStart = torch.flatten(mstSL - srcStartPt, start_dim=1)
    #
    #     ax1 = plt.figure().add_subplot()
    #     ax1.plot(range(len(srcFwdSL)), srcAdjDist.numpy(), color='b', marker='.', markersize=5, label='Src')
    #     ax1.plot(range(len(mstFwdSL)), mstAdjDist.numpy(), color='m', marker='.', markersize=5, label='Mst')
    #     ax1.set_xlabel('Timestep (T=1000, DDIM_Steps=100)')
    #     ax1.set_ylabel('L2 Distance between I2N<->N2I')
    #     ax1.legend()
    #     plt.show()
    #
    #     ax11 = plt.figure().add_subplot()
    #     ax11.plot(range(len(srcSL)), torch.linalg.vector_norm(mstSL - srcSL, ord=2, dim=(1, 2, 3, 4)).numpy(), color='b', marker='.', markersize=5)
    #     ax11.set_xlabel('Image2Noise2Image Step')
    #     ax11.set_ylabel('L2 Distance between Src-Mst Streamlines')
    #     ax11.set_ylim([0, None])
    #     plt.show()
    #
    #     ax2 = plt.figure().add_subplot(projection='3d')
    #     ax2.plot(srcDist2Start[0:101].numpy(), srcDist2Mid[0:101].numpy(), srcDist2End[0:101].numpy(), color='b', marker='.', markersize=5, label='Src I2N')
    #     ax2.plot(srcDist2Start[100:201].numpy(), srcDist2Mid[100:201].numpy(), srcDist2End[100:201].numpy(), color='r', marker='.', markersize=5, label='Src N2I')
    #     ax2.plot(mstDist2Start[0:101].numpy(), mstDist2Mid[0:101].numpy(), mstDist2End[0:101].numpy(), color='m', marker='.', markersize=5, label='Mst I2N')
    #     ax2.plot(mstDist2Start[100:201].numpy(), mstDist2Mid[100:201].numpy(), mstDist2End[100:201].numpy(), color='y', marker='.', markersize=5, label='Mst N2I')
    #     ax2.set_xlabel('Dist to Start')
    #     ax2.set_ylabel('Dist to Mid')
    #     ax2.set_zlabel('Dist to End')
    #     ax2.legend()
    #     plt.show()
    #
    #     emb = MDS(n_components=3)
    #     mdsVecFromStart = emb.fit_transform(torch.cat((srcVecFromStart, mstVecFromStart), dim=0).numpy())
    #     mdsVecFromStart = mdsVecFromStart - mdsVecFromStart[0, :]  # Reset such that start point is at (0,0)
    #     ax3 = plt.figure().add_subplot(projection='3d')
    #     ax3.plot(mdsVecFromStart[:201, 0], mdsVecFromStart[:201, 1], mdsVecFromStart[:201, 2], color='b', marker='.', markersize=5, label='Src')
    #     ax3.plot(mdsVecFromStart[201:, 0], mdsVecFromStart[201:, 1], mdsVecFromStart[201:, 2], color='m', marker='.', markersize=5, label='Mst')
    #     ax3.set_xlabel('MDS Dim 1')
    #     ax3.set_ylabel('MDS Dim 2')
    #     ax3.set_zlabel('MDS Dim 3')
    #     ax3.legend()
    #     plt.show()


class No_Labels_Images(torch.utils.data.Dataset):
    def __init__(self, rootDir, transform, inputSize):
        self.rootDir = rootDir
        self.transform = transform
        self.inputSize = inputSize
        self.filesList = [f for f in os.listdir(self.rootDir) if os.path.isfile(os.path.join(self.rootDir, f))]

    def __len__(self):
        return len(self.filesList)

    def __getitem__(self, index):
        img = PIL.Image.open(os.path.join(self.rootDir, self.filesList[index])).convert('RGB').resize((self.inputSize, self.inputSize), resample=PIL.Image.BICUBIC)
        #img = PIL.Image.open(os.path.join(self.rootDir, self.filesList[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = 0

        return img, target

class No_Labels(torch.utils.data.Dataset):
    def __init__(self, rootDir, transform, loader, tensIdx=None):
        self.rootDir = rootDir
        self.transform = transform
        self.loader = loader
        self.tensIdx = tensIdx
        self.filesList = [f for f in os.listdir(self.rootDir) if os.path.isfile(os.path.join(self.rootDir, f))]

    def __len__(self):
        return len(self.filesList)

    def __getitem__(self, index):
        sample = self.loader(os.path.join(self.rootDir, self.filesList[index]))
        if self.tensIdx is not None:
            sample = sample[self.tensIdx]
        if self.transform is not None:
            sample = self.transform(sample)
        target = 0

        return sample, target

def noref_metric_calc(norefFn, srcDL, device, avgIdxList=None):
    norefArr = np.zeros((len(srcDL.dataset)))
    for i, (srcTens, _) in enumerate(srcDL):
        srcTens = srcTens.to(device)
        norefArr[i] = norefFn(srcTens, device)
    if avgIdxList is not None:
        norefArr = np.mean(norefArr, axis=0, keepdims=False)
    return norefArr

def ref_metric_calc(refFn, srcDL, trgDL, model, device, corrperm='corr', avgIdxList=None, maskDiag=False):
    if corrperm == 'corr':
        assert len(srcDL.dataset) == len(trgDL.dataset)
        refArr = np.zeros((len(srcDL.dataset)))
        for i, ((srcTens, _), (trgTens, _)) in enumerate(zip(srcDL, trgDL)):
            srcTens = srcTens.to(device)
            trgTens = trgTens.to(device)
            refArr[i] = refFn(srcTens, trgTens, model, device)
        if avgIdxList is not None:
            refArr = np.mean(refArr, axis=0, keepdims=False)
    elif corrperm == 'perm':
        refArr = np.zeros((len(srcDL.dataset), len(trgDL.dataset)))
        for i, (srcTens, _) in enumerate(srcDL):
            srcTens = srcTens.to(device)
            for j, (trgTens, _) in enumerate(trgDL):
                trgTens = trgTens.to(device)
                refArr[i, j] = refFn(srcTens, trgTens, model, device)
        if maskDiag:
            assert refArr.shape[0] == refArr.shape[1], 'Non-square array'
            refArr = np.ma.masked_array(refArr, mask=np.eye(refArr.shape[0]))
        if avgIdxList is not None:
            for idx in avgIdxList[::-1]:
                refArr = np.mean(refArr, axis=idx, keepdims=False)
    return refArr

def brisque_metric(srcTens, *args):
    return piq.brisque(srcTens, data_range=1., reduction='none')

def clipiqa_metric(srcTens, device):
    return piq.CLIPIQA(data_range=1.).to(device)(srcTens)

def lpips_metric(srcTens, trgTens, *args):
    return piq.LPIPS(reduction='none')(srcTens, trgTens)

def ssim_metric(srcTens, trgTens, *args):
    return piq.ssim(srcTens, trgTens, data_range=1.)

def clip_metric(srcTens, trgTens, clipModel, device):
    srcEnc = clipModel.encode_image(srcTens.to(device))
    trgEnc = clipModel.encode_image(trgTens.to(device))
    return torch.nn.functional.cosine_similarity(srcEnc, trgEnc)

def linf_metric(srcTens, trgTens, *args):
    return torch.linalg.vector_norm(srcTens - trgTens, ord=np.inf)

def l2_metric(srcTens, trgTens, *args):
    return torch.linalg.vector_norm(srcTens - trgTens, ord=2)

def cosine_metric(srcTens, trgTens, *args):
    return torch.nn.functional.cosine_similarity(torch.flatten(srcTens), torch.flatten(trgTens), dim=0)


if __name__ == "__main__":
    run_main()
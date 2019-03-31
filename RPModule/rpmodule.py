import glob
import numpy as np
import cv2
import sys
from utils import torch_op
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
from .rputil import *
from open3d import *
import torch
import util
import copy
import logging
logger=logging.getLogger(__name__)

def horn87_np(src, tgt,weight=None):
    '''
    # src: [(k), 3, n]
    # tgt: [(k), 3, n]
    # weight: [(k), n]
    # return: 
    #   (R, t) ([(k),3,3], [(k),3,1])
    '''
    if len(src.shape) == 2 and len(tgt.shape) == 2:
        src, tgt = src[np.newaxis,:], tgt[np.newaxis,:]
    assert(src.shape[2] == tgt.shape[2])
    nPts = src.shape[2]
    k = src.shape[0]
    has_weight=False
    if weight is None:
        weight = np.ones([k,1,nPts])
    else:
        has_weight=True
        weight = weight.reshape(k,1,nPts)
    src_ = src
    tgt_ = tgt

    if has_weight:
        tgt_ = tgt_.copy()
        for i in range(k):
            tgt_[i] *= weight[i]
    M = np.matmul(src_, tgt_.transpose(0,2,1))
    R_ret = []
    for i in range(k):
        N = np.array([[M[i,0, 0] + M[i,1, 1] + M[i,2, 2], M[i,1, 2] - M[i,2, 1], M[i,2, 0] - M[i,0, 2], M[i,0, 1] - M[i,1, 0]], 
                        [M[i,1, 2] - M[i,2, 1], M[i,0, 0] - M[i,1, 1] - M[i,2, 2], M[i,0, 1] + M[i,1, 0], M[i,0, 2] + M[i,2, 0]], 
                        [M[i,2, 0] - M[i,0, 2], M[i,0, 1] + M[i,1, 0], M[i,1, 1] - M[i,0, 0] - M[i,2, 2], M[i,1, 2] + M[i,2, 1]], 
                        [M[i,0, 1] - M[i,1, 0], M[i,2, 0] + M[i,0, 2], M[i,1, 2] + M[i,2, 1], M[i,2, 2] - M[i,0, 0] - M[i,1, 1]]])
        v, u = np.linalg.eig(N)
        id = v.argmax()

        q = u[:, id]
        R_ret.append(np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])], 
                        [2*(q[2]*q[1]+q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]-q[0]*q[1])], 
                        [2*(q[3]*q[1]-q[0]*q[2]), 2*(q[3]*q[2]+q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]]).reshape(1,3,3))
    R_ret = np.concatenate(R_ret)
    return R_ret

def fit_horn87(allSP,allTP,allSN,allTN,allWP,allWN,mu):
    """
    allSP: source keypoint position
    allTP: target keypoint position
    allSN: source keypoint normal
    allTN: target keypoint normal
    allWP: point weight
    allWP: weight for position
    allWN: weight for normal
    mu:   a scalar weight for position
    """
    EPS = 1e-12
    SPmean = (allSP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
    allSPc = allSP - SPmean
    TPmean = (allTP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
    allTPc = allTP - TPmean
    allS = np.concatenate((allSPc,allSN))
    allT = np.concatenate((allTPc,allTN))
    allW = np.concatenate((allWP*mu,allWN))
    R_hat = horn87_np(allS.T,allT.T,allW)
    t_hat = -np.matmul(R_hat.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()
    R        = np.eye(4)
    R[:3,:3] = R_hat
    R[:3,3]  = t_hat
    return R

def fit_spectral(allSP,allTP,allSN,allTN,allWP,allWN,w_i1i2j1j2,mu,row,col,numFea_s,numFea_t):
    """
    allSP: source keypoint position
    allTP: target keypoint position
    allSN: source keypoint normal
    allTN: target keypoint normal
    allWP: point weight
    allWP: weight for position
    allWN: weight for normal
    w_i1i2j1j2: weight for correspondence pair
    mu:   a scalar weight for position
    row:   1st component of correspondence pair
    col:   2nd component of correspondence pair
    numFea_s: number of source keypoint
    numFea_t: number of target keypoint
    """
    num_alter = 5
    offset    = 50
    EPS       = 1e-12

    # compute center
    SPmean = (allSP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
    allSPc = allSP - SPmean
    TPmean = (allTP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
    allTPc = allTP - TPmean

    # compute R,t
    allS = np.concatenate((allSPc,allSN))
    allT = np.concatenate((allTPc,allTN))
    allW = np.concatenate((allWP*mu,allWN))
    R_hat = horn87_np(allS.T,allT.T,allW)
    t_hat = -np.matmul(R_hat.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()

    
    R_cur = R_hat.squeeze()
    t_cur = t_hat.squeeze()
    for j in range(num_alter):
        r_i1i2j1j2 = (mu*np.power(np.matmul(R_cur,allSPc.T)-allTPc.T,2).sum(0)+ \
            np.power(np.matmul(R_cur,allSN.T)-allTN.T,2).sum(0))
    
        a_i1i2j1j2 = allWP*(offset - r_i1i2j1j2)
        a_i1i2j1j2[a_i1i2j1j2<0] = 0
        a_i1i2j1j2 = a_i1i2j1j2.reshape(2,-1).sum(0)
        
        # construct the A matrix, compute most significant eigenvector
        A = csc_matrix((a_i1i2j1j2, (row,col)), shape=(numFea_s*numFea_t, numFea_s*numFea_t))
        A = A+A.T
        
        vals, u = sparse.linalg.eigs(A, k=1)
        u=u.real
        u /= np.linalg.norm(u)
        x_i1i2j1j2 = (u[row]*u[col]).squeeze()
        
        x_i1i2j1j2[x_i1i2j1j2<0] = 0
        x_i1i2j1j2 *= w_i1i2j1j2
        
        # initialize new weight 
        allW = np.tile(x_i1i2j1j2,4)
        # apply mu on weight corresponding to position
        allW[:len(allW)//2]*=mu

        # compute center
        allWP=allW[:len(allW)//2]
        SPmean = (allSP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
        allSPc = allSP - SPmean
        TPmean = (allTP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
        allTPc = allTP - TPmean

        allS = np.concatenate((allSPc,allSN))
        allT = np.concatenate((allTPc,allTN))

        # compute R,t using current weight
        R_tp = horn87_np(allS.T,allT.T,allW).reshape(3,3)
        t_tp = -np.matmul(R_tp.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()
    
        R_cur = R_tp
        t_cur = t_tp

    R        = np.eye(4)
    R[:3,:3] = R_cur
    R[:3,3]  = t_cur
    return R

def fit_irls(allSP,allTP,allSN,allTN,allWP,allWN,mu):
    """
    allSP: source keypoint position
    allTP: target keypoint position
    allSN: source keypoint normal
    allTN: target keypoint normal
    allWP: point weight
    allWP: weight for position
    allWN: weight for normal
    w_i1i2j1j2: weight for correspondence pair
    mu:   a scalar weight for position
    """
    num_reweighted = 5
    resSigma       = 1
    EPS            = 1e-12
    allW = np.concatenate((allWP*mu,allWN))
    for j in range(num_reweighted):
        # get the weight for position
        allWP=allW[:len(allW)//2]
        # compute center
        SPmean = (allSP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
        allSPc = allSP - SPmean
        TPmean = (allTP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
        allTPc = allTP - TPmean

        allS = np.concatenate((allSPc,allSN))
        allT = np.concatenate((allTPc,allTN))

        # compute R,t using current weight
        R_cur = horn87_np(allS.T,allT.T,allW)
        t_cur = -np.matmul(R_cur.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()

        # compute new weight
        residualPc = mu*np.power(np.matmul(R_cur.squeeze(),allSPc.T)-allTPc.T,2).sum(0)
        residualN = np.power(np.matmul(R_cur.squeeze(),allSN.T)-allTN.T,2).sum(0)
        residual = np.concatenate((residualPc,residualN))
        allW = allW * resSigma**2/(resSigma**2+residual)

    R        = np.eye(4)
    R[:3,:3] = R_cur
    R[:3,3]  = t_cur
    return R

def fit_irls_sm(allSP,allTP,allSN,allTN,allWP,allWN,w_i1i2j1j2,mu,row,col,numFea_s,numFea_t):
    """
    allSP: source keypoint position
    allTP: target keypoint position
    allSN: source keypoint normal
    allTN: target keypoint normal
    allWP: point weight
    allWP: weight for position
    allWN: weight for normal
    w_i1i2j1j2: weight for correspondence pair
    mu:   a scalar weight for position
    row:   1st component of correspondence pair
    col:   2nd component of correspondence pair
    numFea_s: number of source keypoint
    numFea_t: number of target keypoint
    """
    num_reweighted = 5
    num_alter      = 5
    resSigma       = 1
    offset         = 50
    EPS            = 1e-12
    allW = np.concatenate((allWP*mu,allWN))
    
    # initialize R,t
    for j in range(num_reweighted):
        # get the weight for position
        allWP=allW[:len(allW)//2]
        # compute center
        SPmean = (allSP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
        allSPc = allSP - SPmean
        TPmean = (allTP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
        allTPc = allTP - TPmean

        allS = np.concatenate((allSPc,allSN))
        allT = np.concatenate((allTPc,allTN))

        # compute R,t using current weight
        R_hat = horn87_np(allS.T,allT.T,allW)
        t_hat = -np.matmul(R_hat.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()
        # compute new weight
        residualPc = mu*np.power(np.matmul(R_hat.squeeze(),allSPc.T)-allTPc.T,2).sum(0)
        residualN = np.power(np.matmul(R_hat.squeeze(),allSN.T)-allTN.T,2).sum(0)
        residual = np.concatenate((residualPc,residualN))
        allW = allW * resSigma**2/(resSigma**2+residual)

    R_cur = R_hat.squeeze()
    t_cur = t_hat.squeeze()

    # alternate between spectral method and irls
    for j in range(num_alter):
        r_i1i2j1j2 = (mu*np.power(np.matmul(R_cur,allSPc.T)-allTPc.T,2).sum(0)+ \
            np.power(np.matmul(R_cur,allSN.T)-allTN.T,2).sum(0))
        
        a_i1i2j1j2 = np.tile(w_i1i2j1j2,2)*(offset - r_i1i2j1j2)
        a_i1i2j1j2[a_i1i2j1j2<0] = 0
        a_i1i2j1j2 = a_i1i2j1j2.reshape(2,-1).sum(0)
        
        # construct the A matrix, compute most significant eigenvector
        A = csc_matrix((a_i1i2j1j2, (row,col)), shape=(numFea_s*numFea_t, numFea_s*numFea_t))
        A = A+A.T
        
        vals, u = sparse.linalg.eigs(A, k=1)
        u=u.real

        u /= np.linalg.norm(u)
        x_i1i2j1j2 = (u[row]*u[col]).squeeze()

        x_i1i2j1j2[x_i1i2j1j2<0] = 0
        x_i1i2j1j2 *= w_i1i2j1j2

        # get new weight
        allW = np.tile(x_i1i2j1j2,4)
        # apply mu on weight corresponding to position
        allW[:len(allW)//2]*=mu
        
        for j in range(num_reweighted):
            # get the weight for position
            allWP=allW[:len(allW)//2]
            # compute center
            SPmean = (allSP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
            allSPc = allSP - SPmean
            TPmean = (allTP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
            allTPc = allTP - TPmean

            allS = np.concatenate((allSPc,allSN))
            allT = np.concatenate((allTPc,allTN))

            # compute R,t using current weight
            R_tp = horn87_np(allS.T,allT.T,allW).reshape(3,3)
            t_tp = -np.matmul(R_tp.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()

            # compute new weight
            residualPc = mu*np.power(np.matmul(R_tp.squeeze(),allSPc.T)-allTPc.T,2).sum(0)
            residualN = np.power(np.matmul(R_tp.squeeze(),allSN.T)-allTN.T,2).sum(0)
            residual = np.concatenate((residualPc,residualN))
            allW = allW * resSigma**2/(resSigma**2+residual)
        R_cur = R_tp
        t_cur = t_tp


    R        = np.eye(4)
    R[:3,:3] = R_cur
    R[:3,3]  = t_cur
    return R

def RelativePoseEstimation_helper(dataS, dataT,para):
    """
    Given two set of keypoint, this function estimate relative pose.
    dataS/dataT need to contain following entries:
        'pc': [k,3]
        'normal': [k,3]
        'feat': [k,32]
        'weight': [k]
    para is an instance of opts(please refer to rputil.py)
    """
    FEAT_SCALING = 100
    OBS_W        = 1.2
    # keypoint position
    sourcePC = dataS['pc']
    targetPC = dataT['pc']

    # keypoint normal
    sourceNormal = dataS['normal']
    targetNormal = dataT['normal']

    # keypoint weight, 1.0 for keypoint within observed region.
    sourcePCw = dataS['weight']
    targetPCw = dataT['weight']
    
    # keypoint descriptor
    sourceDess = dataS['feat'] / FEAT_SCALING.
    targetDess = dataT['feat'] / FEAT_SCALING.

    # if two few keypoints, directly return
    if sourcePC.shape[0] < 3 or targetPC.shape[0]<3:
        logger.info(f"stage-1: not enough!\n return identity: {np.eye(3)}")
        return np.eye(4)

    numFea_s = sourcePC.shape[0]
    numFea_t = targetPC.shape[0]

    # compute wij based on descriptor distance. 
    pcWij = np.expand_dims(sourcePCw,1)*np.expand_dims(targetPCw,0)
    dij = np.power(np.expand_dims(sourceDess,1)-np.expand_dims(targetDess,0),2).sum(2)
    sigmaij = np.ones(pcWij.shape)*para.sigmaFeat
    sigmaij[pcWij==1] = para.sigmaFeat/OBS_W # require smaller distance if both keypoints are within observed region
    wij = np.exp(np.divide(-dij, 2*np.power(sigmaij/5, 2)))
    nm = np.linalg.norm(wij, axis=1, keepdims=True)
    equalzero = (nm==0)
    nm[equalzero] = 1
    wij/=nm
    wij[equalzero.squeeze(),:]=0

    logger.info(f"wij great zero : {sum(wij.sum(1)!=0)}/{sum(wij.sum(0)!=0)}\n")

    # prune for top K correspondence for simplicity
    topK = min(para.topK, wij.shape[1]-1)
    topIdx = np.argpartition(-wij,topK,axis=1)[:, :topK]
    
    corres = np.zeros([2, numFea_s * topK])
    corres[0, :] = np.arange(numFea_s).repeat(topK)
    corres[1, :] = topIdx.flatten()
    corres = corres.astype('int')
    num_corres = corres.shape[1]

    if num_corres < 3:
        logger.info('stage0: not enough!\n\n')
        return np.eye(4)

    # compute wi1i2j1j2
    idx = np.tile(np.arange(num_corres),num_corres).reshape(-1,num_corres)
    idy = idx.T
    valid = (idx > idy)
    idx = idx[valid]
    idy = idy[valid]

    # distance consistency
    pci1 = sourcePC[corres[0,idy],:]
    pcj1 = targetPC[corres[1,idy],:]
    pci2 = sourcePC[corres[0,idx],:]
    pcj2 = targetPC[corres[1,idx],:]

    ni1 = sourceNormal[corres[0,idy],:]
    nj1 = targetNormal[corres[1,idy],:]
    ni2 = sourceNormal[corres[0,idx],:]
    nj2 = targetNormal[corres[1,idx],:]

    dis_s = np.linalg.norm(pci1 - pci2,axis=1)
    dis_t = np.linalg.norm(pcj1 - pcj2,axis=1)
    d_i1i2j1j2 = np.power(dis_s - dis_t,2)
    

    filterIdx = np.logical_and(d_i1i2j1j2 < np.power(para.distThre,2),np.minimum(dis_s,dis_t) > 1.5*np.power(para.distSepThre,2))
    logger.info(f"dist delete:{(~filterIdx).sum()}")
    if filterIdx.sum() < 3:
        logger.info('stage1: not enough!\n\n')
        return np.eye(4)
    
    # collect index that passed the distance test
    idx = idx[filterIdx]
    idy = idy[filterIdx]
    pci1 = pci1[filterIdx]
    pcj1 = pcj1[filterIdx]
    pci2 = pci2[filterIdx]
    pcj2 = pcj2[filterIdx]
    ni1 = ni1[filterIdx]
    nj1 = nj1[filterIdx]
    ni2 = ni2[filterIdx]
    nj2 = nj2[filterIdx]
    d_i1i2j1j2 = d_i1i2j1j2[filterIdx]

    # angle consistency
    e1 = (pci1-pci2)
    e2 = (pcj1-pcj2)
    e1 /= np.linalg.norm(e1,axis=1,keepdims=True)
    e2 /= np.linalg.norm(e2,axis=1,keepdims=True)

    
    alpha_i1i2j1j2 = np.power(np.arccos((ni1*ni2).sum(1).clip(-1,1)) - np.arccos((nj1*nj2).sum(1).clip(-1,1)),2)
    beta_i1i2j1j2 = np.power(np.arccos((ni1*e1).sum(1).clip(-1,1)) - np.arccos((nj1*e2).sum(1).clip(-1,1)),2)
    gamma_i1i2j1j2 = np.power(np.arccos((ni2*e1).sum(1).clip(-1,1)) - np.arccos((nj2*e2).sum(1).clip(-1,1)),2)
    
    filterIdx = np.logical_and.reduce((alpha_i1i2j1j2 < np.power(para.angleThre,2),
                    beta_i1i2j1j2 < np.power(para.angleThre,2),
                    gamma_i1i2j1j2 < np.power(para.angleThre,2)))
    

    logger.info(f"angle delete:{(~filterIdx).sum()}")
    if filterIdx.sum() < 3:
        logger.info('stage2: not enough  !\n\n')
        logger.info(f"return identity: {np.eye(3)}")
        return np.eye(4)

    # collect index that passed the angle test
    idx = idx[filterIdx]
    idy = idy[filterIdx]
    d_i1i2j1j2 = d_i1i2j1j2[filterIdx]
    alpha_i1i2j1j2 = alpha_i1i2j1j2[filterIdx]
    beta_i1i2j1j2 = beta_i1i2j1j2[filterIdx]
    gamma_i1i2j1j2 = gamma_i1i2j1j2[filterIdx]
    
    f_i1j1 = wij[corres[0,idy],corres[1,idy]]
    f_i2j2 = wij[corres[0,idx],corres[1,idx]]
    logger.info(f"num f not equal zero:{sum((f_i1j1*f_i2j2)!=0)}\n")

    w_i1i2j1j2 = f_i1j1*f_i2j2*np.exp(-d_i1i2j1j2/(2*para.sigmaDist**2)\
        -alpha_i1i2j1j2/(2*para.sigmaAngle1**2)\
        -beta_i1i2j1j2/(2*para.sigmaAngle2**2)\
        -gamma_i1i2j1j2/(2*para.sigmaAngle2**2))
    
    pi1w=sourcePCw[corres[0,idy]]
    pj1w=targetPCw[corres[1,idy]]
    pi2w=sourcePCw[corres[0,idx]]
    pj2w=targetPCw[corres[1,idx]]
    ww_i1i2j1j2=pi1w*pi2w*pj1w*pj2w
    w_i1i2j1j2[ww_i1i2j1j2!=1] *= 0.6

    if (w_i1i2j1j2!=0).sum() < 1:
        logger.info('stage3: not enough  !\n\n')
        logger.info(f"return identity: {np.eye(3)}")
        return np.eye(4)

    pi1=sourcePC[corres[0,idy],:]
    pj1=targetPC[corres[1,idy],:]
    pi2=sourcePC[corres[0,idx],:]
    pj2=targetPC[corres[1,idx],:]
    ni1=sourceNormal[corres[0,idy],:]
    nj1=targetNormal[corres[1,idy],:]
    ni2=sourceNormal[corres[0,idx],:]
    nj2=targetNormal[corres[1,idx],:]


    allSP = np.concatenate((pi1,pi2))
    allTP = np.concatenate((pj1,pj2))
    allSN = np.concatenate((ni1,ni2))
    allTN = np.concatenate((nj1,nj2))
    allWP = np.concatenate((w_i1i2j1j2, w_i1i2j1j2))
    allWN = allWP.copy()
    
    if para.method == 'horn87':
        return fit_horn87(allSP,allTP,allSN,allTN,allWP,allWN,para.mu)

    elif para.method == 'spectral':
        row = corres[0,idy]*numFea_t+corres[1,idy]
        col = corres[0,idx]*numFea_t+corres[1,idx]
        return fit_spectral(allSP,allTP,allSN,allTN,allWP,allWN,w_i1i2j1j2,para.mu,row,col,numFea_s,numFea_t)
    
    elif para.method == 'irls':
        return fit_irls(allSP,allTP,allSN,allTN,allWP,allWN,para.mu)

    elif para.method == 'irls+sm':
        row = corres[0,idy]*numFea_t+corres[1,idy]
        col = corres[0,idx]*numFea_t+corres[1,idx]
        return fit_irls_sm(allSP,allTP,allSN,allTN,allWP,allWN,w_i1i2j1j2,para.mu,row,col,numFea_s,numFea_t)

    else:
        raise Exception("unknown method!")


def getMatchingPrimitive(dataS, dataT, dataset, representation, doCompletion):
    """
    - detect keypoint
    - get keypoint 3d position/normal/feature
    """
    # compute keypoint
    if 'suncg' in dataset or 'matterport' in dataset:
        pts,ptsNorm,ptsW,ptt,pttNorm,pttW = getKeypoint(dataS['rgb'],dataT['rgb'],dataS['feat'],dataT['feat'])
    elif 'scannet' in dataset:
        pts,ptsNorm,ptsW,ptt,pttNorm,pttW = getKeypoint_kinect(dataS['rgb'],dataT['rgb'],dataS['feat'],dataT['feat'],dataS['rgb_full'],dataT['rgb_full'])

    # early return if too few keypoint detected
    if pts is None or ptt is None or pts.shape[1]<2 or ptt.shape[1]<2:
        return None,None,None,None,None,None,None,None

    # get the 3d location of matches
    pts3d,ptsns = getPixel(dataS['depth'],dataS['normal'],pts,dataset=dataset,representation=representation)
    ptt3d,ptsnt = getPixel(dataT['depth'],dataT['normal'],ptt,dataset=dataset,representation=representation) 

    # interpolate the nn feature map to get feature vectors
    dess = torch_op.npy(interpolate(dataS['feat'],torch_op.v(ptsNorm))).T
    dest = torch_op.npy(interpolate(dataT['feat'],torch_op.v(pttNorm))).T

    if not doCompletion:
        # filter out those keypoint from unobserved region
        pts3d,ptsns,dess,ptsW = pts3d[:,ptsW==1],ptsns[ptsW==1],dess[ptsW==1],ptsW[ptsW==1]
        ptt3d,ptsnt,dest,pttW = ptt3d[:,pttW==1],ptsnt[pttW==1],dest[pttW==1],pttW[pttW==1]
    return pts3d,ptt3d,ptsns,ptsnt,dess,dest,ptsW,pttW

def RelativePoseEstimation(dataS, dataT, para, dataset, representation,maskMethod,doCompletion=True,index=None):
    """
    Given two completed scans, this function first found keypoints, which includes
    position,normal,and feature vectors, then estimate relative pose using proposed
    irls+spectral geometric matching.
    dataS/dataT need to contain following entries:
        'rgb': [160,640,3]
        'normal': [160,640,3]
        'depth': [160,640]
        'feat': [32,160,640]
        'rgb_full': [480,640,3] (for scannet, we use original sized scan(480x640) to detect keypoint.)
    """
    # initialize default return value as Identity
    R_hat=np.eye(4)

    # extract matching primitive from image representation
    pts3d,ptt3d,ptsns,ptsnt,dess,dest,ptsW,pttW = getMatchingPrimitive(dataS,dataT,dataset,representation,doCompletion)

    # early return if too few keypoint detected
    if pts3d is None or ptt3d is None or pts3d.shape[0]<2 or pts3d.shape[0]<2:
        logger.info(f"no pts detected or less than 2 keypoint detected, return identity: {np.eye(3)}")
        return R_hat
    
    # estimate relative pose
    R_hat = RelativePoseEstimation_helper({'pc':pts3d.T,'normal':ptsns,'feat':dess,'weight':ptsW},{'pc':ptt3d.T,'normal':ptsnt,'feat':dest,'weight':pttW},para)

    return R_hat


def RelativePoseEstimationViaCompletion(net, data_s, data_t, args):
    """
    The main algorithm:
    Given two set of scans, alternate between scan completion and pairwise matching 
    args need to contain: 
        snumclass: number of semantic class
        featureDim: feature dimension
        outputType: ['rgb':color,'d':depth,'n':normal,'s':semantic,'f':feature]
        maskMethod: ['second']
        alterStep:
        dataset:
        para:
    """
    EPS = 1e-12
    args.idx_f_start = 0        
    if 'rgb' in args.outputType:
        args.idx_f_start += 3
    if 'n' in args.outputType:
        args.idx_f_start += 3
    if 'd' in args.outputType:
        args.idx_f_start += 1
    if 's' in args.outputType:
        args.idx_f_start += args.snumclass
    if 'f' in args.outputType:
        args.idx_f_end   = args.idx_f_start + args.featureDim

    with torch.set_grad_enabled(False):
        R_hat=np.eye(4)
        
        # get the complete scans
        complete_s=torch.cat((torch_op.v(data_s['rgb']),torch_op.v(data_s['norm']),torch_op.v(data_s['depth']).unsqueeze(2)),2).permute(2,0,1).unsqueeze(0)
        complete_t=torch.cat((torch_op.v(data_t['rgb']),torch_op.v(data_t['norm']),torch_op.v(data_t['depth']).unsqueeze(2)),2).permute(2,0,1).unsqueeze(0)
        
        # apply the observation mask
        view_s,mask_s,_ = util.apply_mask(complete_s.clone(),args.maskMethod)
        view_t,mask_t,_ = util.apply_mask(complete_t.clone(),args.maskMethod)
        mask_s=torch_op.npy(mask_s[0,:,:,:]).transpose(1,2,0)
        mask_t=torch_op.npy(mask_t[0,:,:,:]).transpose(1,2,0)

        # append mask for valid data
        tpmask = (view_s[:,6:7,:,:]!=0).float().cuda()
        view_s=torch.cat((view_s,tpmask),1)
        tpmask = (view_t[:,6:7,:,:]!=0).float().cuda()
        view_t=torch.cat((view_t,tpmask),1)

        for alter_ in range(args.alterStep):
            # warp the second scan using current transformation estimation
            view_t2s=torch_op.v(util.warping(torch_op.npy(view_t),np.linalg.inv(R_hat),args.dataset))
            view_s2t=torch_op.v(util.warping(torch_op.npy(view_s),R_hat,args.dataset))
            # append the warped scans
            view0 = torch.cat((view_s,view_t2s),1)
            view1 = torch.cat((view_t,view_s2t),1)

            # generate complete scans
            f=net(torch.cat((view0,view1)))
            f0=f[0:1,:,:,:]
            f1=f[1:2,:,:,:]
            
            data_sc,data_tc={},{}
            # replace the observed region with gt depth/normal
            data_sc['normal'] = (1-mask_s)*torch_op.npy(f0[0,3:6,:,:]).transpose(1,2,0)+mask_s*data_s['norm']
            data_tc['normal'] = (1-mask_t)*torch_op.npy(f1[0,3:6,:,:]).transpose(1,2,0)+mask_t*data_t['norm']
            data_sc['normal']/= (np.linalg.norm(data_sc['normal'],axis=2,keepdims=True)+EPS)
            data_tc['normal']/= (np.linalg.norm(data_tc['normal'],axis=2,keepdims=True)+EPS)
            data_sc['depth']  = (1-mask_s[:,:,0])*torch_op.npy(f0[0,6,:,:])+mask_s[:,:,0]*data_s['depth']
            data_tc['depth']  = (1-mask_t[:,:,0])*torch_op.npy(f1[0,6,:,:])+mask_t[:,:,0]*data_t['depth']

            data_sc['obs_mask']   = mask_s.copy()
            data_tc['obs_mask']   = mask_t.copy()
            data_sc['rgb']    = (mask_s*data_s['rgb']*255).astype('uint8')
            data_tc['rgb']    = (mask_t*data_t['rgb']*255).astype('uint8')
            
            # for scannet, we use the original size rgb image(480x640) to extract sift keypoint
            if 'scannet' in args.dataset:
                data_sc['rgb_full'] = (data_s['rgb_full']*255).astype('uint8')
                data_tc['rgb_full'] = (data_t['rgb_full']*255).astype('uint8')
                data_sc['depth_full'] = data_s['depth_full']
                data_tc['depth_full'] = data_t['depth_full']
            
            # extract feature maps
            f0_feat=f0[:,args.idx_f_start:args.idx_f_end,:,:]
            f1_feat=f1[:,args.idx_f_start:args.idx_f_end,:,:]
            data_sc['feat']=f0_feat.squeeze(0)
            data_tc['feat']=f1_feat.squeeze(0)

            para_this = copy.copy(args.para)
            para_this.sigmaAngle1 = para_this.sigmaAngle1[alter_]
            para_this.sigmaAngle2 = para_this.sigmaAngle2[alter_]
            para_this.sigmaDist = para_this.sigmaDist[alter_]
            para_this.sigmaFeat = para_this.sigmaFeat[alter_]
            # run relative pose module to get next estimate
            R_hat = RelativePoseEstimation(data_sc,data_tc,para_this,args.dataset,args.representation,doCompletion=args.completion,maskMethod=args.maskMethod,index=None)
            
    return R_hat


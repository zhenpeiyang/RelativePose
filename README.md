
# Extreme Relative Pose Estimation for RGB-D Scans via Scene Completion
Pytorch implementation of paper ["Extreme Relative Pose Estimation for RGB-D Scans via Scene Completion"](https://www.google.com)

![alt tag](overview.png)

## Prerequisites:
* pytorch (>0.4)
* open3d 
* scipy,sklearn

##  Folder Organization
please make sure to have following folder structure:
``` shell
RelativePose/
    data/
        dataList/
        pretrained_model/
    experiments/
    tmp/
```

##  Dataset Download
uploading soon

images: [suncg](https://drive.google.com/open?id=1Gr-BLYrMm7zM_Q0rum_uKM_TwMJg10Mf),[matterport](https://drive.google.com/open?id=12PcZK89YX7zbR2sP_vjT-n8NyqakNNge),[scannet](https://drive.google.com/open?id=1lwF7gTQg4rS5-lJ-cVXHf7Uch0vRvoc1)<br/>
data list: [suncg](https://drive.google.com/open?id=1mJ8l8z9nlrtG5MGrb5y1Ww4Cu9Zhg4ac),[matterport](https://drive.google.com/open?id=1-CHcXCT-J--JuFDXDTv_WfUDHWTdyX3A),[scannet](https://drive.google.com/open?id=1bKDSdnjmMFjXgpWlFjBjxjo-VrSg5xmo)<br/>
pretrained model: [suncg](https://drive.google.com/open?id=1MCovN5WtQWKd6GeN0HNZ-vrcg-gefZ8B),[matterport](https://drive.google.com/open?id=1TGZwBuALDxzkRQXbn1oBZ2N804FAXmUP),[scannet](https://drive.google.com/open?id=1KM_a6kIn-TrJ_DM87dDugzjVotA3BjhI)<br/>
Images should be uncompressed under data/ folder. The data list contains the split used in our experiments, and should be placed under data/dataList/ folder. The pretrained model should be placed under data/pretrained_model/ folder. 
## Usage
### training feature network
```
# suncg 
python mainFeatureLearning.py --exp featSuncg --g --batch_size=2 --featurelearning=1 --maskMethod=second --resume --dataList=suncg --outputType=rgbdnsf --snumclass=15
# matterport 
python mainFeatureLearning.py --exp featMatterport --g --batch_size=2 --featurelearning=1 --maskMethod=second --resume --dataList=matterport --outputType=rgbdnsf --snumclass=15
# scannet 
python mainFeatureLearning.py --exp featScannet --g --batch_size=2 --featurelearning=1 --maskMethod=kinect --resume --dataList=scannet --outputType=rgbdnsf --snumclass=21
```

### training completion module
```
# suncg 
python mainPanoCompletion2view.py --exp compSuncg--g --batch_size=2 --featurelearning=1 --maskMethod=second --resume --dataList=suncg --outputType=rgbdnsf --snumclass=15
# matterport 
python mainPanoCompletion2view.py --exp compMatterport --g --batch_size=2 --featurelearning=1 --maskMethod=second --resume --dataList=matterport --outputType=rgbdnsf --snumclass=15
# scannet 
python mainPanoCompletion2view.py --exp compScannet  --g --batch_size=2 --featurelearning=1 --maskMethod=kinect --resume --dataList=scannet --outputType=rgbdnsf --snumclass=21 --useTanh=0
```

### train relative pose module
```
python trainRelativePoseModuleRecFD.py --exp fd_param --dataset=suncg --snumclass=15 --split=val --para_init={param for previous iter} --rlevel={recurrent level}
```

The trained parameters for relative pose module are provided in data/relativePoseModule/

## Evaluation
```
python evaluation.py --dataList={suncg,matterport,scannet} --method={ours,ours_nr,ours_nc,gs,cgs,super4pcs} --exp=eval --num_repeat=10 --para={param file}
```

## Author

Zhenpei Yang





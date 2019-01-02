
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
        relativePoseModule/
        pretrained_model/
    experiments/
    tmp/
```

##  Dataset Download
uploading soon

images: [suncg](https://www.google.com),[matterport](https://www.google.com),[scannet](https://www.google.com)<br/>
data list: [suncg](https://www.google.com),[matterport](https://www.google.com),[scannet](https://www.google.com)<br/>
pretrained model: [suncg](https://www.google.com),[matterport](https://www.google.com),[scannet](https://www.google.com)<br/>
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





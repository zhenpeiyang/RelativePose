
# Extreme Relative Pose Estimation for RGB-D Scans via Scene Completion
Pytorch implementation of paper ["Extreme Relative Pose Estimation for RGB-D Scans via Scene Completion"](https://arxiv.org/abs/1901.00063)

![alt tag](overview.png)

## Prerequisites:
* pytorch (>0.4)
* open3d
* scipy,sklearn
* [torchvision](https://github.com/warmspringwinds/vision/tree/eb6c13d3972662c55e752ce7a376ab26a1546fb5)

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
images: [suncg](https://drive.google.com/file/d/1SHWCUNG6gdNRRl1kw0sXhm9KMnI6I7v2/view?usp=sharing),[matterport](https://drive.google.com/file/d/1LlscT6ejo1xFgsz5BTOpLQki1lMXsokr/view?usp=sharing),[scannet](https://drive.google.com/file/d/1HNbMvV_ybWesKhZby0FoPfk-DFbl9JJ5/view?usp=sharing)<br/>
data list: [suncg](https://drive.google.com/open?id=1mJ8l8z9nlrtG5MGrb5y1Ww4Cu9Zhg4ac),[matterport](https://drive.google.com/open?id=1-CHcXCT-J--JuFDXDTv_WfUDHWTdyX3A),[scannet](https://drive.google.com/open?id=1bKDSdnjmMFjXgpWlFjBjxjo-VrSg5xmo)<br/>
pretrained model: [suncg](https://drive.google.com/drive/folders/1YQTqr28JHdE3YWNW1GxSQUkGKnLIiBkd?usp=sharing),[matterport](https://drive.google.com/drive/folders/1G_iseDs0KtMU_NcBBPuyVm5O9NZsKvOE?usp=sharing),[scannet](https://drive.google.com/drive/folders/14LWD1KWr3GFe-KrLn-ZPWWMzrw4LuXAB?usp=sharing)<br/>
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
Noted that you need place Super4PCS binary under the RelativePose/ in order to run its evaluation.

## Author

Zhenpei Yang





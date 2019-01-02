import numpy as np

nViews = 2
pano_width = 640
pano_height = 160

# suncg color encoding
suncgCats=['ceiling','wall','floor','window','bed','door','cabinet','chair','sofa','television','table','object','computer','lamp','curtain']
suncg_color_palette =[
    (209,97,0), # ceiling
    (4,247,87),# wall
    (255,181,0),# floor
    (0,0,53),# window
    (254,255,230),# bed
    (163,200,201),# door
    (87,83,41), # cabinet
    (48,0,24), # chair
    (0,137,65),# sofa
    (0,194,160), # television
    (111,0,98),# table
    (82,84,163), #object
    (90,0,7), # computer
    (107,0,44), # lamp
    (58,36,101) # curtain
]
suncg_color_palette = np.stack(suncg_color_palette)

# matterport color encoding
matterportCats=['unknown','wall','floor','chair','door','table','picture','cabinet','window','sofa','bed','plant',
    'sink','stairs','ceiling','toilet','mirror','bathtub','counter','railing','shelving'] # swap computer to 
matterport_color_palette =[
    [143, 176, 255], # unknown
    [  4, 247,  87],# wall
    [255, 181,   0],# floor
    [48,  0, 24],# chair
    [163, 200, 201],# door
    [111,   0,  98],# table
    [161, 194, 153],# picture
    [55, 33,  1],# cabinet
    [ 0,  0, 53],# window
    [  0, 137,  65],# sofa
    [254, 255, 230],# bed
    [ 79, 198,   1],# plant
    [167, 117,   0],# sink
    [128, 150, 147],# stairs
    [209,  97,   0],# ceiling 
    [122,  73,   0],# toilet
    [ 28, 230, 255],# mirrow
    [255, 138, 154],# bathtub
    [146,  35,  41],# counter
    [255, 246, 159],# railing
    [255,  47, 128],# shelving
]
matterport_color_palette = np.stack(matterport_color_palette)

# scannet color encoding
scannetCats=['unknown','wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','desk'
,'curtain','refrigerator','shower curtain','toilet','sink','bathtub','otherfurn']
scannet_color_palette =[
       (0,   0,   0),       # unknown
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (247, 182, 210),		# desk
       (219, 219, 141),		# curtain
       (255, 127, 14), 		# refrigerator
       (158, 218, 229),		# shower curtain
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (227, 119, 194),		# bathtub
       (82, 84, 163),  		# otherfurn
    ]
scannet_color_palette = np.stack(scannet_color_palette)


---
layout: post
title: Image Classification- Cassava Leaf Disease
date: 2021-02-17 00:00:00 +0300
description: using Deep Neural Networks

img: cassavaCOLLAGE.jpg 
tags: [Deep Learning, Neural Networks, Machine Learning]
---
# Image Classification- Cassava Leaf Disease 

### Comparison of different neural network models using PyTorch

Image classification is a supervised learning problem: define a set of target classes (objects to identify in images), and train a model to recognize them using labelled example photos. [[Source]](https://developers.google.com/machine-learning/practica/image-classification)

Deep learning, a subset of machine learning algorithms, is good at recognising patterns. Hence, it is widely used for image classification. The adjective "deep" in deep learning refers to the use of multiple layers in the network, where each layer progressively extracts higher-level features from the raw input. Deep learning models can have different architectures.

This is a project to classify the images of cassava plant leaves into five categories based on the disease affecting them. The dataset consist of 21,367 labeled images of cassava plant leaves, obtained from a [Cassava Leaf Disease Classification competition in Kaggle](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview). The project aims to classify the images using the following neural network architectures and compare their performances:


1.   Feed Forward Neural Network
2.   Convolutional Neural Network
3.   Resnet34 pretrained architecture
4.   Efficientnet-B4 pretrained architecture
5.   Resnext50_32x4d pretrained architecture


The project is inspired from the [Zero to GANs](https://jovian.ai/learn/deep-learning-with-pytorch-zero-to-gans) course by the data science learning platform, [Jovian](https://www.jovian.ai).

## Data description
The dataset consist of 21,367 labeled images of cassava plant leaves collected during a regular survey in Uganda. Each image is an RGB image of size 600 x 800 pixels. Most images were crowdsourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala. This is in a format that most realistically represents what farmers would need to diagnose in real life.

## Data exploration
Let us begin by downloading the dataset.


```python
!pip install jovian opendatasets --upgrade --quiet
```


```python
import opendatasets as od
#dataset_url='https://www.kaggle.com/aswiniabraham/cassava-leaf-disease-image-folders-600x800'

dataset_url='https://www.kaggle.com/c/cassava-leaf-disease-classification/data'
od.download(dataset_url)
```

    Please provide your Kaggle credentials to download this dataset. Learn more: http://bit.ly/kaggle-creds
    Your Kaggle username: aswiniabraham
    Your Kaggle Key: ········
    

      0%|          | 10.0M/5.76G [00:00<01:04, 96.4MB/s]

    Downloading cassava-leaf-disease-classification.zip to ./cassava-leaf-disease-classification
    

    100%|██████████| 5.76G/5.76G [00:58<00:00, 105MB/s] 

    
    

    
    


```python
%%time
from zipfile import ZipFile

with ZipFile('cassava-leaf-disease-classification/cassava-leaf-disease-classification.zip') as zipper:
    zipper.extractall('./data')
```

    CPU times: user 26.1 s, sys: 10.6 s, total: 36.7 s
    Wall time: 2min 1s
    


```python
import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
%matplotlib inline

matplotlib.rcParams['figure.facecolor'] = '#ffffff'
```


```python
os.listdir('./data')
```




    ['train_images',
     'test_tfrecords',
     'sample_submission.csv',
     'label_num_to_disease_map.json',
     'train.csv',
     'train_tfrecords',
     'test_images']



The extracted dataset contains mainly the following folders/files:

1. **train_images:** contains images in jpg format for training
2.   **train.csv:** contains the filename of the image and the ID code of the disease.
3. **label_num_to_disease_map.json:** The mapping between each disease code and the real disease name.



```python
images_labels = pd.read_csv('./data/train.csv')
images_labels.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000015157.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000201771.jpg</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100042118.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000723321.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000812911.jpg</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
label_map = pd.read_json('./data/label_num_to_disease_map.json', orient='index')
label_map
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cassava Bacterial Blight (CBB)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cassava Brown Streak Disease (CBSD)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cassava Green Mottle (CGM)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cassava Mosaic Disease (CMD)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Healthy</td>
    </tr>
  </tbody>
</table>
</div>




```python
os.listdir('./data/train_images')
```




    ['2528148363.jpg',
     '3174632328.jpg',
     '2406694792.jpg',
     '2530575673.jpg',
     '2387502649.jpg',
     '3882641600.jpg',
     '2955761671.jpg',
     '2468469374.jpg',
     '425020385.jpg',
     '760916412.jpg',
     '3923213129.jpg',
     '2164751945.jpg',
     '3907936185.jpg',
     '2041164461.jpg',
     '2228874754.jpg',
     '881727330.jpg',
     '117969308.jpg',
     '672634169.jpg',
     '1369983957.jpg',
     '2873954031.jpg',
     '1123691141.jpg',
     '2523040233.jpg',
     '2146353282.jpg',
     '2958183891.jpg',
     '1535573498.jpg',
     '417083161.jpg',
     '3012872087.jpg',
     '3722905230.jpg',
     '2771630964.jpg',
     '1044411903.jpg',
     '1485335587.jpg',
     '921975951.jpg',
     '2240245430.jpg',
     '1100532272.jpg',
     '1383472035.jpg',
     '1218546888.jpg',
     '3470374968.jpg',
     '1990169159.jpg',
     '306351373.jpg',
     '1959890279.jpg',
     '3639594578.jpg',
     '1861354699.jpg',
     '3878867821.jpg',
     '3222825564.jpg',
     '1179913576.jpg',
     '4159206286.jpg',
     '498277992.jpg',
     '2026049227.jpg',
     '1819546557.jpg',
     '304370319.jpg',
     '260241921.jpg',
     '822187943.jpg',
     '1378105752.jpg',
     '2944944834.jpg',
     '335373931.jpg',
     '3013901544.jpg',
     '2923119977.jpg',
     '192312490.jpg',
     '3959266436.jpg',
     '2055831715.jpg',
     '3953560426.jpg',
     '3994140306.jpg',
     '133458128.jpg',
     '4045262215.jpg',
     '1849702886.jpg',
     '1067829359.jpg',
     '3986821371.jpg',
     '3061399544.jpg',
     '3701789990.jpg',
     '3671711005.jpg',
     '2365326310.jpg',
     '608243264.jpg',
     '110366351.jpg',
     '1302328544.jpg',
     '1666776898.jpg',
     '2034410033.jpg',
     '2255664961.jpg',
     '778148477.jpg',
     '365293292.jpg',
     '1638509786.jpg',
     '1150408914.jpg',
     '2668114223.jpg',
     '3575391896.jpg',
     '1305572752.jpg',
     '2509756796.jpg',
     '2247014271.jpg',
     '1135103288.jpg',
     '3360732633.jpg',
     '4284260544.jpg',
     '3942046671.jpg',
     '4012035359.jpg',
     '3371574415.jpg',
     '3436142956.jpg',
     '1931683048.jpg',
     '2028194351.jpg',
     '68764231.jpg',
     '4201154907.jpg',
     '2704844325.jpg',
     '2742945698.jpg',
     '879939600.jpg',
     '3687069352.jpg',
     '2446013457.jpg',
     '1806336706.jpg',
     '2570672557.jpg',
     '1397843043.jpg',
     '2169305287.jpg',
     '3295623672.jpg',
     '1948579246.jpg',
     '4281502207.jpg',
     '976680630.jpg',
     '1643316247.jpg',
     '3543731892.jpg',
     '897638042.jpg',
     '840889363.jpg',
     '2260738205.jpg',
     '3192927904.jpg',
     '514376645.jpg',
     '3554722517.jpg',
     '1514150347.jpg',
     '2916471006.jpg',
     '759934720.jpg',
     '2755812259.jpg',
     '2633910453.jpg',
     '3762629314.jpg',
     '2358762057.jpg',
     '2616538732.jpg',
     '1697663817.jpg',
     '788479472.jpg',
     '680852379.jpg',
     '2434587882.jpg',
     '2967526329.jpg',
     '1620888831.jpg',
     '3475528953.jpg',
     '1435727833.jpg',
     '376932028.jpg',
     '211225277.jpg',
     '1775446541.jpg',
     '762652990.jpg',
     '140439386.jpg',
     '2098671662.jpg',
     '3406705095.jpg',
     '1892378935.jpg',
     '854627770.jpg',
     '2177328825.jpg',
     '3582059053.jpg',
     '415809799.jpg',
     '1376689947.jpg',
     '3856935048.jpg',
     '1011571614.jpg',
     '2294615213.jpg',
     '628791743.jpg',
     '473941755.jpg',
     '3337773185.jpg',
     '2644010748.jpg',
     '3782729754.jpg',
     '4239047736.jpg',
     '4222515459.jpg',
     '1172621803.jpg',
     '809808575.jpg',
     '2471372576.jpg',
     '3492042850.jpg',
     '3662018109.jpg',
     '545206870.jpg',
     '3718836957.jpg',
     '1284343377.jpg',
     '2406862248.jpg',
     '1794977133.jpg',
     '4186633206.jpg',
     '2684757873.jpg',
     '1845227647.jpg',
     '2196411231.jpg',
     '4192717491.jpg',
     '3715954473.jpg',
     '3021759885.jpg',
     '1227531167.jpg',
     '1136746572.jpg',
     '1616161207.jpg',
     '3541933479.jpg',
     '1169555364.jpg',
     '695172067.jpg',
     '2301739298.jpg',
     '35045621.jpg',
     '1428921734.jpg',
     '3504669411.jpg',
     '1102683272.jpg',
     '2757749488.jpg',
     '76754228.jpg',
     '1179134330.jpg',
     '212825302.jpg',
     '3584346239.jpg',
     '744676370.jpg',
     '1772264324.jpg',
     '2373129151.jpg',
     '3313790300.jpg',
     '1133309905.jpg',
     '2594655853.jpg',
     '1225715359.jpg',
     '3161586140.jpg',
     '3015228467.jpg',
     '1814561933.jpg',
     '3305328622.jpg',
     '2120113052.jpg',
     '561444501.jpg',
     '2916689767.jpg',
     '255319476.jpg',
     '905336234.jpg',
     '1361142044.jpg',
     '171026079.jpg',
     '1737101644.jpg',
     '2160006188.jpg',
     '29400522.jpg',
     '3859714996.jpg',
     '1072171567.jpg',
     '913436788.jpg',
     '3983622976.jpg',
     '1148829591.jpg',
     '3574527119.jpg',
     '633010478.jpg',
     '3246462405.jpg',
     '2734186624.jpg',
     '1301425081.jpg',
     '1856743882.jpg',
     '2653588387.jpg',
     '1558894441.jpg',
     '1466991020.jpg',
     '446546740.jpg',
     '249583013.jpg',
     '955794248.jpg',
     '2052500079.jpg',
     '3874234483.jpg',
     '3306255309.jpg',
     '1495877886.jpg',
     '3542007623.jpg',
     '2412339095.jpg',
     '185826608.jpg',
     '2339078999.jpg',
     '3809761215.jpg',
     '5546511.jpg',
     '2198789414.jpg',
     '906115763.jpg',
     '2588186121.jpg',
     '2508483326.jpg',
     '495499602.jpg',
     '127665486.jpg',
     '4005501772.jpg',
     '3403536238.jpg',
     '3696011777.jpg',
     '2512241884.jpg',
     '301297241.jpg',
     '1947542350.jpg',
     '2252678193.jpg',
     '2571058084.jpg',
     '2993271745.jpg',
     '4012105605.jpg',
     '343259772.jpg',
     '2302834762.jpg',
     '4222447484.jpg',
     '3441642787.jpg',
     '223648042.jpg',
     '999616605.jpg',
     '3051256988.jpg',
     '4176726961.jpg',
     '3688735381.jpg',
     '2096389137.jpg',
     '3187755920.jpg',
     '3837192356.jpg',
     '2207334085.jpg',
     '390092040.jpg',
     '1277358374.jpg',
     '3856973359.jpg',
     '342796483.jpg',
     '3923893955.jpg',
     '3557979582.jpg',
     '3026594115.jpg',
     '1702300197.jpg',
     '976955239.jpg',
     '2007737070.jpg',
     '52729180.jpg',
     '545818494.jpg',
     '4060707407.jpg',
     '3324852988.jpg',
     '3369127189.jpg',
     '2795132787.jpg',
     '1805580430.jpg',
     '3346966351.jpg',
     '4092403597.jpg',
     '3522526655.jpg',
     '3418474102.jpg',
     '3470341002.jpg',
     '1613918546.jpg',
     '1656709823.jpg',
     '827007782.jpg',
     '3572465399.jpg',
     '3812991688.jpg',
     '1696170421.jpg',
     '3383147575.jpg',
     '238671270.jpg',
     '266070647.jpg',
     '1040079282.jpg',
     '560592460.jpg',
     '535503922.jpg',
     '1869166281.jpg',
     '986965803.jpg',
     '1991345445.jpg',
     '814547184.jpg',
     '3184771451.jpg',
     '3706939010.jpg',
     '3275961494.jpg',
     '4271963761.jpg',
     '1733437484.jpg',
     '4156350478.jpg',
     '1367265515.jpg',
     '336804409.jpg',
     '1347907652.jpg',
     '219119835.jpg',
     '3267768763.jpg',
     '2169173551.jpg',
     '397896685.jpg',
     '2130379306.jpg',
     '3408566327.jpg',
     '1315872826.jpg',
     '3319077836.jpg',
     '2323936.jpg',
     '2432364538.jpg',
     '2218894352.jpg',
     '969719974.jpg',
     '600869262.jpg',
     '750645140.jpg',
     '1901823348.jpg',
     '424257956.jpg',
     '2577910904.jpg',
     '3817704326.jpg',
     '547059239.jpg',
     '786431540.jpg',
     '4126983355.jpg',
     '2723167628.jpg',
     '4165205071.jpg',
     '2026174154.jpg',
     '878936941.jpg',
     '876588489.jpg',
     '1683556923.jpg',
     '874716908.jpg',
     '1788333451.jpg',
     '2801727274.jpg',
     '4181351157.jpg',
     '1437908880.jpg',
     '636559480.jpg',
     '2937152803.jpg',
     '326211712.jpg',
     '3460751312.jpg',
     '2972239724.jpg',
     '450239051.jpg',
     '3388143466.jpg',
     '1587009529.jpg',
     '2935393677.jpg',
     '2982385664.jpg',
     '154831470.jpg',
     '3204228673.jpg',
     '3568561283.jpg',
     '718198849.jpg',
     '2334816415.jpg',
     '2601029045.jpg',
     '1229830686.jpg',
     '2950166442.jpg',
     '1882919886.jpg',
     '3557354369.jpg',
     '3239851865.jpg',
     '3826378692.jpg',
     '2439113989.jpg',
     '146632116.jpg',
     '3688869074.jpg',
     '3547680343.jpg',
     '1541855978.jpg',
     '1129878051.jpg',
     '3987029837.jpg',
     '4176553783.jpg',
     '4252015282.jpg',
     '980682914.jpg',
     '3441281524.jpg',
     '2121641395.jpg',
     '2191379565.jpg',
     '2241577548.jpg',
     '82290706.jpg',
     '2765983833.jpg',
     '2787710566.jpg',
     '271289942.jpg',
     '2532608819.jpg',
     '1812948134.jpg',
     '855456135.jpg',
     '3162392075.jpg',
     '2090702902.jpg',
     '199112616.jpg',
     '4285755413.jpg',
     '1411799307.jpg',
     '2733802395.jpg',
     '2393701311.jpg',
     '1197262681.jpg',
     '1252720161.jpg',
     '827746278.jpg',
     '3885271451.jpg',
     '1339337700.jpg',
     '3383325389.jpg',
     '1368184561.jpg',
     '3794499013.jpg',
     '3617067141.jpg',
     '611831978.jpg',
     '3275064872.jpg',
     '2884824828.jpg',
     '1864285719.jpg',
     '3429167683.jpg',
     '3468200884.jpg',
     '2359855528.jpg',
     '416350020.jpg',
     '1969328625.jpg',
     '308791950.jpg',
     '2665787416.jpg',
     '2524786194.jpg',
     '832440144.jpg',
     '685840333.jpg',
     '2289892824.jpg',
     '1807740232.jpg',
     '1815863009.jpg',
     '106169527.jpg',
     '651255888.jpg',
     '4007234267.jpg',
     '3380981345.jpg',
     '983014273.jpg',
     '118845002.jpg',
     '2004186800.jpg',
     '3009725992.jpg',
     '3881936041.jpg',
     '4194767619.jpg',
     '3402428593.jpg',
     '1109535439.jpg',
     '95322868.jpg',
     '2757300199.jpg',
     '2659874728.jpg',
     '839709957.jpg',
     '4122834254.jpg',
     '91429187.jpg',
     '1347999958.jpg',
     '1321771793.jpg',
     '1161422335.jpg',
     '2856556640.jpg',
     '1989388407.jpg',
     '2834820650.jpg',
     '2781831798.jpg',
     '307148103.jpg',
     '2613849666.jpg',
     '2016669057.jpg',
     '3118120871.jpg',
     '1112568678.jpg',
     '2819659045.jpg',
     '2492339911.jpg',
     '936775758.jpg',
     '1354318448.jpg',
     '1344686527.jpg',
     '3782663909.jpg',
     '4134583704.jpg',
     '807063038.jpg',
     '719512192.jpg',
     '3465620541.jpg',
     '4031374252.jpg',
     '1796303978.jpg',
     '3758873726.jpg',
     '1620428397.jpg',
     '918168306.jpg',
     '3024619952.jpg',
     '2690070450.jpg',
     '2410891976.jpg',
     '1377639054.jpg',
     '1435182915.jpg',
     '1352385379.jpg',
     '3716704922.jpg',
     '2155641559.jpg',
     '2064463834.jpg',
     '274726002.jpg',
     '2853279464.jpg',
     '2205985978.jpg',
     '117953500.jpg',
     '2041011559.jpg',
     '395181448.jpg',
     '1939567096.jpg',
     '1179062519.jpg',
     '3335092693.jpg',
     '1091837239.jpg',
     '3088621387.jpg',
     '1608599582.jpg',
     '4114142974.jpg',
     '1222730682.jpg',
     '4204052226.jpg',
     '2027299307.jpg',
     '1798415301.jpg',
     '4067939346.jpg',
     '864933566.jpg',
     '180910875.jpg',
     '1541808301.jpg',
     '2105492888.jpg',
     '2556598810.jpg',
     '2384352551.jpg',
     '3992333811.jpg',
     '1690276814.jpg',
     '2325685392.jpg',
     '1303557485.jpg',
     '2935901261.jpg',
     '3971676147.jpg',
     '1313972896.jpg',
     '325202356.jpg',
     '3607083147.jpg',
     '660715584.jpg',
     '1731589806.jpg',
     '618654809.jpg',
     '3225508752.jpg',
     '1491905259.jpg',
     '3726258018.jpg',
     '1036302176.jpg',
     '117226420.jpg',
     '2470191455.jpg',
     '525742373.jpg',
     '2481691686.jpg',
     '1653502098.jpg',
     '726377415.jpg',
     '984417293.jpg',
     '2846104272.jpg',
     '3495378844.jpg',
     '2130791173.jpg',
     '700772842.jpg',
     '4214804223.jpg',
     '2187716233.jpg',
     '1100123652.jpg',
     '780810143.jpg',
     '3533832269.jpg',
     '948846771.jpg',
     '3977938536.jpg',
     '1008244905.jpg',
     '1044930186.jpg',
     '401824702.jpg',
     '2517495253.jpg',
     '790756634.jpg',
     '2442913111.jpg',
     '2681562878.jpg',
     '749454717.jpg',
     '1352692605.jpg',
     '2026227726.jpg',
     '269949452.jpg',
     '4175864678.jpg',
     '1959596149.jpg',
     '4254971723.jpg',
     '3556554979.jpg',
     '4057315544.jpg',
     '1142627620.jpg',
     '1654445556.jpg',
     '1505418979.jpg',
     '2402393622.jpg',
     '1559511598.jpg',
     '2690891696.jpg',
     '4239262882.jpg',
     '462053910.jpg',
     '804347789.jpg',
     '138133533.jpg',
     '1931114133.jpg',
     '2699398200.jpg',
     '891029839.jpg',
     '2033707322.jpg',
     '4276075687.jpg',
     '4013834415.jpg',
     '1629563382.jpg',
     '3758359822.jpg',
     '1983593299.jpg',
     '2435051665.jpg',
     '2269255258.jpg',
     '2658779358.jpg',
     '1281314946.jpg',
     '4136626919.jpg',
     '2190685942.jpg',
     '2526462874.jpg',
     '2256094590.jpg',
     '1507118084.jpg',
     '3095137637.jpg',
     '254949924.jpg',
     '2200762237.jpg',
     '3995652879.jpg',
     '3689984405.jpg',
     '241455389.jpg',
     '886793145.jpg',
     '4170279280.jpg',
     '2278582450.jpg',
     '3071845704.jpg',
     '4124651871.jpg',
     '2560941741.jpg',
     '2520536047.jpg',
     '4034495674.jpg',
     '27622239.jpg',
     '1028105805.jpg',
     '3177907789.jpg',
     '1319252065.jpg',
     '2845701741.jpg',
     '2884532990.jpg',
     '275310206.jpg',
     '3384875851.jpg',
     '1456262062.jpg',
     '735529446.jpg',
     '2045155911.jpg',
     '138347790.jpg',
     '90625412.jpg',
     '2934683232.jpg',
     '471733925.jpg',
     '232710623.jpg',
     '3805606853.jpg',
     '224203532.jpg',
     '1115389496.jpg',
     '3232568726.jpg',
     '3426576546.jpg',
     '2745720901.jpg',
     '2203128362.jpg',
     '2592820175.jpg',
     '88427493.jpg',
     '3173423403.jpg',
     '3452847789.jpg',
     '3644704769.jpg',
     '4267085223.jpg',
     '3381316827.jpg',
     '1276900213.jpg',
     '1148865724.jpg',
     '2302050669.jpg',
     '3806360308.jpg',
     '3443521560.jpg',
     '1344694162.jpg',
     '1492444202.jpg',
     '4169606060.jpg',
     '2946211830.jpg',
     '3325640914.jpg',
     '113803368.jpg',
     '1694999394.jpg',
     '3421208425.jpg',
     '2494764703.jpg',
     '4241010942.jpg',
     '2864532885.jpg',
     '3516709559.jpg',
     '826231979.jpg',
     '1118049363.jpg',
     '2241795551.jpg',
     '4102729978.jpg',
     '3997854029.jpg',
     '3388533190.jpg',
     '1052881053.jpg',
     '1539561036.jpg',
     '2738985524.jpg',
     '3856210949.jpg',
     '623631977.jpg',
     '2469652661.jpg',
     '3254381404.jpg',
     '1901119946.jpg',
     '846824837.jpg',
     '1427622404.jpg',
     '3580451894.jpg',
     '451922996.jpg',
     '715057901.jpg',
     '1128067244.jpg',
     '2238900239.jpg',
     '1700559021.jpg',
     '2021239499.jpg',
     '2148834762.jpg',
     '2853436606.jpg',
     '2092629811.jpg',
     '2431603043.jpg',
     '3114592390.jpg',
     '4256663928.jpg',
     '3506949016.jpg',
     '183998159.jpg',
     '4037044151.jpg',
     '1318025509.jpg',
     '1884499886.jpg',
     '2323982153.jpg',
     '1598396225.jpg',
     '2826018346.jpg',
     '555296352.jpg',
     '301572737.jpg',
     '3751724682.jpg',
     '2039512472.jpg',
     '1831075447.jpg',
     '2491061027.jpg',
     '792936600.jpg',
     '512575634.jpg',
     '2459666631.jpg',
     '4016407107.jpg',
     '3677212401.jpg',
     '2261145380.jpg',
     '2000880399.jpg',
     '3027608054.jpg',
     '2302182518.jpg',
     '4258928980.jpg',
     '3210844755.jpg',
     '332612328.jpg',
     '2205809644.jpg',
     '2283773342.jpg',
     '1799648561.jpg',
     '1968596086.jpg',
     '1383484107.jpg',
     '2457355421.jpg',
     '1595096662.jpg',
     '3793664928.jpg',
     '1918762654.jpg',
     '1415860816.jpg',
     '293090840.jpg',
     '3019739118.jpg',
     '2789524005.jpg',
     '2002906625.jpg',
     '720304671.jpg',
     '1383579851.jpg',
     '2182500020.jpg',
     '2803200919.jpg',
     '2232257022.jpg',
     '1457828872.jpg',
     '176734204.jpg',
     '1558649503.jpg',
     '224402404.jpg',
     '3765800503.jpg',
     '1667962035.jpg',
     '2087367640.jpg',
     '2234346927.jpg',
     '1276952217.jpg',
     '1861251806.jpg',
     '23975111.jpg',
     '184437770.jpg',
     '1337934930.jpg',
     '1269866934.jpg',
     '3019705895.jpg',
     '581235562.jpg',
     '1148412619.jpg',
     '747770020.jpg',
     '3171101040.jpg',
     '1235142158.jpg',
     '3378948686.jpg',
     '1733574271.jpg',
     '2360164721.jpg',
     '1599378224.jpg',
     '2801308538.jpg',
     '938742634.jpg',
     '119573280.jpg',
     '1202619622.jpg',
     '1557739195.jpg',
     '445689900.jpg',
     '3421134001.jpg',
     '2178418518.jpg',
     '2801552324.jpg',
     '36120366.jpg',
     '496631673.jpg',
     '2690224472.jpg',
     '1766960814.jpg',
     '4149113936.jpg',
     '1994597996.jpg',
     '4188579631.jpg',
     '3316969906.jpg',
     '1888672105.jpg',
     '2738117175.jpg',
     '1385808202.jpg',
     '1445721278.jpg',
     '833188663.jpg',
     '4250538490.jpg',
     '2814644879.jpg',
     '2489350383.jpg',
     '2027821196.jpg',
     '2522103312.jpg',
     '151894628.jpg',
     '2817619344.jpg',
     '3442029144.jpg',
     '4037316351.jpg',
     '743105570.jpg',
     '3846071178.jpg',
     '2161004909.jpg',
     '326839479.jpg',
     '1117199954.jpg',
     '1054179399.jpg',
     '328474315.jpg',
     '3354494955.jpg',
     '1122080430.jpg',
     '400794559.jpg',
     '1108862699.jpg',
     '1186498322.jpg',
     '1045776023.jpg',
     '2535406918.jpg',
     '1156464284.jpg',
     '1241084858.jpg',
     '4262559239.jpg',
     '2728936550.jpg',
     '3101645005.jpg',
     '2495314990.jpg',
     '4137735351.jpg',
     '4119076771.jpg',
     '1079224858.jpg',
     '3080866826.jpg',
     '2380764597.jpg',
     '3670188705.jpg',
     '3177385615.jpg',
     '2123430239.jpg',
     '3043785326.jpg',
     '3558352705.jpg',
     '496924131.jpg',
     '1003218714.jpg',
     '228290866.jpg',
     '3250253495.jpg',
     '3645381564.jpg',
     '640668148.jpg',
     '944726140.jpg',
     '2719462467.jpg',
     '2918749013.jpg',
     '1488135132.jpg',
     '3427302906.jpg',
     '2296087463.jpg',
     '3738990958.jpg',
     '1945042930.jpg',
     '1730671477.jpg',
     '1649487665.jpg',
     '2052193319.jpg',
     '2114705493.jpg',
     '823568510.jpg',
     '379034771.jpg',
     '3773257510.jpg',
     '2491752039.jpg',
     '2329257679.jpg',
     '2089741702.jpg',
     '2118210458.jpg',
     '967603965.jpg',
     '397477697.jpg',
     '3345615928.jpg',
     '3881028757.jpg',
     '3164095616.jpg',
     '457102375.jpg',
     '2731309139.jpg',
     '1230025143.jpg',
     '1220026240.jpg',
     '3451871229.jpg',
     '2220309469.jpg',
     '3299026258.jpg',
     '1858241102.jpg',
     '3638291703.jpg',
     '2115818711.jpg',
     '4109440762.jpg',
     '1940410728.jpg',
     '2223402762.jpg',
     '611122414.jpg',
     '888139689.jpg',
     '1014433832.jpg',
     '3708657911.jpg',
     '1593185834.jpg',
     '1516982275.jpg',
     '3304481536.jpg',
     '719168391.jpg',
     '4223800974.jpg',
     '546003679.jpg',
     '2073687427.jpg',
     '2502748604.jpg',
     '1274870847.jpg',
     '3395212339.jpg',
     '3092828212.jpg',
     '117837871.jpg',
     '22341045.jpg',
     '1463895571.jpg',
     '2465281174.jpg',
     '1478310133.jpg',
     '4001662386.jpg',
     '3198959559.jpg',
     '4262860938.jpg',
     '550742055.jpg',
     '2995689898.jpg',
     '2176535101.jpg',
     '2055261864.jpg',
     '2731822095.jpg',
     '2911150732.jpg',
     '1256756297.jpg',
     '3225863041.jpg',
     '711417337.jpg',
     '4142973523.jpg',
     '3677912635.jpg',
     '1161718761.jpg',
     '484286450.jpg',
     '1270512550.jpg',
     '3036253841.jpg',
     '2748770173.jpg',
     '2867619022.jpg',
     '2341588461.jpg',
     '4029728754.jpg',
     '525960141.jpg',
     '4045015795.jpg',
     '4162062319.jpg',
     '1482562334.jpg',
     '1545341197.jpg',
     '3028978294.jpg',
     '39272616.jpg',
     '3917910837.jpg',
     '3585442386.jpg',
     '3739148870.jpg',
     '2513639394.jpg',
     '1723001945.jpg',
     '2078503653.jpg',
     '2824560733.jpg',
     '3963914024.jpg',
     '2373132884.jpg',
     '945491819.jpg',
     '472300747.jpg',
     '3720190596.jpg',
     '695438825.jpg',
     '1742921296.jpg',
     '3397166306.jpg',
     '1998622752.jpg',
     '595285213.jpg',
     '806702626.jpg',
     '2297582762.jpg',
     '1578495019.jpg',
     '3240028167.jpg',
     '404115232.jpg',
     '404837485.jpg',
     '3221415774.jpg',
     '335787453.jpg',
     '873205488.jpg',
     '1573629437.jpg',
     '2383750742.jpg',
     '1857766258.jpg',
     '3248325951.jpg',
     '1807427029.jpg',
     '2002289812.jpg',
     '1673707752.jpg',
     '604445173.jpg',
     '1166269139.jpg',
     '4169558391.jpg',
     '2092002351.jpg',
     '858590964.jpg',
     '3044159810.jpg',
     '2666077956.jpg',
     '1979482917.jpg',
     '1529906734.jpg',
     '2391557874.jpg',
     '3590920692.jpg',
     '1203777682.jpg',
     '3822484793.jpg',
     '1333145781.jpg',
     '3813835902.jpg',
     '3875177069.jpg',
     '1831712346.jpg',
     '4098892948.jpg',
     '662844751.jpg',
     '3205007771.jpg',
     '1418105858.jpg',
     '3347678996.jpg',
     '2948928390.jpg',
     '2154958086.jpg',
     '3464808111.jpg',
     '1895234488.jpg',
     '328383026.jpg',
     '2718170987.jpg',
     '2499466480.jpg',
     '248967809.jpg',
     '2672327544.jpg',
     '3343671398.jpg',
     '3138454359.jpg',
     '2092518082.jpg',
     '1077728760.jpg',
     '3854923530.jpg',
     '2196148634.jpg',
     '2822454255.jpg',
     '3016051736.jpg',
     '598088658.jpg',
     '1038640139.jpg',
     '1029778366.jpg',
     '3442647074.jpg',
     '3485849218.jpg',
     '3517991317.jpg',
     '3275347756.jpg',
     '3794607673.jpg',
     '3382800297.jpg',
     '1776479289.jpg',
     '44779769.jpg',
     '1677224730.jpg',
     '1287334854.jpg',
     '616745238.jpg',
     '3903787097.jpg',
     '2644006691.jpg',
     '437629432.jpg',
     '2432250705.jpg',
     '1600111988.jpg',
     '3630665946.jpg',
     '1812865076.jpg',
     '2784153881.jpg',
     '2265748268.jpg',
     '794665522.jpg',
     '1581140405.jpg',
     '4083726805.jpg',
     '565965080.jpg',
     '166356403.jpg',
     '232991186.jpg',
     '3766549502.jpg',
     '1402359353.jpg',
     '2051128435.jpg',
     '1098441542.jpg',
     '2984202132.jpg',
     '1368550859.jpg',
     '901063337.jpg',
     '3854645209.jpg',
     '2547206016.jpg',
     ...]



Since all the training images are present in a single folder, we need to classify them into sub folders such that each folder contains images of its class. This kind of classification will make it possible to use the ImageFolder class of PyTorch.

Let us save the training images into seperate subfolders based on their disease classes. Also, we set aside 10% of images randomly chosen from each class as test images.

## Creating custom PyTorch datatset


```python
os.getcwd()
```




    '/kaggle/working'




```python
base_dir = './data'

train_dir = base_dir + '/train'
os.mkdir(train_dir)
test_dir = base_dir + '/test'
os.mkdir(test_dir)
```


```python
import shutil

c = 0
for i in range(len(images_labels.label.unique())):
    new_dir = train_dir + '/' + label_map.iloc[i].item()
    os.mkdir(new_dir)
    for filename in images_labels[images_labels.label == i]['image_id']:
        for file in os.listdir('./data/train_images'):
            if file == filename:
                shutil.move('./data/train_images/' + file, new_dir + '/' + file)
                c += 1
                if c % 500 == 0:
                    print(f"Moved {c} images.")
                break
#print(f"Moved all {c} images.")
```

    Moved 500 images.
    Moved 1000 images.
    Moved 1500 images.
    Moved 2000 images.
    Moved 2500 images.
    Moved 3000 images.
    Moved 3500 images.
    Moved 4000 images.
    Moved 4500 images.
    Moved 5000 images.
    Moved 5500 images.
    Moved 6000 images.
    Moved 6500 images.
    Moved 7000 images.
    Moved 7500 images.
    Moved 8000 images.
    Moved 8500 images.
    Moved 9000 images.
    Moved 9500 images.
    Moved 10000 images.
    Moved 10500 images.
    Moved 11000 images.
    Moved 11500 images.
    Moved 12000 images.
    Moved 12500 images.
    Moved 13000 images.
    Moved 13500 images.
    Moved 14000 images.
    Moved 14500 images.
    Moved 15000 images.
    Moved 15500 images.
    Moved 16000 images.
    Moved 16500 images.
    Moved 17000 images.
    Moved 17500 images.
    Moved 18000 images.
    Moved 18500 images.
    Moved 19000 images.
    Moved 19500 images.
    Moved 20000 images.
    Moved 20500 images.
    Moved 21000 images.
    

Let us check if the count of images in the subfolders matches with the count of images belonging to that category. This way we can verify if we have moved all the images into the correct subfolders. 


```python
images_labels.groupby('label').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1087</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2189</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2386</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13158</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2577</td>
    </tr>
  </tbody>
</table>
</div>




```python
class_folders=os.listdir(train_dir)
class_folders
```




    ['Cassava Mosaic Disease (CMD)',
     'Healthy',
     'Cassava Green Mottle (CGM)',
     'Cassava Bacterial Blight (CBB)',
     'Cassava Brown Streak Disease (CBSD)']




```python
index=0
for index in range(len(class_folders)):
  print(len(os.listdir(train_dir+'/'+class_folders[index])))
```

    13158
    2577
    2386
    1087
    2189
    

Now, let us craete a test dataset using 10% of random images from each sub-class of the train dataset.


```python
import random
random_seed=42

for folder in class_folders:
  new_dir = test_dir + '/' + folder
  os.mkdir(new_dir)
  files = os.listdir(train_dir + '/' + folder)
  to_move = random.sample(files, int(len(files)*0.1))
  for filename in to_move:
    for file in os.listdir(train_dir + '/' + folder):
      if file == filename:
        shutil.move(train_dir + '/' + folder + '/' + file, new_dir + '/' + file)
        break
```


```python
ls -l
```

    total 165892

    ---------- 1 root root      263 Feb 14 22:00 __notebook_source__.ipynb

    -rw-r--r-- 1 root root 71000449 Feb 15 01:14 cassava-enetb4.pth

    -rw-r--r-- 1 root root  6298853 Feb 15 04:42 cassava-feedfwd.pth

    drwxr-xr-x 2 root root     4096 Feb 15 04:59 [0m[01;34mcassava-leaf-disease-classification[0m/

    drwxr-xr-x 4 root root     4096 Feb 15 02:48 [01;34mcassava-leaf-disease-image-folders-600x800[0m/

    -rw-r--r-- 1 root root 92349231 Feb 15 04:02 cassava-resnext50.pth

    -rw-r--r-- 1 root root   199534 Feb 15 04:43 cassava_project.ipynb

    drwxr-xr-x 8 root root     4096 Feb 15 05:03 [01;34mdata[0m/




```python
dataset= ImageFolder('./data',transform=ToTensor())
```

## View some elements of the dataset

Let us picturise a few training images.


```python
def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))
```


```python
img, label = dataset[0]
show_example(img, label)
```

    Label:  test (0)
    


    
![png](cassava-project_files/cassava-project_28_1.png)
    



```python
batch_size=15
data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
```


```python
for images, _ in data_loader:
    print('images.shape:', images.shape)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([]); ax.set_yticks([])
    #denorm_images = denormalize(images, *stats)
    ax.imshow(make_grid(images, nrow=5).permute(1, 2, 0).clamp(0,1))
    break
```

    images.shape: torch.Size([15, 3, 600, 800])
    


    
![png](cassava-project_files/cassava-project_30_1.png)
    


## Prepare dataset for training

In-order to avoid overfitting, we apply the following transformations while loading images from training dataset:
* **randomised data augmentaton:** applying randomly chosen transformations such as cropping, horizondal flipping, changing brightness/contrast/saturation of images.
* **data normalization:** to prevent the values from any one channel from disproportionately affecting the losses and gradients while training by having a higher or wider range of values than others.
* **Early stopping of model's training**, when validation loss starts to increase.


```python
'''def GetStat(data):
  """Function to calculate the mean and std of the input tensors.
  Args:
  Input: ImageFolder containing tensors
  Return: tuple of mean and std 
  """
  loader = DataLoader(data,
                         batch_size=batch_size,
                         num_workers=0,
                         shuffle=False)
  mean = 0.
  std = 0.
  for images, _ in loader:
      batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
      images = images.view(batch_samples, images.size(1), -1)
      mean += images.mean(2).sum(0)
      std += images.std(2).sum(0)

  mean /= len(loader.dataset)
  std /= len(loader.dataset)
  mean= mean.numpy()
  std=std.numpy()
  return (mean, std)
stats=GetStat(train_ds)
print(stats)'''
```


```python
# Data transforms (normalization & data augmentation)

stats=((0.43043306, 0.4969931 , 0.3137205 ), (0.21940342, 0.22414596, 0.20117915))

train_tfms = tt.Compose([tt.RandomCrop((128,128), padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         #tt.RandomRotation,
                         #tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1.0, 1.0)), 
                         tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)])
valid_tfms = tt.Compose([tt.CenterCrop(128),tt.ToTensor(), tt.Normalize(*stats)])
```


```python
# PyTorch datasets
train_ds = ImageFolder('./cassava-leaf-disease-image-folders-600x800/train', train_tfms)
valid_ds = ImageFolder('./cassava-leaf-disease-image-folders-600x800/test', valid_tfms)
```

Define data loaders for training and validation, to load the data in batches.


```python
batch_size=10
```


```python
# PyTorch data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)
```

### Base Model Class and Training on GPU

### Base model class

Let's create a base model class, which contains everything except the model architecture i.e. it wil not contain the __init__ and __forward__ methods. We will later extend this class to try out different architectures.


```python
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
```

### Using GPU

To seamlessly use a GPU, if one is available, we define a couple of helper functions (get_default_device & to_device) and a helper class DeviceDataLoader to move our model & data to the GPU as required. 


```python
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
```


```python
device = get_default_device()
device
```




    device(type='cuda')



Let's move our data loaders to the appropriate device.


```python
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
```

### Helper functions for plotting loss and accuracy

Let us also define a couple of helper functions for plotting the losses & accuracies.


```python
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
```


```python
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
```


```python
def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');
```

### Training loop

We define a fit_one_cycle function for training the model. We do the following processes in the fit_one_cycle function to improve its performance.

* **Learning rate scheduling:** Instead of using a fixed learning rate, we will use a learning rate scheduler, which will change the learning rate after every batch of training. There are many strategies for varying the learning rate during training, and the one we'll use is called the "One Cycle Learning Rate Policy", which involves starting with a low learning rate, gradually increasing it batch-by-batch to a high learning rate for about 30% of epochs, then gradually decreasing it to a very low value for the remaining epochs. [[Source]](https://sgugger.github.io/the-1cycle-policy.html)
* **Weight decay:** We also use weight decay, which is yet another regularization technique which prevents the weights from becoming too large by adding an additional term to the loss function.[[Source]](https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab)
* **Gradient clipping:** Apart from the layer weights and outputs, it also helpful to limit the values of gradients to a small range to prevent undesirable changes in parameters due to large gradient values. This simple yet effective technique is called gradient clipping. [[Source]](https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48)

Let's define a fit_one_cycle function now. We'll also record the learning rate used for each batch.


```python
from tqdm.notebook import tqdm

@torch.no_grad() 
def evaluate(model, val_loader):
    model.eval() 
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache() 
    history = [] 
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
                                                  #steps_per_epoch= batches/ epoch
    for epoch in range(epochs):
        # Training Phase 
        model.train() #batchnorm layers can train their parameters beta and gamma, dropout can drop 20% of values
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: #if any gradients > set value, hey get clipped
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step() # perform gradient descent, add weight decay to loss, derivative for weight decay, perform gradient decsendt
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer)) # record lr for each batch
            sched.step() #calc next lr based on one cycle policy
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history
```

## Model-1: Feed Forward Neural Networks


```python
input_size= 3*128*128
```


```python
class FeedFwdModel(ImageClassificationBase):
    def __init__(self, input_size,output_size):
        super().__init__()
        self.linear1=nn.Linear(input_size,32)
        self.linear2=nn.Linear(32,32)
        self.linear3=nn.Linear(32,output_size)
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        # Apply layers & activation functions
        out=self.linear1(out)
        out=F.relu(out)
        out=self.linear2(out)
        out=F.relu(out)
        out=self.linear3(out)
        out=F.relu(out)
        return out
```

You can now instantiate the model, and move it the appropriate device.


```python
model= to_device(FeedFwdModel(input_size,len(train_ds.classes)), device)
```


```python
history = [evaluate(model, valid_dl)]
history
```




    [{'val_loss': 1.6378871202468872, 'val_acc': 0.0621495321393013}]




```python
epochs = 8
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
```


```python
%%time
history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
```


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [0], last_lr: 0.00396, train_loss: 1.6068, val_loss: 1.6045, val_acc: 0.0561
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [1], last_lr: 0.00936, train_loss: 1.6418, val_loss: 2.7624, val_acc: 0.0654
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [2], last_lr: 0.00972, train_loss: 1.6118, val_loss: 1.6094, val_acc: 0.0505
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [3], last_lr: 0.00812, train_loss: 1.7070, val_loss: 1.6094, val_acc: 0.0505
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [4], last_lr: 0.00556, train_loss: 1.6094, val_loss: 1.6094, val_acc: 0.0505
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [5], last_lr: 0.00283, train_loss: 1.6338, val_loss: 1.6094, val_acc: 0.0505
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [6], last_lr: 0.00077, train_loss: 1.6094, val_loss: 1.6094, val_acc: 0.0505
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [7], last_lr: 0.00000, train_loss: 1.6094, val_loss: 1.6094, val_acc: 0.0505
    CPU times: user 1min 39s, sys: 19.8 s, total: 1min 59s
    Wall time: 29min 1s
    


```python
train_time='29:01'
```


```python
plot_accuracies(history)
```


    
![png](cassava-project_files/cassava-project_66_0.png)
    



```python
plot_losses(history)
```


    
![png](cassava-project_files/cassava-project_67_0.png)
    



```python
plot_lrs(history)
```


    
![png](cassava-project_files/cassava-project_68_0.png)
    


Let us record the hyperparameters and final metrics achieved by the model for reference, analysis and comparison. We can record them using jovian.log_hyperparams.



```python
jovian.reset()
jovian.log_hyperparams(arch='feed forward network', 
                       epochs=epochs, 
                       lr=max_lr, 
                       scheduler='one-cycle', 
                       weight_decay=weight_decay, 
                       grad_clip=grad_clip,
                       opt=opt_func.__name__)
```

    [jovian] Hyperparams logged.[0m
    


```python
jovian.log_metrics(val_loss=history[-1]['val_loss'], 
                   val_acc=history[-1]['val_acc'],
                   train_loss=history[-1]['train_loss'],
                   time=train_time)
```

    [jovian] Metrics logged.[0m
    


```python
torch.save(model.state_dict(), 'cassava-feedfwd.pth')
```


```python
jovian.commit(project='cassava_project', environment=None, outputs=['cassava-feedfwd.pth'])
```


    <IPython.core.display.Javascript object>


    [jovian] Attempting to save notebook..[0m
    [jovian] Detected Kaggle notebook...[0m
    [jovian] Uploading notebook to https://jovian.ai/aswiniabraham/cassava_project[0m
    


    <IPython.core.display.Javascript object>


## Model-2: Convolutional Neural Networks


```python
class CnnModel(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 64 x 64

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 32 x 32

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 16 x 16

            nn.Flatten(), 
            nn.Linear(256*16*16, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes))
        
    def forward(self, xb):
        return self.network(xb)
```


```python
model= CnnModel(len(train_ds.classes))
to_device(model, device)
```




    CnnModel(
      (network): Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU()
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU()
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU()
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU()
        (14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (15): Flatten(start_dim=1, end_dim=-1)
        (16): Linear(in_features=65536, out_features=4096, bias=True)
        (17): ReLU()
        (18): Linear(in_features=4096, out_features=256, bias=True)
        (19): ReLU()
        (20): Linear(in_features=256, out_features=5, bias=True)
      )
    )




```python
history= []
```


```python
history = [evaluate(model, valid_dl)]
history
```




    [{'val_loss': 1.5846003293991089, 'val_acc': 0.10794392973184586}]




```python
epochs = 8
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
```


```python
%%time
history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
```


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [0], last_lr: 0.00396, train_loss: 1.1939, val_loss: 1.1860, val_acc: 0.6145
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [1], last_lr: 0.00936, train_loss: 34.7533, val_loss: 1.1943, val_acc: 0.6145
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [2], last_lr: 0.00972, train_loss: 16609.8730, val_loss: 1.1914, val_acc: 0.6145
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [3], last_lr: 0.00812, train_loss: 4.8816, val_loss: 1.1838, val_acc: 0.6145
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [4], last_lr: 0.00556, train_loss: 1.1995, val_loss: 1.1841, val_acc: 0.6145
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [5], last_lr: 0.00283, train_loss: 1.1852, val_loss: 1.1838, val_acc: 0.6145
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [6], last_lr: 0.00077, train_loss: 1.1833, val_loss: 1.1839, val_acc: 0.6145
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [7], last_lr: 0.00000, train_loss: 1.1834, val_loss: 1.1837, val_acc: 0.6145
    CPU times: user 3min 30s, sys: 21.7 s, total: 3min 51s
    Wall time: 30min 46s
    


```python
train_time='30:46'
```


```python
plot_accuracies(history)
```


    
![png](cassava-project_files/cassava-project_82_0.png)
    



```python
plot_losses(history)
```


    
![png](cassava-project_files/cassava-project_83_0.png)
    


Let us record the hyperparameters and final metrics achieved by the model.


```python
jovian.reset()
jovian.log_hyperparams(arch='convolutional neural network', 
                       epochs=epochs, 
                       lr=max_lr, 
                       scheduler='one-cycle', 
                       weight_decay=weight_decay, 
                       grad_clip=grad_clip,
                       opt=opt_func.__name__)
```

    [jovian] Hyperparams logged.[0m
    


```python
jovian.log_metrics(val_loss=history[-1]['val_loss'], 
                   val_acc=history[-1]['val_acc'],
                   train_loss=history[-1]['train_loss'],
                   time=train_time)
```

    [jovian] Metrics logged.[0m
    


```python
torch.save(model.state_dict(), 'cnn.pth')
```


```python
jovian.commit(project='cassava_project', environment=None, outputs=['cnn.pth'])
```


    <IPython.core.display.Javascript object>


    [jovian] Attempting to save notebook..[0m
    [jovian] Detected Kaggle notebook...[0m
    [jovian] Uploading notebook to https://jovian.ai/aswiniabraham/cassava_project[0m
    


    <IPython.core.display.Javascript object>


## Model-3: Resnet34 and transfer learning


```python
# Data transforms (normalization & data augmentation)

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tfms = tt.Compose([tt.RandomCrop((128,128), padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         #tt.RandomRotation,
                         #tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1.0, 1.0)), 
                         tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                         tt.ToTensor(), 
                         tt.Normalize(*imagenet_stats,inplace=True)])
valid_tfms = tt.Compose([tt.CenterCrop(128),tt.ToTensor(), tt.Normalize(*imagenet_stats)])
```


```python
# PyTorch datasets
train_ds = ImageFolder('./cassava-leaf-disease-image-folders-600x800/train', train_tfms)
valid_ds = ImageFolder('./cassava-leaf-disease-image-folders-600x800/test', valid_tfms)
```


```python
batch_size=10
```


```python
# PyTorch data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)
```


```python
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
```


```python
from torchvision import models

class Resnet34Model(ImageClassificationBase):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=pretrained)
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)
```


```python
model = to_device(Resnet34Model(len(train_ds.classes)), device)
```


```python
history= []
```


```python
history = [evaluate(model, valid_dl)]
history
```




    [{'val_loss': 1.6652694940567017, 'val_acc': 0.24976633489131927}]




```python
epochs = 8
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
```


```python
%%time
history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
```


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [0], last_lr: 0.00396, train_loss: 1.2018, val_loss: 1.7423, val_acc: 0.6206
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [1], last_lr: 0.00936, train_loss: 1.1898, val_loss: 1.2172, val_acc: 0.5972
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [2], last_lr: 0.00972, train_loss: 1.1844, val_loss: 1.1469, val_acc: 0.6196
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [3], last_lr: 0.00812, train_loss: 1.1836, val_loss: 1.1837, val_acc: 0.6145
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [4], last_lr: 0.00556, train_loss: 1.1753, val_loss: 1.0930, val_acc: 0.6248
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [5], last_lr: 0.00283, train_loss: 1.1428, val_loss: 1.0670, val_acc: 0.6224
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [6], last_lr: 0.00077, train_loss: 1.1171, val_loss: 1.0275, val_acc: 0.6290
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [7], last_lr: 0.00000, train_loss: 1.0956, val_loss: 0.9982, val_acc: 0.6352
    CPU times: user 16min 30s, sys: 30.2 s, total: 17min
    Wall time: 38min 46s
    


```python
train_time='38:46'
```


```python
plot_accuracies(history)
```


    
![png](cassava-project_files/cassava-project_102_0.png)
    



```python
plot_losses(history)
```


    
![png](cassava-project_files/cassava-project_103_0.png)
    


Let us record the hyperparameters and final metrics achieved by the model.


```python
jovian.reset()
jovian.log_hyperparams(arch='Resnet34 network', 
                       epochs=epochs, 
                       lr=max_lr, 
                       scheduler='one-cycle', 
                       weight_decay=weight_decay, 
                       grad_clip=grad_clip,
                       opt=opt_func.__name__)
```

    [jovian] Hyperparams logged.[0m
    


```python
jovian.log_metrics(val_loss=history[-1]['val_loss'], 
                   val_acc=history[-1]['val_acc'],
                   train_loss=history[-1]['train_loss'],
                   time=train_time)
```

    [jovian] Metrics logged.[0m
    


```python
torch.save(model.state_dict(), 'cassava-resnet34.pth')
```


```python
jovian.commit(project='cassava_project', environment=None, outputs=['cassava-resnet34.pth'])
```


    <IPython.core.display.Javascript object>


    [jovian] Attempting to save notebook..[0m
    [jovian] Detected Kaggle notebook...[0m
    [jovian] Uploading notebook to https://jovian.ai/aswiniabraham/cassava_project[0m
    


    <IPython.core.display.Javascript object>


## Model-4: EfficientNet B4 model


```python
# Data transforms (normalization & data augmentation)

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tfms = tt.Compose([tt.RandomCrop((128,128), padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         #tt.RandomRotation,
                         #tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1.0, 1.0)), 
                         tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                         tt.ToTensor(), 
                         tt.Normalize(*imagenet_stats,inplace=True)])
valid_tfms = tt.Compose([tt.CenterCrop(128),tt.ToTensor(), tt.Normalize(*imagenet_stats)])
```


```python
# PyTorch datasets
train_ds = ImageFolder('./cassava-leaf-disease-image-folders-600x800/train', train_tfms)
valid_ds = ImageFolder('./cassava-leaf-disease-image-folders-600x800/test', valid_tfms)
```


```python
batch_size=10
```


```python
# PyTorch data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)
```


```python
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
```


```python
! pip install efficientnet-pytorch
```


```python
from efficientnet_pytorch import EfficientNet

class Enetb4Model(ImageClassificationBase):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.network = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)

    def forward(self, xb):
        return self.network(xb)
```


```python
model = to_device(Enetb4Model(len(train_ds.classes)), device)
```


```python
history= []
```


```python
history = [evaluate(model, valid_dl)]
history
```


```python
epochs = 8
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
```


```python
%%time
history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
```


```python
train_time='1:05:57'
```


```python
plot_accuracies(history)
```


```python
plot_losses(history)
```

Let us record the hyperparameters and final metrics achieved by the model.


```python
!pip install jovian --upgrade --quiet
```


```python
import jovian
```


```python
jovian.reset()
jovian.log_hyperparams(arch='Effiecientnet B4', 
                       epochs=epochs, 
                       lr=max_lr, 
                       scheduler='one-cycle', 
                       weight_decay=weight_decay, 
                       grad_clip=grad_clip,
                       opt=opt_func.__name__)
```


```python
jovian.log_metrics(val_loss=history[-1]['val_loss'], 
                   val_acc=history[-1]['val_acc'],
                   train_loss=history[-1]['train_loss'],
                   time=train_time)
```


```python
torch.save(model.state_dict(), 'cassava-enetb4.pth')
```


```python
jovian.commit(project='cassava_project', environment=None, outputs=['cassava-enetb4.pth'])
```


```python
os.getcwd()
```


```python
os.listdir('./')
```


```python
from IPython.display import FileLink
FileLink(r'cassava-enetb4.pth')
```

## Model-5: Resnext50_32x4d


```python
# ================================================
# resnext50_32x4d architecture
# ================================================

from torchvision import models

class ResnextModel(ImageClassificationBase):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnext50_32x4d(pretrained=pretrained)
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)
```


```python
model = ResnextModel(len(train_ds.classes), pretrained= True)
```


```python
to_device(model, device)
```




    ResnextModel(
      (network): ResNet(
        (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (layer1): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (2): Bottleneck(
            (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (layer2): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (2): Bottleneck(
            (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (3): Bottleneck(
            (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (layer3): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (2): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (3): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (4): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (5): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (layer4): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (2): Bottleneck(
            (conv1): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (bn2): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        (fc): Linear(in_features=2048, out_features=5, bias=True)
      )
    )




```python
history= []
```


```python
history = [evaluate(model, valid_dl)]
history
```




    [{'val_loss': 1.709054946899414, 'val_acc': 0.11039718985557556}]




```python
%%time
history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
```


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [0], last_lr: 0.00396, train_loss: 1.1113, val_loss: 1.0380, val_acc: 0.6275
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [1], last_lr: 0.00936, train_loss: 1.1528, val_loss: 1.6740, val_acc: 0.6145
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [2], last_lr: 0.00972, train_loss: 1.1777, val_loss: 1.2194, val_acc: 0.6145
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [3], last_lr: 0.00812, train_loss: 1.2058, val_loss: 1.1862, val_acc: 0.6145
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [4], last_lr: 0.00556, train_loss: 1.2107, val_loss: 1.1467, val_acc: 0.6145
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [5], last_lr: 0.00283, train_loss: 1.1942, val_loss: 1.1059, val_acc: 0.6221
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [6], last_lr: 0.00077, train_loss: 1.1408, val_loss: 1.0327, val_acc: 0.6341
    


      0%|          | 0/1927 [00:00<?, ?it/s]


    Epoch [7], last_lr: 0.00000, train_loss: 1.1161, val_loss: 1.0162, val_acc: 0.6398
    CPU times: user 31min 28s, sys: 33.8 s, total: 32min 2s
    Wall time: 45min 17s
    


```python
train_time='45:17'
```


```python
plot_accuracies(history)
```


    
![png](cassava-project_files/cassava-project_143_0.png)
    



```python
plot_losses(history)
```


    
![png](cassava-project_files/cassava-project_144_0.png)
    


Let us record the hyperparameters and final metrics achieved by the model.


```python
jovian.reset()
jovian.log_hyperparams(arch='resnext50_32x4d', 
                       epochs=epochs, 
                       lr=max_lr, 
                       scheduler='one-cycle', 
                       weight_decay=weight_decay, 
                       grad_clip=grad_clip,
                       opt=opt_func.__name__)
```

    [jovian] Hyperparams logged.[0m
    


```python
jovian.log_metrics(val_loss=history[-1]['val_loss'], 
                   val_acc=history[-1]['val_acc'],
                   train_loss=history[-1]['train_loss'],
                   time=train_time)
```

    [jovian] Metrics logged.[0m
    


```python
torch.save(model.state_dict(), 'cassava-resnext50.pth')
```


```python
from IPython.display import FileLink
FileLink(r'cassava-resnext50.pth')
```




<a href='cassava-resnext50.pth' target='_blank'>cassava-resnext50.pth</a><br>




```python
jovian.commit(project='cassava_project', environment=None, outputs=['cassava-resnext50.pth'])
```


    <IPython.core.display.Javascript object>


    [jovian] Attempting to save notebook..[0m
    [jovian] Detected Kaggle notebook...[0m
    [jovian] Uploading notebook to https://jovian.ai/aswiniabraham/cassava_project[0m
    


    <IPython.core.display.Javascript object>


## Ensemble two models


```python
! pip install efficientnet-pytorch
#! pip install timm
```


```python
# ================================================
# efficientnet-b4 architecture
# ================================================

from efficientnet_pytorch import EfficientNet

#import timm

class EnetModel(ImageClassificationBase):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.network = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        #self.network=timm.create_model('tf_efficientnet_b4_ns', pretrained=pretrained)
        # Replace last layer
        #self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)
    
    def forward(self, xb):
        return self.network(xb)
```


```python
# ================================================
# resnext50_32x4d architecture
# ================================================

from torchvision import models

class ResnextModel(ImageClassificationBase):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnext50_32x4d(pretrained=pretrained)
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)
```


```python
EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
```


```python
models.resnext50_32x4d(pretrained=False)
```


```python
# Ensemble two models

class MyEnsemble(ImageClassificationBase):
    def __init__(self, modelA, modelB, num_classes):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # Remove last linear layer
        self.modelA.fc= nn.Identity()
        self.modelB.fc= nn.Identity()

        # Create new classifier
        self.classifier= nn.Linear(1792+2040, num_classes)
        
    def forward(self, x):
      x1= self.modelA(x.clone())
      x1= x1.view(x1.size(0),-1)
      x2= self.modelB(x)
      x2= x2.view(x2.size(0),-1)
      x= torch.cat((x1,x2),dim=1)

      x= self.classifier(F.relu(x))
      return x

```


```python
MyEnsemble(modelA, modelB, len(train_ds.classes))
```


```python
# Load models
modelA = EnetModel(len(train_ds.classes), pretrained= False)
modelB = ResnextModel(len(train_ds.classes), pretrained= False)

# Load state dicts
modelA.load_state_dict(torch.load('cassava-efficientnet-04_67pct_8epch.pth'))
modelB.load_state_dict(torch.load('cassava-resnext50_32x4d.pth'))

model = MyEnsemble(modelA, modelB, len(train_ds.classes))

# Load to device 
model= to_device(model, device)
```


```python
epochs = 8
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
```


```python
history = [evaluate(model, valid_dl)]
history
```


```python
%%time
history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
```


```python
train_time=':'
```


```python
plot_accuracies(history)
```


```python
plot_losses(history)
```


```python
plot_lrs(history)
```

### Testing with individual images


```python
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return train_ds.classes[preds[0].item()]
```


```python
img, label = valid_ds[0]
plt.imshow(img.permute(1, 2, 0).clamp(0, 1))
print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))
```


```python
img, label = valid_ds[1002]
plt.imshow(img.permute(1, 2, 0))
print('Label:', valid_ds.classes[label], ', Predicted:', predict_image(img, model))
```


```python
img, label = valid_ds[153]
plt.imshow(img.permute(1, 2, 0))
print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))
```

## Summary of training results

The table below gives the validation loss, validation accuracy and time taken for training different models for the following hyperparameters:
epochs = 8
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

|Model| val_accuracy| val_loss| time|
|:----|---|---|---|
|Feed Forward Neural Network|05.05% |1.60944 |29:01 |
|Convolutional Neural Network|61.45% |1.18407 | 46:52|
|Resnet34|64.92%|0.94607|34:30|
|Efficientnet-B4|65.25%|0.89396|1:05:57|
|Resnext50_32x4d|63.98%|1.01616|45:17|

## Save


```python
jovian.commit(project='cassava_project', environment=None)
```


    <IPython.core.display.Javascript object>


    [jovian] Attempting to save notebook..[0m
    [jovian] Detected Kaggle notebook...[0m
    [jovian] Uploading notebook to https://jovian.ai/aswiniabraham/cassava_project[0m
    


    <IPython.core.display.Javascript object>


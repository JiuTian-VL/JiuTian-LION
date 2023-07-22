# JiuTian
<div align="center">

<h1>JiuTian (九天) </h1>

<div>
:fire: Details will be released. Stay tuned :beers: :+1: 
</div>
<br>
  
<img src='assets/model.jpg' width='100%'>

</div>

## Training Data

|<sub><sup>Dataset</sup></sub>|<sub><sup>Size</sup></sub>|<sub><sup>Pretraining</sup></sub>|<sub><sup>Instruction Tuning</sup></sub>|
|:-|:-:|:-:|:-:|
|<sub><sup>[LAION](https://laion.ai/)<sup>(*)</sup></sup></sub>|<sub><sup>100M</sup></sub>| :white_check_mark:
|<sub><sup>[COCOCN](https://github.com/li-xirong/coco-cn)|<sub><sup>27K</sup></sub>|:white_check_mark:|
|<sub><sup>[MIMIC-IT](https://github.com/Luodian/Otter/blob/main/mimic-it/README.md)<sup>(**)</sup></sup></sub>|<sub><sup>1.2M</sup></sub>||:white_check_mark:
|<sub><sup>[LRV](https://fuxiaoliu.github.io/LRV/)|<sub><sup>20K</sup></sub>||:white_check_mark:
|<sub><sup>[LLaVAR](https://llavar.github.io/)|<sub><sup>158K</sup></sub>||:white_check_mark:
|<sub><sup>[TextCap](https://textvqa.org/textcaps/)|<sub><sup>21K</sup></sub>||:white_check_mark:
|<sub><sup>[VQA v2.0](https://visualqa.org/)|<sub><sup>82K</sup></sub>||:white_check_mark:
|<sub><sup>[GQA](https://github.com/stanfordnlp/mac-network)|<sub><sup>148K</sup></sub>||:white_check_mark:
|<sub><sup>[IconQA](https://iconqa.github.io/)|<sub><sup>19K</sup></sub>||:white_check_mark:
|<sub><sup>[OK-VQA](https://okvqa.allenai.org/)|<sub><sup>9K</sup></sub>||:white_check_mark:
|<sub><sup>[A-OKVQA](https://allenai.org/project/a-okvqa/home)|<sub><sup>17K</sup></sub>||:white_check_mark:

Notes:
(*): using our designed rules to filter the original data;
(**): only including its open source part
# YOLO-v1 implemented by pytorch.
*This is just an experimental code to deepen the understanding of YOLO-v1. The overall design is based on YOLO-v1, but some structures have been changed, such as using deep separable convolutions and discarding fully connected layers. And i generated watermark data to verify the validity of the network.*

----
## train

*easy to train*

### prepare training data, voc format
    
    data/
        imgs/
            ...
        annotations/
            ...

### Specify category name list and your hyper-parameter in train.py

```shell script
nohup python train.py &
```

----
## detect

```shell script
python detect.py --input_path XXXXX
```

## result
<div><div align="left">
<img src="result/1.jpg" width="500" height="500">
<img src="result/2.jpg" width="500" height="500">
<img src="result/3.jpg" width="500" height="500">
<img src="result/4.jpg" width="500" height="500"></div>
</div>



## ref

[my blog of yolo-v1](http://note4lin.top/post/yolo/)

[yolo-v1](https://pjreddie.com/darknet/yolov1/)

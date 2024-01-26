# Store tfrecord here
The data is VOC format, reference [here](sample.xml)     
data path format  
VOCdevkit  
>VOCdevkit_train  
>>Annotation  
>>JPEGImages   

>VOCdevkit_test   
>>Annotation   
>>JPEGImages   

python ./data/io/convert_data_to_tfrecord.py --VOC_dir='***/VOCdevkit/VOCdevkit_train/' --save_name='train' --img_format='.tif' --dataset='ship'
python convert_data_to_tfrecord.py --VOC_dir=C:/Users/zhuhu/Desktop/R2DCNN/R2CNN_FPN_Tensorflow-master/data/test/ --save_name=test --img_format=.jpg --dataset=gprmax
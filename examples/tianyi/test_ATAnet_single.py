#encoding:utf-8
import numpy as np
import sys
import os
import cv2
import caffe
import csv
import time
import math
caffe_root = '/home/clxia/caffe-master/'
sys.path.insert(0, caffe_root + 'python')

net_file = '/home/clxia/caffe-master/models/ATAnet/ATAnet_test.prototxt'
caffe_model = '/home/clxia/caffe-master/models/ATAnet/solver_iter_60000_ialt_MWI_mean_label1.caffemodel'
mean_file = '/home/clxia/caffe-master/data/tianyi/imagenet_mean.binaryproto'
print('Params loaded!')
caffe.set_mode_gpu()
net = caffe.Net(net_file,
                caffe_model,
                caffe.TEST)

# net_file = '/home/clxia/caffe-master/models/resnext_32/resnext101-32x4d_test.prototxt'
# caffe_model = '/home/clxia/caffe-master/models/resnext_32/solver_iter_60000.caffemodel'
# mean_file = '/home/clxia/caffe-master/data/tianyi/imagenet_mean.binaryproto'
# print('Params loaded!')
# caffe.set_mode_gpu()
# net = caffe.Net(net_file,
#                 caffe_model,
#                 caffe.TEST)

#MWI
mean_blob = caffe.proto.caffe_pb2.BlobProto()  # 创建protobuf blob
mean_blob.ParseFromString(open(mean_file, 'rb').read())    # 读入mean.binaryproto文件内容, # 解析文件内容到blob
mean_npy = caffe.io.blobproto_to_array(mean_blob) # 将blob中的均值转换成numpy格式，array的shape （mean_number，channel, hight, width）


#IALT
ialt_mean_file = '/home/clxia/caffe-master/data/tianyi/ialt_MWI_imagenet_mean.binaryproto'
ialt_mean_blob = caffe.proto.caffe_pb2.BlobProto()  # 创建protobuf blob
ialt_mean_blob.ParseFromString(open(ialt_mean_file, 'rb').read())    # 读入mean.binaryproto文件内容, # 解析文件内容到blob
ialt_mean_npy=caffe.io.blobproto_to_array(ialt_mean_blob)

#MWI
a = mean_npy[0, :, 0, 0]
print(net.blobs['data'].data.shape)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', a)
transformer.set_raw_scale('data', 255.0)
transformer.set_channel_swap('data', (2, 1, 0))
 
#IALT
# ialt_mean=80.893915
# ialt=np.array([ialt_mean,ialt_mean,ialt_mean])
ialt=ialt_mean_npy[0,:,0,0]
transformer_ialt = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer_ialt.set_transpose('data', (2, 0, 1))
transformer_ialt.set_mean('data', ialt)
transformer_ialt.set_raw_scale('data', 255.0)
transformer_ialt.set_channel_swap('data', (2, 1, 0))



test_image_path='/home/clxia/caffe-master/data/test'
# test_image_path='/home/clxia/Downloads/高度计数据集-预算训练集500/预赛训练集-500'

image_names=os.listdir(test_image_path)
image_names.sort()

ft=open('/home/clxia/caffe-master/data/res.txt','w')
# 写入csv文件，设置newline，否则两行之间会空一行
csvfile=open('/home/clxia/caffe-master/data/pre.csv','w') 
#delimiter默认是逗号
writer=csv.writer(csvfile,delimiter=',')

# names = ['DESERT', 'OCEAN', 'MOUNTAIN', 'FARMLAND', 'LAKE', 'CITY']
# 与原文label相反:1
names= ['CITY','LAKE','FARMLAND','MOUNTAIN','OCEAN','DESERT']
# 与原文label相反:2
# names= ['LAKE','FARMLAND','OCEAN','DESERT','CITY','MOUNTAIN']
# 与原文label相反:3
# names= ['CITY','MOUNTAIN','LAKE','FARMLAND','OCEAN','DESERT']
#与原文label相反　：４乱序
# names=['OCEAN','DESERT','CITY','MOUNTAIN','LAKE','FARMLAND']
for image in image_names:
    start=time.clock()
    img_path='/home/clxia/caffe-master/data/test/'+image
    # img_path='/home/clxia/Downloads/高度计数据集-预算训练集500/预赛训练集-500/'+image

    im = caffe.io.load_image(img_path)
    # cv2.imwrite('/home/clxia/caffe-master/data/error/'+image,im)
    # print (im.shape,len(im.shape))
    # break
    #多光谱图
    # ig=cv2.imread(img_path)

    if 'MWI' in image:
        
        # img=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        # im=cv2.merge([img,img,img])
        # size(256,256)
        net.blobs['data'].data[...] = transformer.preprocess('data', im)
        predict = net.forward()
        prob = net.blobs['prob'].data[0].flatten()
        # print(np.argmax(prob)/2.0,round(np.argmax(prob)/2.0)) 
        image_label=names[np.argmax(prob)]
        ft.write(image+','+image_label+'\n')
        writer.writerow([image,image_label])
        # img=cv2.imread(img_path)
        # cv2.imwrite('/home/clxia/caffe-master/data/'+image_label+'/'+image,img)
    #微波图像
    if 'IALT' in image:
        print 'IALT'
        # img=np.zeros(3,im.shape[0],im.shape[1])
        # img[0,:,:]=im
        # img[1,:,:]=im
        # img[2,:,:]=im
        net.blobs['data'].data[...] = transformer_ialt.preprocess('data', im)
        predict = net.forward()
        prob = net.blobs['prob'].data[0].flatten()
        image_label=names[np.argmax(prob)]
        ft.write(image+','+image_label+'\n')
        writer.writerow([image,image_label])

    elapsed=(time.clock()-start)
    print('time used:',elapsed)

ft.close()
csvfile.close()

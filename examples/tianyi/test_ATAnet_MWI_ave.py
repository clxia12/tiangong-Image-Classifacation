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
caffe.set_mode_gpu()

net_file = '/home/clxia/caffe-master/models/ATAnet/ATAnet_test.prototxt'
caffe_model = '/home/clxia/caffe-master/models/ATAnet/solver_iter_60000_label2.caffemodel'
mean_file = '/home/clxia/caffe-master/data/tianyi/imagenet_mean.binaryproto'
print('Params loaded!')
net = caffe.Net(net_file,
                caffe_model,
                caffe.TEST)
#逆序ｌａｂｅｌ:3
net_file_invert = '/home/clxia/caffe-master/models/ATAnet/ATAnet_test.prototxt'
caffe_model_invert = '/home/clxia/caffe-master/models/ATAnet/solver_iter_60000_label3.caffemodel'
net_invert = caffe.Net(net_file_invert,
                caffe_model_invert,
                caffe.TEST)
#乱序label:4
net_file_disorder = '/home/clxia/caffe-master/models/ATAnet/ATAnet_test.prototxt'
caffe_model_disorder = '/home/clxia/caffe-master/models/ATAnet/solver_iter_60000_label4.caffemodel'
net_disorder = caffe.Net(net_file_disorder,
                caffe_model_disorder,
                caffe.TEST)

mean_blob = caffe.proto.caffe_pb2.BlobProto()  # 创建protobuf blob
mean_blob.ParseFromString(open(mean_file, 'rb').read())    # 读入mean.binaryproto文件内容, # 解析文件内容到blob
mean_npy = caffe.io.blobproto_to_array(mean_blob) # 将blob中的均值转换成numpy格式，array的shape （mean_number，channel, hight, width）

#IALT
ialt_mean_file = '/home/clxia/caffe-master/data/tianyi/ialt_MWI_imagenet_mean.binaryproto'
ialt_mean_blob = caffe.proto.caffe_pb2.BlobProto()  # 创建protobuf blob
ialt_mean_blob.ParseFromString(open(ialt_mean_file, 'rb').read())    # 读入mean.binaryproto文件内容, # 解析文件内容到blob
ialt_mean_npy=caffe.io.blobproto_to_array(ialt_mean_blob)

#MWI
#label2
a = mean_npy[0, :, 0, 0]
print(net.blobs['data'].data.shape)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', a)
transformer.set_raw_scale('data', 255.0)
transformer.set_channel_swap('data', (2, 1, 0))

#逆序ｌａｂｅｌ3
print(net_invert.blobs['data'].data.shape)
transformer_invert = caffe.io.Transformer({'data': net_invert.blobs['data'].data.shape})
transformer_invert.set_transpose('data', (2, 0, 1))
transformer_invert.set_mean('data', a)
transformer_invert.set_raw_scale('data', 255.0)
transformer_invert.set_channel_swap('data', (2, 1, 0))
#乱序label4
print(net_disorder.blobs['data'].data.shape)
transformer_disorder = caffe.io.Transformer({'data': net_disorder.blobs['data'].data.shape})
transformer_disorder.set_transpose('data', (2, 0, 1))
transformer_disorder.set_mean('data', a)
transformer_disorder.set_raw_scale('data', 255.0)
transformer_disorder.set_channel_swap('data', (2, 1, 0))


#IALT
#label2
ialt=ialt_mean_npy[0,:,0,0]
print(net.blobs['data'].data.shape)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', ialt)
transformer.set_raw_scale('data', 255.0)
transformer.set_channel_swap('data', (2, 1, 0))

#逆序ｌａｂｅｌ3
print(net_invert.blobs['data'].data.shape)
transformer_invert = caffe.io.Transformer({'data': net_invert.blobs['data'].data.shape})
transformer_invert.set_transpose('data', (2, 0, 1))
transformer_invert.set_mean('data', ialt)
transformer_invert.set_raw_scale('data', 255.0)
transformer_invert.set_channel_swap('data', (2, 1, 0))
#乱序label4
print(net_disorder.blobs['data'].data.shape)
transformer_disorder = caffe.io.Transformer({'data': net_disorder.blobs['data'].data.shape})
transformer_disorder.set_transpose('data', (2, 0, 1))
transformer_disorder.set_mean('data', ialt)
transformer_disorder.set_raw_scale('data', 255.0)
transformer_disorder.set_channel_swap('data', (2, 1, 0))



# test_image_path='/home/clxia/caffe-master/data/test'
test_image_path='/home/clxia/Downloads/宽波段数据集-预赛训练集2000/UNKNOWN/'
# test_image_path='/home/clxia/Downloads/高度计数据集-预算训练集500/预赛训练集-500'

image_names=os.listdir(test_image_path)
image_names.sort()

ft=open('/home/clxia/caffe-master/data/res.txt','w')
# 写入csv文件，设置newline，否则两行之间会空一行
csvfile=open('/home/clxia/caffe-master/data/pre.csv','w') 
#delimiter默认是逗号
writer=csv.writer(csvfile,delimiter=',')
#opposite label 0
# names = ['DESERT', 'OCEAN', 'MOUNTAIN', 'FARMLAND', 'LAKE', 'CITY']
# 与原文label相反:1
# names_invert= ['CITY','LAKE','FARMLAND','MOUNTAIN','OCEAN','DESERT']
# names_disorder=['CITY','LAKE','FARMLAND','MOUNTAIN','OCEAN','DESERT']
# 与原文label相反:2
names_disorder= ['LAKE','FARMLAND','OCEAN','DESERT','CITY','MOUNTAIN']
# 与原文label相反:3
names_label3= ['CITY','MOUNTAIN','LAKE','FARMLAND','OCEAN','DESERT']
#与原文label相反　：４乱序
names_label4=['OCEAN','DESERT','CITY','MOUNTAIN','LAKE','FARMLAND']

pred=['LAKE','FARMLAND','OCEAN','DESERT','CITY','MOUNTAIN']
for image in image_names:
    ave_prob={'LAKE':0.0,'FARMLAND':0.0,'OCEAN':0.0,'DESERT':0.0,'CITY':0.0,'MOUNTAIN':0.0}
    start=time.clock()
    # img_path='/home/clxia/caffe-master/data/test/'+image
    img_path='/home/clxia/Downloads/宽波段数据集-预赛训练集2000/UNKNOWN/'+image
    # img_path='/home/clxia/Downloads/高度计数据集-预算训练集500/预赛训练集-500/'+image

    im = caffe.io.load_image(img_path)
    #多光谱图
    # ig=cv2.imread(img_path)
    flag=1
    # if flag==1:
    if 'MWI' in image:
        # img=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        # im=cv2.merge([img,img,img])
        # size(256,256)
        net.blobs['data'].data[...] = transformer.preprocess('data', im)
        predict = net.forward()
        prob = net.blobs['prob'].data[0].flatten()
        ######################
        ave_prob[names_disorder[0]]+=prob[0]/3
        ave_prob[names_disorder[1]]+=prob[1]/3
        ave_prob[names_disorder[2]]+=prob[2]/3
        ave_prob[names_disorder[3]]+=prob[3]/3
        ave_prob[names_disorder[4]]+=prob[4]/3
        ave_prob[names_disorder[5]]+=prob[5]/3
        ##############        
        #
        net_invert.blobs['data'].data[...] = transformer_invert.preprocess('data', im)
        predict_invert = net_invert.forward()
        prob_invert = net_invert.blobs['prob'].data[0].flatten()
        ######################
        ave_prob[names_label3[0]]+=prob_invert[0]/3
        ave_prob[names_label3[1]]+=prob_invert[1]/3
        ave_prob[names_label3[2]]+=prob_invert[2]/3
        ave_prob[names_label3[3]]+=prob_invert[3]/3
        ave_prob[names_label3[4]]+=prob_invert[4]/3
        ave_prob[names_label3[5]]+=prob_invert[5]/3
        ##############       

        #
        net_disorder.blobs['data'].data[...] = transformer_disorder.preprocess('data', im)
        predict_disorder = net_disorder.forward()
        prob_disorder = net_disorder.blobs['prob'].data[0].flatten()
        ######################
        ave_prob[names_label4[0]]+=prob_disorder[0]/3
        ave_prob[names_label4[1]]+=prob_disorder[1]/3
        ave_prob[names_label4[2]]+=prob_disorder[2]/3
        ave_prob[names_label4[3]]+=prob_disorder[3]/3
        ave_prob[names_label4[4]]+=prob_disorder[4]/3
        ave_prob[names_label4[5]]+=prob_disorder[5]/3
        ##############
        index_in=[ave_prob[pred[0]],ave_prob[pred[1]],ave_prob[pred[2]],ave_prob[pred[3]],ave_prob[pred[4]],ave_prob[pred[5]]]
        image_label=pred[np.argmax(index_in)]
        if ave_prob[image_label]<0.9:
            image_label='UNKNOWN'
            ave_prob[image_label]=0.00001    
        # label0='('+pred[0]+','+str(round(ave_prob[pred[0]],2))+')'
        # label1='('+pred[1]+','+str(round(ave_prob[pred[1]],2))+')'
        # label2='('+pred[2]+','+str(round(ave_prob[pred[2]],2))+')'
        # label3='('+pred[3]+','+str(round(ave_prob[pred[3]],2))+')'
        # label4='('+pred[4]+','+str(round(ave_prob[pred[4]],2))+')'
        # label5='('+pred[5]+','+str(round(ave_prob[pred[5]],2))+')'
        label0='('+pred[0]+','+str(ave_prob[pred[0]])+')'
        label1='('+pred[1]+','+str(ave_prob[pred[1]])+')'
        label2='('+pred[2]+','+str(ave_prob[pred[2]])+')'
        label3='('+pred[3]+','+str(ave_prob[pred[3]])+')'
        label4='('+pred[4]+','+str(ave_prob[pred[4]])+')'
        label5='('+pred[5]+','+str(ave_prob[pred[5]])+')'
        writer.writerow((image,image_label,label0,label1,label2,label3,label4,label5)) 
        # ft.write(image+','+image_label+'\n')
        # writer.writerow([image,image_label])
        p=image+','+image_label+','+label0+','+label1+','+label2+','+label3+','+label4+','+label5+'\n'
        ft.write(p)

    #微波图像
    if 'IALT' in image:

        net.blobs['data'].data[...] = transformer.preprocess('data', im)
        predict = net.forward()
        prob = net.blobs['prob'].data[0].flatten()
        ######################
        ave_prob[names_disorder[0]]+=prob[0]/3
        ave_prob[names_disorder[1]]+=prob[1]/3
        ave_prob[names_disorder[2]]+=prob[2]/3
        ave_prob[names_disorder[3]]+=prob[3]/3
        ave_prob[names_disorder[4]]+=prob[4]/3
        ave_prob[names_disorder[5]]+=prob[5]/3
        ##############        
        #
        net_invert.blobs['data'].data[...] = transformer_invert.preprocess('data', im)
        predict_invert = net_invert.forward()
        prob_invert = net_invert.blobs['prob'].data[0].flatten()
        ######################
        ave_prob[names_label3[0]]+=prob_invert[0]/3
        ave_prob[names_label3[1]]+=prob_invert[1]/3
        ave_prob[names_label3[2]]+=prob_invert[2]/3
        ave_prob[names_label3[3]]+=prob_invert[3]/3
        ave_prob[names_label3[4]]+=prob_invert[4]/3
        ave_prob[names_label3[5]]+=prob_invert[5]/3
        ##############        
        #
        net_disorder.blobs['data'].data[...] = transformer_disorder.preprocess('data', im)
        predict_disorder = net_disorder.forward()
        prob_disorder = net_disorder.blobs['prob'].data[0].flatten()
        ######################
        ave_prob[names_label4[0]]+=prob_disorder[0]/3
        ave_prob[names_label4[1]]+=prob_disorder[1]/3
        ave_prob[names_label4[2]]+=prob_disorder[2]/3
        ave_prob[names_label4[3]]+=prob_disorder[3]/3
        ave_prob[names_label4[4]]+=prob_disorder[4]/3
        ave_prob[names_label4[5]]+=prob_disorder[5]/3
        ##############
        index_in=[ave_prob[pred[0]],ave_prob[pred[1]],ave_prob[pred[2]],ave_prob[pred[3]],ave_prob[pred[4]],ave_prob[pred[5]]]
        image_label=pred[np.argmax(index_in)]
        if ave_prob[image_label]<0.9:
            image_label='UNKNOWN'
            ave_prob[image_label]=0.00001    
        # label0='('+pred[0]+','+str(round(ave_prob[pred[0]],2))+')'
        # label1='('+pred[1]+','+str(round(ave_prob[pred[1]],2))+')'
        # label2='('+pred[2]+','+str(round(ave_prob[pred[2]],2))+')'
        # label3='('+pred[3]+','+str(round(ave_prob[pred[3]],2))+')'
        # label4='('+pred[4]+','+str(round(ave_prob[pred[4]],2))+')'
        # label5='('+pred[5]+','+str(round(ave_prob[pred[5]],2))+')'
        label0='('+pred[0]+','+str(ave_prob[pred[0]])+')'
        label1='('+pred[1]+','+str(ave_prob[pred[1]])+')'
        label2='('+pred[2]+','+str(ave_prob[pred[2]])+')'
        label3='('+pred[3]+','+str(ave_prob[pred[3]])+')'
        label4='('+pred[4]+','+str(ave_prob[pred[4]])+')'
        label5='('+pred[5]+','+str(ave_prob[pred[5]])+')'
        writer.writerow([image,image_label,label0,label1,label2,label3,label4,label5]) 
        # ft.write(image+','+image_label+'\n')
        # writer.writerow([image,image_label])
        p=image+','+image_label+','+label0+','+label1+','+label2+','+label3+','+label4+','+label5+'\n'
        ft.write(p)       

    elapsed=(time.clock()-start)
    # print('time used:',elapsed)
ft.close()
csvfile.close()

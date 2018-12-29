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

net_file = '/home/clxia/caffe-master/models/SeNet/SE-ResNeXt-101_test.prototxt'
caffe_model = '/home/clxia/caffe-master/models/SeNet/solver_iter_60000_label2.caffemodel'
mean_file = '/home/clxia/caffe-master/data/tianyi/imagenet_mean.binaryproto'
print('Params loaded!')
net = caffe.Net(net_file,
                caffe_model,
                caffe.TEST)
#逆序ｌａｂｅｌ
net_file_invert = '/home/clxia/caffe-master/models/SeNet/SE-ResNeXt-101_test.prototxt'
caffe_model_invert = '/home/clxia/caffe-master/models/SeNet/solver_iter_60000-499-label3.caffemodel'
net_invert = caffe.Net(net_file_invert,
                caffe_model_invert,
                caffe.TEST)
#乱序label
net_file_disorder = '/home/clxia/caffe-master/models/SeNet/SE-ResNeXt-101_test.prototxt'
caffe_model_disorder = '/home/clxia/caffe-master/models/SeNet/solver_iter_60000_label4.caffemodel'
net_disorder = caffe.Net(net_file_disorder,
                caffe_model_disorder,
                caffe.TEST)

mean_blob = caffe.proto.caffe_pb2.BlobProto()  # 创建protobuf blob
mean_blob.ParseFromString(open(mean_file, 'rb').read())    # 读入mean.binaryproto文件内容, # 解析文件内容到blob
mean_npy = caffe.io.blobproto_to_array(mean_blob) # 将blob中的均值转换成numpy格式，array的shape （mean_number，channel, hight, width）

#MWI
a = mean_npy[0, :, 0, 0]
print(net.blobs['data'].data.shape)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', a)
transformer.set_raw_scale('data', 255.0)
transformer.set_channel_swap('data', (2, 1, 0))

#逆序ｌａｂｅｌ
print(net_invert.blobs['data'].data.shape)
transformer_invert = caffe.io.Transformer({'data': net_invert.blobs['data'].data.shape})
transformer_invert.set_transpose('data', (2, 0, 1))
transformer_invert.set_mean('data', a)
transformer_invert.set_raw_scale('data', 255.0)
transformer_invert.set_channel_swap('data', (2, 1, 0))
#乱序label
print(net_disorder.blobs['data'].data.shape)
transformer_disorder = caffe.io.Transformer({'data': net_disorder.blobs['data'].data.shape})
transformer_disorder.set_transpose('data', (2, 0, 1))
transformer_disorder.set_mean('data', a)
transformer_disorder.set_raw_scale('data', 255.0)
transformer_disorder.set_channel_swap('data', (2, 1, 0))


#IALT
ialt_mean=80.893915
ialt=np.array([ialt_mean,ialt_mean,ialt_mean])
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
csvfile=open('/home/clxia/caffe-master/data/res-54.csv','w') 
#delimiter默认是逗号
writer=csv.writer(csvfile,delimiter=',')
#opposite label
names = ['DESERT', 'OCEAN', 'MOUNTAIN', 'FARMLAND', 'LAKE', 'CITY']
# 与原文label相反:1
names_invert= ['CITY','LAKE','FARMLAND','MOUNTAIN','OCEAN','DESERT']
# 与原文label相反:2
names_disorder= ['LAKE','FARMLAND','OCEAN','DESERT','CITY','MOUNTAIN']
# 与原文label相反:3
names_label3= ['CITY','MOUNTAIN','LAKE','FARMLAND','OCEAN','DESERT']
#与原文label相反　：４乱序
names_label4=['OCEAN','DESERT','CITY','MOUNTAIN','LAKE','FARMLAND']

imtonu={'LAKE':0,'FARMLAND':1,'OCEAN':2,'DESERT':3,'CITY':4,'MOUNTAIN':5}

pred=['LAKE','FARMLAND','OCEAN','DESERT','CITY','MOUNTAIN']
for image in image_names:
    start=time.clock()
    img_path='/home/clxia/caffe-master/data/test/'+image
    # img_path='/home/clxia/Downloads/高度计数据集-预算训练集500/预赛训练集-500/'+image

    im = caffe.io.load_image(img_path)
    #多光谱图
    ig=cv2.imread(img_path)
    pre=[]

    if 'MWI' in image:
        # size(256,256)
        net.blobs['data'].data[...] = transformer.preprocess('data', im)
        predict = net.forward()
        prob = net.blobs['prob'].data[0].flatten()
        image_label=names_disorder[np.argmax(prob)]
        pre.append(imtonu[image_label])
        #
        net_invert.blobs['data'].data[...] = transformer_invert.preprocess('data', im)
        predict_invert = net_invert.forward()
        prob_invert = net_invert.blobs['prob'].data[0].flatten()
        image_label_invert=names_label3[np.argmax(prob_invert)]
        pre.append(imtonu[image_label_invert])

        #
        net_disorder.blobs['data'].data[...] = transformer_disorder.preprocess('data', im)
        predict_disorder = net_disorder.forward()
        prob_disorder = net_disorder.blobs['prob'].data[0].flatten()
        image_label_disorder=names_label4[np.argmax(prob_disorder)]
        pre.append(imtonu[image_label_disorder])

        # pre_index=max(pre.count(x) for x in set(pre))

        # image_label=pred[pre_index]
        # print image_label
        # print image_label_invert,
        # print image_label_disorder
        # print pre_index
        # print pre
        image_label=pred[sorted(pre)[1]]
        ft.write(image+','+image_label+'\n')
        writer.writerow([image,image_label])
        # img=cv2.imread(img_path)
        # cv2.imwrite('/home/clxia/caffe-master/data/'+image_label+'/'+image,img)
        # break
    #微波图像
    if 'IALT' in image:
        # print 'IALT'
        # img=np.zeros(3,im.shape[0],im.shape[1])
        # img[0,:,:]=im
        # img[1,:,:]=im
        # img[2,:,:]=im
        net.blobs['data'].data[...] = transformer.preprocess('data', im)
        predict = net.forward()
        prob = net.blobs['prob'].data[0].flatten()
        image_label=names_disorder[np.argmax(prob)]
        pre.append(imtonu[image_label])
        #
        net_invert.blobs['data'].data[...] = transformer_invert.preprocess('data', im)
        predict_invert = net_invert.forward()
        prob_invert = net_invert.blobs['prob'].data[0].flatten()
        image_label_invert=names_label3[np.argmax(prob_invert)]
        pre.append(imtonu[image_label_invert])

        #
        net_disorder.blobs['data'].data[...] = transformer_disorder.preprocess('data', im)
        predict_disorder = net_disorder.forward()
        prob_disorder = net_disorder.blobs['prob'].data[0].flatten()
        image_label_disorder=names_label4[np.argmax(prob_disorder)]
        pre.append(imtonu[image_label_disorder])

    elapsed=(time.clock()-start)
    print('time used:',elapsed)




    # # size(256,256)
    # net.blobs['data'].data[...] = transformer.preprocess('data', im)
    # predict = net.forward()
    # prob = net.blobs['prob'].data[0].flatten()
    # image_label=names[np.argmax(prob)]
    # ft.write(image+','+image_label+'\n')
    # writer.writerow([image,image_label])


    # size=(384,384)
    # i=cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
    # img=cv2.resize(im, (320,320), interpolation=cv2.INTER_LINEAR)
    # img_leftup =i[0:320,0:320]
    # img_rightup=i[0:320,64:384]
    # img_leftdown=i[64:384,0:320]
    # img_rightdowm=i[64:384,64:384]

    # size=(336,336)
    # i=cv2.resize(im, (384,384), interpolation=cv2.INTER_LINEAR)
    # img=cv2.resize(im, (336,336), interpolation=cv2.INTER_LINEAR)
    # img_leftup =i[0:336,0:336]
    # img_rightup=i[0:336,48:384]
    # img_leftdown=i[48:384,0:336]
    # img_rightdowm=i[48:384,48:384]

    # size = (224, 224)
    # img = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
    # img_leftup =im[0:224,0:224]
    # img_rightup=im[0:224,32:256]
    # img_leftdown=im[32:256,0:224]
    # img_rightdowm=im[32:256,32:256]
    # L=[]

    # img_list=[img,img_leftdown,img_leftup,img_rightdowm,img_rightup]
    # for imgdata in img_list:
    #     net.blobs['data'].data[...] = transformer.preprocess('data', imgdata)
    #     predict = net.forward()
    #     prob = net.blobs['prob'].data[0].flatten()
    #     L.append(np.argmax(prob))

        # net_senet.blobs['data'].data[...] = transformer.preprocess('data', imgdata)
        # predict = net_senet.forward()
        # prob = net_senet.blobs['prob'].data[0].flatten()
        # L.append(np.argmax(prob))



    # net.blobs['data'].data[...] = transformer.preprocess('data', img)
    # predict = net.forward()
    # prob = net.blobs['prob'].data[0].flatten()
    # L.append(np.argmax(prob))

    # net.blobs['data'].data[...] = transformer.preprocess('data', img_leftup)
    # predict = net.forward()
    # prob = net.blobs['prob'].data[0].flatten()
    # L.append(np.argmax(prob))

    # net.blobs['data'].data[...] = transformer.preprocess('data', img_rightup)
    # predict = net.forward()
    # prob = net.blobs['prob'].data[0].flatten()
    # L.append(np.argmax(prob))

    # net.blobs['data'].data[...] = transformer.preprocess('data', img_leftdown)
    # predict = net.forward()
    # prob = net.blobs['prob'].data[0].flatten()
    # L.append(np.argmax(prob))

    # net.blobs['data'].data[...] = transformer.preprocess('data', img_rightdowm)
    # predict = net.forward()
    # prob = net.blobs['prob'].data[0].flatten()
    # L.append(np.argmax(prob))

    # L_list_sort=sorted(L)

    # if L_list_sort[0] != L_list_sort[2]:
    #     print(image,'prop',L)
    #     img=cv2.imread(img_path)
    #     cv2.imwrite('/home/clxia/caffe-master/data/error/'+image,img)

    # print('prob',sorted(L)[2])
    
    # print('prob: ', prob)
    # print('class: ', names[np.argmax(prob)])

    # ft.write(image+','+names[L_list_sort[2]]+'\n')
    # image_label=names[np.argmax(prob)]
    # # image_label=names[sorted(L)[2]]
    # img=cv2.imread(img_path)
    # cv2.imwrite('/media/clxia/FA5A7ADC63CE92E0/tainyi/'+image_label+'/'+image,img)
    # cv2.imwrite('/home/clxia/caffe-master/data/error/'+image,img)

   

ft.close()
csvfile.close()

# test_img = '/home/clxia/caffe-master/data/test/MWI_kbTEjOqrAKksH9Uj.jpg'
# im = caffe.io.load_image(test_img)
# size=(224,224)
# img=cv2.resize(im,size,interpolation=cv2.INTER_LINEAR)
# net.blobs['data'].data[...] = transformer.preprocess('data', img)
# predict = net.forward()
# names=['DESERT','OCEAN','MOUNTAIN','FARMLAND','LAKE','CITY']
# prob = net.blobs['prob'].data[0].flatten()
# print('prob: ', prob)
# print('class: ', names[np.argmax(prob)])
# names = []
# with open('/home/clxia/caffe-master/examples/tianyi/label.txt', 'r+') as f:
#     for l in f.readlines():
#         names.append(l.split(' ')[1].strip())
#         print(names)
#         prob = net.blobs['prob'].data[0].flatten()
#         print('prob: ', prob)
#         print('class: ', names[np.argmax(prob)])
        # img = cv2.imread(test_img)
        # cv2.imshow('Image', img)
        # cv2.waitKey(0)

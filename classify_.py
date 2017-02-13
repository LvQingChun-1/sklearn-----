#coding:utf8
from  rgbhistogram import RGBHistogram
from sklearn.preprocessing import LabelEncoder	      #机械学习分类器
from sklearn.ensemble import RandomForestClassifier   #分类模型
from sklearn.cross_validation import train_test_split #训练和测试数据 
from sklearn.metrics import classification_report     #判断数据是否检测过
import numpy as np
import argparse
import cv2
import glob

ap=argparse.ArgumentParser()
ap.add_argument("-i","--IMAGES",required=True,help="path to the dataset")
ap.add_argument("-m","--MASKS",required=True,help="path to the image mask")

args=vars(ap.parse_args())


imagePaths=sorted(glob.glob(args["IMAGES"]+"/*.png"))
maskPaths=sorted(glob.glob(args["MASKS"]+"/*.png"))


data=[]
target=[]


desc=RGBHistogram([8,8,8])

for (imagePath,maskPath) in zip(imagePaths,maskPaths):
	image=cv2.imread(imagePath)
	mask=cv2.imread(maskPath)
	mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
	features=desc.describe(image,mask)	
	data.append(features)
	target.append(imagePath.split("_")[-2])


targetNames=np.unique(target)
le=LabelEncoder()
target=le.fit_transform(target)


(trainData,testData,trainTarget,testTarget)=train_test_split(data,target,test_size=0.3,random_state=42)
model=RandomForestClassifier(n_estimators=25,random_state=84)
model.fit(trainData,trainTarget)
print classification_report(testTarget,model.predict(testData),target_names=targetNames)


for i in np.random.choice(np.arange(0,len(imagePaths)),10):
	imagePath=imagePaths[i]	
	image=cv2.imread(imagePath)
	maskPath=maskPaths[i]
	mask=cv2.imread(maskPath)
	mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
	features=desc.describe(image,mask)
	flower=le.inverse_transform(model.predict(features))[0]
	print "I think this flower is a %s" %(flower.upper())
	cv2.imshow("image",image)
	cv2.waitKey(0)

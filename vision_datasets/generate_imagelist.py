import os,glob

fileList1 = []
fileList2 = []

os.chdir("datasets/train/")

fileList1 = glob.glob("*.png")

os.chdir("datasets/trainannot/")

fileList2 = glob.glob("*.png")

assert len(fileList1)==len(fileList2)

os.chdir("../")
fileList1.sort()
fileList2.sort()

textfile = open("train.txt","w")

finalData = [] 
for cnt,i in enumerate(fileList1):
	finalData.append("<"+i+">,<"+fileList2[cnt]+">\n")

textfile.writelines(finalData)

textfile.close()

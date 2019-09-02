import os,glob

fileList1 = []
fileList2 = []

os.chdir("images")

fileList1 = glob.glob("*.png")

os.chdir("../annotations")

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

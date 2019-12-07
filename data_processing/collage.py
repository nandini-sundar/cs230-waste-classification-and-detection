import os
import random
from PIL import Image
from PIL import ImageDraw
import xml.etree.ElementTree as etree
from itertools import groupby
from itertools import combinations

def reload(typeOfWaste):
    wasteList = os.listdir('/Volumes/Data/CS230-Cropped/'+typeOfWaste+'_cropped/')
    wasteList = [x.replace('.jpg','').replace('.png','') for x in wasteList]
    wasteList.remove('.DS_Store')
    return wasteList

def split_text(s):
    for k, g in groupby(s, str.isalpha):
        yield ''.join(g)

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

classes = {'glass':0, 'metal':1, 'plastic':2, 'cardboard':3, 'trash':4, 'paper':5}
masterList = []
glassList = reload("glass")
print(len(glassList))
cardboardList = reload("cardboard")
print(len(cardboardList))
paperList = reload("paper")
print(len(paperList))
plasticList = reload("plastic")
print(len(plasticList))
trashList = reload("trash")
print(len(trashList))
metalList = reload("metal")
print(len(metalList))
comb = combinations(['glass','metal','plastic','cardboard','trash','paper'], 4)
listComb = list(comb)
print(listComb)
for i in range(0,100):
    print(i)
    glass = glassList[i]
    print(glass)
    cardboard = cardboardList[i]
    print(cardboard)
    paper = paperList[i]
    print(paper)
    plastic = plasticList[i]
    print(plastic)
    trash = trashList[i]
    print(trash)
    metal = metalList[i]
    print(metal)
    for c in listComb:
        print(c)
        new_items = []
        listC = list(c)
        print(listC)
        print(type(listC))
        random.shuffle(listC)
        print(listC)
        for x in listC:
            if x == 'glass':
                new_items.append(glass)
            if x == 'metal':
                new_items.append(metal)
            if x == 'plastic':
                new_items.append(plastic)
            if x == 'cardboard':
                new_items.append(cardboard)
            if x == 'trash':
                new_items.append(trash)
            if x == 'paper':
                new_items.append(paper)
        masterList.append(new_items)
print(masterList)
print(len(masterList))

count_filename = 1
for ll in masterList:
    print(ll)
    im = Image.new('RGBA', (1200, 800), 'white')
    im.format = "PNG"
    filename = str(count_filename)+'.png'
    count_filename = count_filename+1
    print(filename)
    A = [os.path.join('/Volumes/Data/CS230-Cropped/',
                      list(split_text(x))[0].replace('.png', '') + '_cropped/' + x+'.png') for x in ll]
    count = 0
    xoff = 15
    yoff = 15
    coordinates = ((0,0),(600,0),(0, 400),(600, 400))
    dictObjects = {}
    for img in A:
        x,y = coordinates[count]
        imgg = Image.open(img)
        object_name = list(split_text(img.split('/')[-1]))[0]
        imgg.thumbnail((512, 384), Image.ANTIALIAS)
        bbox = imgg.convert("RGBa").getbbox()
        draw = ImageDraw.Draw(im)
        xmin = x+bbox[0]
        ymin = y+bbox[1]
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[3]
        dictObjects[object_name] = [xmin, ymin, xmax, ymax]
        im.paste(imgg, (x,y), imgg)
        count = count + 1
    pathh = os.path.join('/Volumes/Data/CS230-Collaged_Data/Images/')
    if not os.path.exists(pathh):
       os.mkdir(pathh)
    else:
        filepath = os.path.join(pathh, filename)
        print(filepath)
    im.save(filepath)
    object_keys = list(dictObjects.keys())
    object_bbx = [dictObjects[x] for x in object_keys]
    print(object_keys)
    print(object_bbx)
    xmlname = filename.split('.')[0]+'.txt'
    with open('/Volumes/Data/CS230-Collaged_Data/Labels/'+xmlname, 'w') as f:
        for bbox,bboxcls in zip(object_bbx,object_keys):
            [xmin,ymin,xmax,ymax] = bbox
            b = (float(xmin), float(xmax), float(ymin), float(ymax))
            bb = convert((1200,800), b)
            print(classes[bboxcls])
            print(bb)
            f.write(str(classes[bboxcls]) + " " + " ".join([str(a) for a in bb]) + '\n')

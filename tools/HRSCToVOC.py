import os
import cv2
from xml.dom.minidom import Document
import xml.dom.minidom

#windows下无需
import sys
stdi, stdo, stde = sys.stdin, sys.stdout, sys.stderr
# reload(sys)
# sys.setdefaultencoding('utf-8')
sys.stdin, sys.stdout, sys.stderr = stdi, stdo, stde


category_set = {}

path = "/Users/liushaodong/AGraduateDesign/dataset/HRSC2016_dataset/HRSC2016/FullDataSet/sysdata.xml"
dom = xml.dom.minidom.parse(path)
root = dom.documentElement
cls = root.getElementsByTagName('HRSC_Class')
for child in cls:
    number = child.getElementsByTagName('Class_NO')
    name = child.getElementsByTagName('Class_Name')
    print(name[0].firstChild.data)
    category_set[number[0].firstChild.data] = name[0].firstChild.data

print(category_set.keys())
print(category_set.values())
print(len(category_set.keys()))


def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])


def limit_value(a,b):
    if a<1:
        a = 1
    if a>=b:
        a = b-1
    return a


def readlabelxml(xml_path, height, width, hbb = True):
    print(xml_path)
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    HRSC_Objects = root.getElementsByTagName('HRSC_Object')
    boxes = []
    for obj in HRSC_Objects:
        xx1 = obj.getElementsByTagName('box_xmin')[0].firstChild.data
        xx2 = obj.getElementsByTagName('box_xmax')[0].firstChild.data
        yy1 = obj.getElementsByTagName('box_ymin')[0].firstChild.data
        yy2 = obj.getElementsByTagName('box_ymax')[0].firstChild.data
        label_ID = obj.getElementsByTagName('Class_ID')[0].firstChild.data
        box = [xx1,yy1,xx2,yy2,category_set[label_ID]]
        boxes.append(box)
    return boxes

def writeXml(tmp, imgname, w, h, d, bboxes, hbb = True):
    doc = Document()
    #owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    #owner
    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    folder_txt = doc.createTextNode("JPEGImages")
    folder.appendChild(folder_txt)

    filename = doc.createElement('filename')
    annotation.appendChild(filename)
    filename_txt = doc.createTextNode(imgname + '.jpg')
    filename.appendChild(filename_txt)
    #ones#
    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    source.appendChild(database)
    database_txt = doc.createTextNode("My Database")
    database.appendChild(database_txt)

    annotation_new = doc.createElement('annotation')
    source.appendChild(annotation_new)
    annotation_new_txt = doc.createTextNode("VOC2007")
    annotation_new.appendChild(annotation_new_txt)

    image = doc.createElement('image')
    source.appendChild(image)
    image_txt = doc.createTextNode("flickr")
    image.appendChild(image_txt)
    #owner
    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid = doc.createElement('flickrid')
    owner.appendChild(flickrid)
    flickrid_txt = doc.createTextNode("NULL")
    flickrid.appendChild(flickrid_txt)

    ow_name = doc.createElement('name')
    owner.appendChild(ow_name)
    ow_name_txt = doc.createTextNode("idannel")
    ow_name.appendChild(ow_name_txt)
    #onee#
    #twos#
    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(w))
    width.appendChild(width_txt)

    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(h))
    height.appendChild(height_txt)

    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode(str(d))
    depth.appendChild(depth_txt)
    #twoe#
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented_txt = doc.createTextNode("0")
    segmented.appendChild(segmented_txt)

    for bbox in bboxes:
        #threes#
        object_new = doc.createElement("object")
        annotation.appendChild(object_new)

        name = doc.createElement('name')
        object_new.appendChild(name)
        name_txt = doc.createTextNode(str(bbox[-1]))
        name.appendChild(name_txt)

        pose = doc.createElement('pose')
        object_new.appendChild(pose)
        pose_txt = doc.createTextNode("Unspecified")
        pose.appendChild(pose_txt)

        truncated = doc.createElement('truncated')
        object_new.appendChild(truncated)
        truncated_txt = doc.createTextNode("0")
        truncated.appendChild(truncated_txt)

        difficult = doc.createElement('difficult')
        object_new.appendChild(difficult)
        difficult_txt = doc.createTextNode("0")
        difficult.appendChild(difficult_txt)
        #threes-1#
        bndbox = doc.createElement('bndbox')
        object_new.appendChild(bndbox)

        if hbb:
            xmin = doc.createElement('xmin')
            bndbox.appendChild(xmin)
            xmin_txt = doc.createTextNode(str(bbox[0]))
            xmin.appendChild(xmin_txt)

            ymin = doc.createElement('ymin')
            bndbox.appendChild(ymin)
            ymin_txt = doc.createTextNode(str(bbox[1]))
            ymin.appendChild(ymin_txt)

            xmax = doc.createElement('xmax')
            bndbox.appendChild(xmax)
            xmax_txt = doc.createTextNode(str(bbox[2]))
            xmax.appendChild(xmax_txt)

            ymax = doc.createElement('ymax')
            bndbox.appendChild(ymax)
            ymax_txt = doc.createTextNode(str(bbox[3]))
            ymax.appendChild(ymax_txt)
        else:
            x0 = doc.createElement('x0')
            bndbox.appendChild(x0)
            x0_txt = doc.createTextNode(str(bbox[0]))
            x0.appendChild(x0_txt)

            y0 = doc.createElement('y0')
            bndbox.appendChild(y0)
            y0_txt = doc.createTextNode(str(bbox[1]))
            y0.appendChild(y0_txt)

            x1 = doc.createElement('x1')
            bndbox.appendChild(x1)
            x1_txt = doc.createTextNode(str(bbox[2]))
            x1.appendChild(x1_txt)

            y1 = doc.createElement('y1')
            bndbox.appendChild(y1)
            y1_txt = doc.createTextNode(str(bbox[3]))
            y1.appendChild(y1_txt)

            x2 = doc.createElement('x2')
            bndbox.appendChild(x2)
            x2_txt = doc.createTextNode(str(bbox[4]))
            x2.appendChild(x2_txt)

            y2 = doc.createElement('y2')
            bndbox.appendChild(y2)
            y2_txt = doc.createTextNode(str(bbox[5]))
            y2.appendChild(y2_txt)

            x3 = doc.createElement('x3')
            bndbox.appendChild(x3)
            x3_txt = doc.createTextNode(str(bbox[6]))
            x3.appendChild(x3_txt)

            y3 = doc.createElement('y3')
            bndbox.appendChild(y3)
            y3_txt = doc.createTextNode(str(bbox[7]))
            y3.appendChild(y3_txt)

    xmlname = os.path.splitext(imgname)[0]
    tempfile = os.path.join(tmp ,xmlname+'.xml')
    with open(tempfile, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    return



# if __name__ == '__main__':
#     label_path = "/Users/liushaodong/AGraduateDesign/dataset/HRSC2016_dataset/HRSC2016/FullDataSet/Annotations"
#     new_path = "/Users/liushaodong/AGraduateDesign/dataset/Ship_All/Annotations"
#     filenames = os.listdir(label_path)
#     for filename in filenames:
#         if filename != '100001073.xml':
#             continue
#         filepath = label_path + '/' + filename
#         dom = xml.dom.minidom.parse(filepath)
#         root = dom.documentElement
#         W = int(root.getElementsByTagName('Img_SizeWidth')[0].firstChild.data)
#         H = int(root.getElementsByTagName('Img_SizeHeight')[0].firstChild.data)
#         D = int(root.getElementsByTagName('Img_SizeDepth')[0].firstChild.data)
#         picname = root.getElementsByTagName('Img_FileName')[0].firstChild.data
#         boxes = readlabelxml(filepath, H, W)
#         if len(boxes) == 0:
#             print('文件为空', filepath)
#         writeXml(new_path, picname, W, H, D, boxes, hbb = True)
#         print('正在处理%s'%filename)






# # 图片的路径
# bmp_dir = '/Users/liushaodong/AGraduateDesign/dataset/HRSC2016_dataset/HRSC2016/FullDataSet/AllImages'
# jpg_dir = '/Users/liushaodong/AGraduateDesign/dataset/Ship_All/JPEGImages'
#
# filelists = os.listdir(bmp_dir)
#
# for i,file in enumerate(filelists):
#     # 读图，-1为不改变图片格式，0为灰度图
#     img = cv2.imread(os.path.join(bmp_dir,file),-1)
#     newName = file.replace('.bmp','.jpg')
#     cv2.imwrite(os.path.join(jpg_dir,newName),img)
#     print('第%d张图：%s'%(i+1,newName))

import requests
import json
import time
import cv2
import re
import base64
import numpy as np

from field import Field

class VLM:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8001"

        self.img_path=""
        self.st_lock=True
        self.range=np.pi/2
        self.max_distance=2
        self.take_width = 2

        self.img=None

    def imgmask2b64(self,path,mask=[]):
        img = cv2.imread(path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # green
        mask_list=[]
        if("green" in mask):
            lower_green = np.array([35, 43, 46])
            upper_green = np.array([77, 255, 255])
            mask_list.append(cv2.inRange(hsv, lower_green, upper_green))

        if ("red" in mask):
            lower_red = np.array([120, 50, 50])
            upper_red = np.array([180, 255, 255])
            mask_list.append(cv2.inRange(hsv, lower_red, upper_red))

        if(len(mask_list)==0):
            mask=None
        elif(len(mask_list)==1):
            mask=mask_list[0]
        else:
            mask = mask_list[0]
            for i in range(1,len(mask_list)):
                mask = cv2.bitwise_or(mask, mask_list[i])

        res = cv2.bitwise_and(img, img, mask=mask)
        self.img = mask
        data = np.array(cv2.imencode('.png', res)[1]).tobytes()
        base64_data = base64.b64encode(data)
        return base64_data

    def img2b64(self,path):
        with open(path, "rb") as f:
            base64_data = base64.b64encode(f.read())
        return base64_data

    def upload(self, img_path, mask=[]):
        if(len(mask)!=0):
            img=self.imgmask2b64(img_path,mask)

        else:
            img=self.img2b64(img_path)
        data = {"img": img, "name": img_path}
        response = requests.post(f"{self.base_url}/upload", data=data)
        r = json.loads(response.text)
        self.img_path = img_path
        # print(r)

        return r


    def analysis(self,prompt):
        prompt='[grounding] ' + prompt + ' What items were described in the above? Where are them?'
        t = time.time()
        data = {"msg": prompt}
        response = requests.post(f"{self.base_url}/ask", data=data)
        r = json.loads(response.text)

        obj = []

        pattern = "<p>[A-Za-z0-9 ]+</p> {<[0-9]+><[0-9]+><[0-9]+><[0-9]+>}"
        p = re.compile(pattern, re.MULTILINE)
        for p0 in p.findall(r["asw"]):
            ptn = re.compile(r'<p>(?P<dscrb>.+)</p> {<(?P<x1>.+)><(?P<y1>.+)><(?P<x2>.+)><(?P<y2>.+)>}', re.MULTILINE)
            msg_data = next(ptn.finditer(p0))
            obj.append({"name": msg_data.group('dscrb'), "pos": ((int(msg_data.group('x1')), int(msg_data.group('y1'))),
                                                                 (int(msg_data.group('x2')),
                                                                  int(msg_data.group('y2'))))})
        # print(obj)
        return obj

    def get_mask_pos(self,mask1):
        shape = mask1.shape
        # print(shape)

        c = np.count_nonzero(mask1 != 0, axis=1)
        min_g = 0
        max_g = 0
        min_max = False
        for i in range(len(c)):
            if (not min_max and c[i] >= 35):
                # print(i)
                for j in range(len(mask1[i])):
                    if (mask1[i, j] != 0):
                        min_g = (i, j)
                        break
                min_max = True
            if (min_max and c[i] < 5):
                # print(i - 1)
                jdg = False
                for j in range(len(mask1[i - 1])):
                    if (mask1[i - 1, j] != 0):
                        jdg = True
                    if (jdg and mask1[i - 1, j] == 0):
                        max_g = (i - 1, j)
                        break
                break
        min_g = (int(min_g[0] / shape[0] * 100), int(min_g[1] / shape[1] * 100))
        max_g = (int(max_g[0] / shape[0] * 100), int(max_g[1] / shape[1] * 100))
        # print(min_g, max_g)
        return (min_g[1], min_g[0]), (max_g[1], max_g[0])

    def detection_color(self,prompt):
        obj=[]
        min_g, max_g=self.get_mask_pos(self.img)
        obj.append({"name": prompt,
                    "pos": (
                        min_g,
                        max_g
                    )})
        return obj

    def detection(self,prompt):
        prompt="[detection] "+prompt
        t = time.time()
        data = {"msg": prompt}
        response = requests.post(f"{self.base_url}/ask", data=data)
        r = json.loads(response.text)

        obj = []

        pattern = "<p>[A-Za-z0-9 ]+</p> {<[0-9]+><[0-9]+><[0-9]+><[0-9]+>}"
        p = re.compile(pattern, re.MULTILINE)
        for p0 in p.findall(r["asw"]):
            ptn = re.compile(r'<p>(?P<dscrb>.+)</p> {<(?P<x1>.+)><(?P<y1>.+)><(?P<x2>.+)><(?P<y2>.+)>}', re.MULTILINE)
            msg_data = next(ptn.finditer(p0))
            obj.append({"name": msg_data.group('dscrb'), "pos": ((int(msg_data.group('x1')), int(msg_data.group('y1'))),
                                                                 (int(msg_data.group('x2')),
                                                                  int(msg_data.group('y2'))))})
        return obj

    def draw(self,obj):
        img = cv2.imread(self.img_path)

        height, width, _ = img.shape
        for i in obj:
            pos = i["pos"]
            posa = (int(pos[0][0] / 100 * width), int(pos[0][1] / 100 * height))
            posb = (int(pos[1][0] / 100 * width), int(pos[1][1] / 100 * height))
            # posa=pos[0]
            # posb=pos[1]
            print(i["name"], posa, posb)
            img = cv2.rectangle(img, posa, posb, (0, 0, 0), 2)
            cv2.putText(img, i["name"], (posa[0], posa[1] + 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('border', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite("aaaa.png", img)

    def mapping(self,pos):
        p1 = pos[0]
        p2 = pos[1]
        w = (p2[0] - p1[0])/100
        h = (p2[1] - p1[1])/100
        # r=1-(w+h)/2
        r=1-w

        r=r*r*(self.max_distance+self.take_width)-self.take_width

        r=r if r>0 else 0

        fi= - ((p2[0] + p1[0])/2 -50)/50*(self.range/2)
        if(self.st_lock):
            st=np.pi/2
        else:
            st = ((p2[1] + p1[1])/2 -50)/50*(self.range/2)+np/2

        if(fi<0):
            fi=np.pi*2+fi
        if (st < 0):
            st = np.pi * 2 +st

        # print(r,st,fi)
        return [r,st,fi]
    

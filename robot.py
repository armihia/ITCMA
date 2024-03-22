import base64
import time
import requests
import json
from vision_language_model import VLM
from field import Field
import numpy as np

class Env:
    def __init__(self):
        env_name = "robot"
        server = False
        self.port = "8888"
        self.base_url = "http://127.0.0.1:" + self.port
        self.controller_url = "http://192.168.119.138:8704"
        self.camera_url = "http://192.168.0.102:8010"

        self.action_space = []
        self.describe = ""
        self.goal = ""
        self.oracle = []
        self.obs = ""
        self.obs_field=None

        self.score = 0
        self.server = server
        self.init_status = False

        self.vlm=VLM()

        self.reset()

        if (len(self.action_space) != 0):
            self.init_status = True

    def show(self):
        print("action_space: ", self.action_space, "\ndescribe: ", self.describe, "\ngoal: ", self.goal,
              "\nobservation: ", self.obs, "\noracle: ", self.oracle, "\nscore: ", self.score)

    def controller(self,mov):
        img_path="tmp.jpg"

        print(mov)
        msg="finish"

        if(msg=="finish"):
            response = requests.post(f"{self.camera_url}/get_img")
            r = json.loads(response.text)

            b64 = r["b64"]
            img_path = r["name"]

            imgdata = base64.b64decode(b64)
            file = open("camera/"+img_path, 'wb')
            file.write(imgdata)
            file.close()
        return "camera/"+img_path

    def controller_imagine(self,f,a):
        f=f.copy()
        distance=0.5
        angel=np.pi/6
        if(a=="move forward"):
            f.relocation([distance,np.pi/2,0])
        elif (a == "move backwards"):
            f.relocation([-distance, np.pi/2, 0])
        elif (a == "move to the left"):
            f.rotate([np.pi/2,np.pi/2])
            f.relocation([distance, np.pi/2, 0])
            f.rotate([np.pi/2, -np.pi / 2])
        elif (a == "move to the right"):
            f.rotate([np.pi/2, -np.pi / 2])
            f.relocation([distance, np.pi/2, 0])
            f.rotate([np.pi/2, np.pi / 2])
        elif (a == "rotate to the left"):
            f.rotate([np.pi/2, angel])
        elif (a == "rotate to the right"):
            f.rotate([np.pi/2, -angel])

        return f


    def interactive_tmp(self,func,data,mov=None,imagine_pi=None):
        img_path = '../demo.png'
        goal = "Your task is to: Push the box to the red area and then to the green area."
        # action_space=["move forward", "move backwards", "move to the left", "move to the right",
        #                                 "rotate to the left", "rotate to the right"]
        as0=["go to red area","go to green area","take one box from one box"]
        as1=["put one box in/on red area"]
        as2=["put one box in/on green area"]
        #action_space = as0+as1+as2

        oracle=["take one box from one box","go to red area","go to green area","put one box in/on green area"]
        score=0

        if (imagine_pi == None):
            if(func in ["action"]):

                act=data["actions"]
                #f=self.controller_imagine(self.obs_field,act)
                img_path=self.controller(mov)
                f = self.img_analysis(img_path)
            else:
                f=self.img_analysis(img_path)
        else:
            f=imagine_pi

        dscrb = f.field2text()
        # f.show()
        if("You are in the middle of a place" in dscrb or "On the one box" in dscrb):
            action_space = as0
        elif ("On the red area" in dscrb and "You are holding a one box" in dscrb):
            action_space = as0+as1
        elif ("On the green area" in dscrb and "You are holding a one box" in dscrb):
            action_space = as0+as2

        # f.show()

        if (func in ["reset"]):
            obs = "-= Welcome to env =-\n\n" + \
                  dscrb + "\n\n" + \
                  goal
        else:
            obs = dscrb

        r = {
            "obs": obs,  # observation
            "infos": {
                'admissible_commands': action_space,  # action_space
                'policy_commands': oracle,  # oracle
            },
            "score": score,
            "field": f
        }
        return r

    def img_analysis(self,img_path):
        obj=[]

        try:
            self.vlm.upload(img_path, ["green"])
            prompt = 'green area'
            obj+=self.vlm.detection_color(prompt)
        except:
            pass

        try:
            self.vlm.upload(img_path, ["red"])
            prompt = 'red area'
            obj+= self.vlm.detection_color(prompt)
        except:
            pass

        self.vlm.upload(img_path)
        prompt = 'one box'
        obj += self.vlm.detection(prompt)

        for i in range(len(obj)):
            pos = obj[i]["pos"]
            obj[i]["coordinate"] = self.vlm.mapping(pos)

        # print(obj)

        f = Field(create="dict_structure", create_elements=obj)
        # f.show(radius=self.vlm.max_distance)
        self.vlm.draw(obj)

        return f




    def connecting(self, func="reset",data=None,mov=None,imagine_pi=None):

        r = self.interactive_tmp(func,data,mov=mov,imagine_pi=imagine_pi)

        return r


    def reset(self):

        r= self.connecting("reset")


        self.action_space = r['infos']['admissible_commands']
        self.oracle = r['infos']['policy_commands']
        _, self.describe, self.goal, _ = self.txt_analysis(r["obs"], False, True)

        self.score = 0
        self.obs = self.describe
        self.obs_field=r["field"]
        return r

    def get_info(self):
        r = self.connecting("get_info")
        self.action_space = r['infos']['admissible_commands']
        self.oracle = r['infos']['policy_commands']

        return r

    def action(self, act,mov=None,imagine_pi=None):
        data = {"actions": act}
        r = self.connecting("action",data=data,mov=mov,imagine_pi=imagine_pi)
        self.action_space = r['infos']['admissible_commands']
        self.oracle = r['infos']['policy_commands']
        self.score += r['score']
        self.obs = r["obs"]
        self.obs_field=r["field"]
        return r['obs']

    def txt_analysis(self, txt, show=False, obs=False):
        action_space = []
        describe = ""
        goal = ""
        oracle = []

        txt = txt.split("\n")

        try:
            if (obs):
                describe = txt[2]
            else:
                describe = txt[3]
        except:
            pass

        for t in txt:
            if ("Available actions: [" in t):
                t = t.replace("Available actions: ", "")
                action_space = eval(t)
            if ("Your task is to: " in t):
                goal = t
            if ("Oracle: [" in t):
                t = t.replace("Oracle: [None/None|(None): ", "").replace("]", "").split(" > ")
                oracle = t
        if (show):
            print("action_space: ", action_space, "\ndescribe: ", describe, "\ngoal: ", goal, "\noracle: ", oracle)

        return action_space, describe, goal, oracle

    def get_status(self):
        try:
            return self.container.status
        except:
            return False

    def run(self, cmd):
        code, stream = self.container.exec_run(cmd, detach=False, stream=True, stderr=True, stdout=True)
        s = ''
        for x in stream:
            s += x.decode()

        # print(s)
        return s

    def stop(self, code=0):
        if (self.server):
            self.container.stop()
            if (code != 0):
                self.container.remove()


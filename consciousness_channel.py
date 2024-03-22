import json
from field import Field
import re
import numpy as np
from memory import Memory
import copy
import time

from large_language_model import LLM

class ConsciousnessChannel:
    def __init__(self,memory_file=None):#
        self.pi=None
        self.re=[]
        self.pro_last={}
        self.pro={}


        self.d=np.array([0,0,0]) #drive
        self.emotion=np.array([0,0,0]) #[Pleasure,Arousal,Dominance], range=[-1, 1]
        self.emotion_weight=np.array([0.4,0.3,0.3])
        self.memory = Memory(memory_file)
        self.activated_memory=[]

        self.re_size=4

        self.llm=LLM()

        self.last_pro_list = []
        self.last_action_list = ["begin"]
        self.last_d_list = []
        self.last_movement_list = []

    def set_activated_memory(self):
        C=self.re+[self.pi]
        self.activated_memory=self.memory.ActivateMemory(C)

    def set_last_protention(self,pro):
        self.pro_last=pro

    def get_driver(self):
        return self.d

    def text2field(self,text,f=None,method="add"):
        #print(text)
        pattern = re.compile('[A-Za-z0-9 /.,]+you see a ([A-Za-z0-9 ,]+)', re.I)
        try:
            e = pattern.findall(text)[0].replace(" a "," ").replace("and ","").split(", ")
        except:
            e=[]
        if(f==None):
            f=Field("circular_structure",e)
        else:
            f=f.copy()
            if(method=="add"):
                data={}
                for i in e:
                    data[i]="front"
                f.add(data)
        return f

    def change(self,obs,action,score=0,action_space=[],memory_save=False,imagine=False,fuc="prompt",init_pi_set=None):
        movement = [0, 0, 0]
        if(imagine):
            pi=self.pi.copy()
            re=[]
            for i in self.re:
                re.append(i.copy())
            emotion=self.emotion.copy()
            d = self.d.copy()
            pro=copy.deepcopy(self.pro)

            last_d_list=self.last_d_list.copy()
            last_action_list=self.last_action_list.copy()
            last_movement_list=self.last_movement_list.copy()
            last_pro_list=self.last_pro_list.copy()
        else:
            pi = self.pi
            re=self.re
            emotion = self.emotion
            d = self.d

            pro = self.pro

            last_d_list = self.last_d_list
            last_action_list = self.last_action_list
            last_movement_list = self.last_movement_list
            last_pro_list = self.last_pro_list

        if(action in ["begin","Begin"] or "Your task is to: " in action):
            if(init_pi_set==None):
                pi=self.text2field(obs)
            else:
                try:
                    pi.update(init_pi_set)
                except:
                    pi=init_pi_set

        else:
            re.append(pi)
            last_action_list.append(action)

            pi=pi.copy()

            #change emotiom
            pleasure=(emotion[0]+1)/0.5+score
            pleasure=np.tanh(pleasure)
            pleasure=(pleasure-0.5)*2
            
            arousal=0
            n=len(re)
            for i in range(n):
                arousal += 2*(i+1)/(n*(n+1)) * pi.diffrence(re[i])
            arousal=np.tanh(arousal)
            arousal=(arousal-0.5)*2

            dominance=0
            if(action in self.pro_last.keys()):
                dominance=pi.diffrence(self.text2field(self.pro_last[action]))
                dominance=np.tanh(dominance)
            else:
                dominance=0.5
            dominance=(dominance-0.5)*2

            emotion=np.array([pleasure,arousal,dominance])
            d=d+emotion*self.emotion_weight

            movement,pi0=self.mov(pi, action, obs)
            try:
                pi.update(pi0)
            except:
                pi = pi0


            last_movement_list.append(movement)
            last_d_list.append(d)




        if (not imagine):
            memory_jdg=False
            re0=re
            if (len(re) > self.re_size):
                memory_jdg = True
                re0=re.copy()
                re = re[-self.re_size:]

            self.pi = pi
            self.re = re
            self.emotion = emotion
            self.d = d

            pro = self.get_pro(action_space, obs_llm=True, save=False,fuc=fuc)
            last_pro_list.append(pro)
            self.pro_last = self.pro
            self.pro = pro

            if (memory_jdg):
                if (memory_save):
                    self.memory.add(re0[0], last_action_list[0], last_d_list[0], last_pro_list[0], last_movement_list[0])
                last_d_list = last_d_list[-self.re_size:]
                last_action_list = last_action_list[-self.re_size-1:]
                last_movement_list = last_movement_list[-self.re_size:]
                last_pro_list=last_pro_list[-self.re_size:]

            self.last_d_list=last_d_list
            self.last_action_list=last_action_list
            self.last_movement_list=last_movement_list
            self.last_pro_list=last_pro_list

        #pi.show()
        #print(self.field2text(pi))
        return movement

    def mov(self, pi, action,obs):
        movement = [0, 0, 0]
        pi=pi.copy()

        if ("go to " in action):
            movement = pi.relocation_by_front_obj(action.replace("go to ", ""))
            pi = self.text2field(obs, pi, "add")
        elif ("take " in action and "from " in action):
            a_txt = action.replace("take ", "")
            a_txt = a_txt.split(" from ")
            take = a_txt[0]
            fro = a_txt[1]
            movement = pi.relocation_by_front_obj(fro)
            pi.obj_movement(take, [0, np.pi / 2, 0])
        elif ("put " in action and "in/on " in action):
            a_txt = action.replace("put ", "")
            a_txt = a_txt.split(" in/on ")
            take = a_txt[0]
            to = a_txt[1]
            movement = pi.relocation_by_front_obj(to)

            pi.obj_movement(take, pi.monad[pi.get_obj(to)].pv)
        elif ("open " in action):
            movement = pi.relocation_by_front_obj(action.replace("open ", ""))
            pi = self.text2field(obs, pi, "add")

        # pi.show()
        # print(self.field2text(pi))
        return movement,pi

    def getProtention_llm(self,re,pi,am,a,fuc):
        arg = {
            "scene_that_have_occurred": re,
            "ongoing_scene": pi,
            "memory": am,
            "action": a
        }
        qst = "Based on scene_that_have_occurred and referring to the content of the memory, what would happen to the ongoing_scene if an action is taken? Please describe in the following format, following the content of ongoing_scene:" \
              + " 'On the [the place your action will go to or the place of the ongoing scene], you see A, B, C...'"\
                +"\nPlease refer to the content in your memory and convert the results to the format shown above."\
                    +"Please note that if there is content in memory that matches the current situation, focus on referring to that content. \n"

        pro_a = "Nothing will happen"
        if(fuc == "prompt"):
            try:

                pro_a,_ = self.llm.prompt_chat(qst, arg, use_stream=False, f_name="getProtention")
            except:
                try:
                    pro_a = self.llm.function_chat(qst, arg, use_stream=False, f_name="getProtention")
                except:
                    pass
        else:
            try:
                pro_a = self.llm.function_chat(qst, arg, use_stream=False, f_name="getProtention")
            except:
                pass
        return pro_a

    ######
    # pi, re, am, d → pro_a(pro_a→(self.field2text)→pro_text)
    ######

    def get_pro_llm(self,action_space):
        pro={} #protention
        pi=self.pi.field2text() #primal impression
        re=[] #retention
        for r in self.re:
            re.append(r.field2text())
        self.set_activated_memory()
        am=[] #activated memory
        for m in self.activated_memory:
            am.append(m.field2text())
        d=self.get_driver() #drive
        for a in action_space:
            pro_a=self.getProtention_llm(re,pi,am,a)
            print(pi,"\n",a,"\n",pro_a,"\n\n")

            pro[a]=pro_a

        self.pro_last=self.pro
        self.pro=pro
        return pro

    def get_pro(self, action_space,obs_llm=False,save=True,fuc="prompt"):
        pro = {}  # protention
        pi=self.pi #primal impression
        re=self.re #retention
        last_action_list=self.last_action_list
        self.set_activated_memory()
        am=self.activated_memory #activated memory
        d = self.get_driver()  # drive

        pi_text = pi.field2text()
        re_text = []


        for i in range(len(re)):
            r=re[i]
            re_text.append("After " + last_action_list[i] + ", " + re[i].field2text()+", you did the action: "+last_action_list[i+1])

        am_text = []


        for m in am:
            am_text.append(m["content"])

        for a in action_space:
            print("pro: ", a)
            pro_text = "Nothing will happen"

            ach=False

            if(not obs_llm):
                try:
                    obs=am[1].field2text()
                except:
                    obs=""
            else:
                obs=""
                for o in am_text:
                    if("After "+a in o):
                        obs=o.replace("After "+a+", ","")
                        ach=True
                        break
                if(obs==""):
                    obs = self.getProtention_llm(re_text, pi_text, am_text, a,fuc)
            movement, pro_a = self.mov(pi, a, obs)
            pro_text=pro_a.field2text()

            if(ach):
                pro_text+=" The goal may be achievable."



            pro[a] = pro_text.replace("see nothing","don't know what's going to happen")



        if(save):
            self.pro_last = self.pro
            self.pro = pro
        return pro

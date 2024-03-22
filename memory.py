import numpy as np
from field import Field
import random
import copy

class Edge:
    def __init__(self,fro,to,action,movement):
        self.fro=fro
        self.to=to
        self.action=action
        self.movement=movement



class Frame:
    def __init__(self,num,obs,driver,pro):
        self.num=num
        self.obs=obs
        self.driver=driver
        self.pro=pro

    def copy(self):
        return Frame(self.num,self.obs.copy(),self.driver.copy(),copy.deepcopy(self.pro))

class Memory:
    def __init__(self,file=None):
        self.frames={0:Frame(0,Field(),[0,0,0], {})}
        self.memory_flow=[]
        self.link=[]

        self.last_frame=0
        
        if(file!=None):
            dic = np.load(file, allow_pickle='TRUE').item()
            self.frames=dic["frames"]
            self.memory_flow=dic["memory_flow"]
            self.link=dic["link"]
            self.last_frame=dic["last_frame"]


    def get_edge_data(self,e):
        try:
            fro = self.frames[e.fro].copy()
        except:
            fro = None
        try:
            to = self.frames[e.to].copy()
        except:
            to = None

        action = e.action
        movement = e.movement.copy()

        return fro,to,action,movement

    def Edge2Dict(self,edg):
        r = []
        if(len(edg)==0):
            return r

        fro, to, action, movement=self.get_edge_data(edg[0])
        r.append({"fro": fro, "to": to, "action": action, "movement": movement,
         "content": fro.obs.field2text()+" The next action is "+action+"."})

        if (len(edg)>1):
            for i in range(len(edg)-1):
                e=edg[i]
                fro, to, action, movement=self.get_edge_data(e)
                r.append({"fro":fro,"to":to,"action":action,"movement":movement,
                        "content":"After "+action+", "+to.obs.field2text()+" The next action is "+edg[i+1].action+"."})

        fro, to, action, movement = self.get_edge_data(edg[-1])
        r.append({"fro": fro, "to": to, "action": action, "movement": movement,
                  "content": "After "+action+", "+to.obs.field2text()+fro.obs.field2text()})

        return r


    def add(self,obs,last_action,d,pro,movement):
        num=0
        while(num in self.frames.keys()):
            num=int(random.random()*10000000)
        frame=Frame(num,obs,d,pro)
        self.frames[num]=frame
        self.memory_flow.append(Edge(self.last_frame,num,last_action,movement))
        self.last_frame=num

    def save(self,file="memory.npy"):
        dic={}
        dic["frames"]=self.frames
        dic["memory_flow"]=self.memory_flow
        dic["link"]=self.link
        dic["last_frame"]=self.last_frame
        np.save(file, dic)

    def MemoryFlow2FieldFlow(self, bg=0, ed=-1): #self.memory_flow[bg:ed] → [f_bg, ...., f_ed]
        try:
            mf = self.memory_flow[bg:ed]
        except:
            mf = self.memory_flow[bg:]
        try:
            ff = [self.frames[mf[0].fro].obs.copy()]
        except:
            ff = []
        for i in mf:
            n = i.to
            ff.append(self.frames[n].obs.copy())
        return ff

    def MemoryFlow2FieldFlowWithAction(self, bg=0, ed=-1): #self.memory_flow[bg:ed] → [f_bg, ...., f_ed]
        if(bg<0):
            bg=0
        try:
            mf = self.memory_flow[bg:ed]
        except:
            mf = self.memory_flow[bg:]
        ff = []
        ff=self.Edge2Dict(mf)
        return ff

    def ActivateMemory(self,C,window=5):
        # print(C)
        satisfaction=0
        diversity=1000000
        activated_memory_n=-1
        activated_memory_window_n = -1


        s2 = C
        for i in range(len(self.memory_flow)-1,-1,-1):
            for j in range(window):
                s1=self.MemoryFlow2FieldFlow(bg=i-j, ed=i+1)

                l=levenshtein_distance(s1, s2)
                if(l<diversity):
                    # print(i-j, i+1,diversity)
                    diversity=l
                    activated_memory_n=i
                    activated_memory_window_n=j

            if(diversity<=satisfaction):
                break

        activated_memory = self.MemoryFlow2FieldFlowWithAction(bg=activated_memory_n, ed=activated_memory_n+window)

        if(activated_memory_n!=-1):
            print("searched:",self.frames[self.memory_flow[activated_memory_n].to].obs.field2text(),"\n")
            for i in activated_memory:
                print(i["content"])
            print(diversity)
        else:
            print("None")

        return activated_memory


def levenshtein_distance(s1, s2): # s=[f1,f2,f3,...]
    m, n = len(s1), len(s2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    # print(dp)
    field_zero = Field()

    for i in range(m + 1):
        diff = 0
        for k in range(i):
            diff += 1 - s1[k].diffrence(field_zero)
        dp[i][0] = diff
    for j in range(n + 1):
        diff = 0
        for k in range(j):
            diff += 1 - s2[k].diffrence(field_zero)
        dp[0][j] = diff

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + (1 - s1[i - 1].diffrence(field_zero)),
                dp[i][j - 1] + (1 - s2[j - 1].diffrence(field_zero)),
                dp[i - 1][j - 1] + (1 - s1[i - 1].diffrence(s2[j - 1]))
            )
    return dp[m][n]
    

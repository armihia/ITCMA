import docker
import time
import requests
import json

class Env:
    def __init__(self):
        env_name="alfworld"
        server=False
        self.port="8888"
        self.base_url = "http://127.0.0.1:"+self.port

        self.action_space=[]
        self.describe=""
        self.goal=""
        self.oracle=[]
        self.obs=""

        self.score=0
        self.server=server
        self.init_status=False

        if(server):
            self.client = docker.from_env()
            self.container = self.client.containers.run(env_name+':itcm', detach=True, remove=True, tty=True, volumes=['C:/Users/Armihia/Desktop/ITCM/environment/'+env_name+':/'+env_name], command='/bin/bash')
            self.container.reload()
        
            self.run("cd alfworld")
            txt=self.run("alfworld-play-tw")
            #print(txt)
            
            self.action_space,self.describe,self.goal,self.oracle=self.txt_analysis(txt,True)
            self.obs = self.describe
        else:
            self.reset()
            
        if(len(self.action_space)!=0):
            self.init_status=True

    def show(self):
        print("action_space: ",self.action_space,"\ndescribe: ",self.describe,"\ngoal: ",self.goal, "\nobservation: ",self.obs,"\noracle: ",self.oracle,"\nscore: ",self.score)
        
    def reset(self):
        response = requests.post(f"{self.base_url}/reset")
        r=json.loads(response.text)
        self.action_space=r['infos']['admissible_commands']
        self.oracle=r['infos']['policy_commands']
        _,self.describe,self.goal,_=self.txt_analysis(r["obs"],False,True)

        self.score=0
        self.obs=self.describe
        return r

    def get_info(self):
        response = requests.post(f"{self.base_url}/get_info")
        r=json.loads(response.text)
        self.action_space=r['infos']['admissible_commands']
        self.oracle=r['infos']['policy_commands']
        
        return r

    def action(self,act):
        data={"actions":act}
        response = requests.post(f"{self.base_url}/action", data=data)
        r=json.loads(response.text)
        self.action_space=r['infos']['admissible_commands']
        self.oracle=r['infos']['policy_commands']
        self.score+=r['score']
        self.obs=r["obs"]
        return r['obs']
    
    def txt_analysis(self,txt,show=False,obs=False):
        action_space=[]
        describe=""
        goal=""
        oracle=[]
        
        txt=txt.split("\n")

        try:
            if(obs):
                describe=txt[2]
            else:
                describe=txt[3]
        except:
            pass
        
        for t in txt:
            if("Available actions: [" in t):
                t=t.replace("Available actions: ","")
                action_space=eval(t)
            if("Your task is to: " in t):
                goal=t
            if("Oracle: [" in t):
                t=t.replace("Oracle: [None/None|(None): ","").replace("]","").split(" > ")
                oracle=t
        if(show):
            print("action_space: ",action_space,"\ndescribe: ",describe,"\ngoal: ",goal, "\noracle: ",oracle)

        return action_space,describe,goal,oracle

    def get_status(self):
        try:
            return self.container.status
        except:
            return False

    def run(self,cmd):
        code,stream = self.container.exec_run(cmd,detach=False,stream=True,stderr=True, stdout=True)
        s = ''   
        for x in stream:
            s += x.decode()
        
        #print(s)
        return s

    def stop(self,code=0):
        if(self.server):
            self.container.stop()
            if(code!=0):
                self.container.remove( )


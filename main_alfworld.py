from alfworld import Env
from agent import Generative_Agent
from consciousness_channel import  ConsciousnessChannel
import numpy as np

e=Env()
# print(e.get_info())
# input(">>")
ga=Generative_Agent()

env_num="trial_T20190908_042713_075616"
trained=False

if(trained):
    file = "memory.npy"
    file = env_num + ".npy"

    memory_save = False

else:
    file = None
    memory_save = True

cc=ConsciousnessChannel(file)

print( 'Container status:', e.get_status(),"Environment staus",e.init_status)
#e.show()
print(e.describe,"\n")
print(e.goal,"\n")
lastaction="begin"
obs=e.describe
score=0
action_space=e.action_space
a="begin"

logs=[]

finish_len=len(e.oracle)
finish_rate=0

if(e.init_status):
    for i in range(20):
        action_space = e.action_space.copy()

        if(a in action_space):
            action_space.remove(a)
        as0=[]
        for i in action_space:
            if("examine " not in i and "look" not in i and "inventory" not in i):
                as0.append(i)
        action_space=as0


        score += e.score
        # print("\n", lastaction, "\n", obs)
        movement = cc.change(e.obs, a, score,action_space,memory_save=memory_save,fuc="function")
        score=score-1 if score>0 else 0
        obs = cc.pi.field2text()
        # cc.pi.show()
        arg = {
                "observation": obs,
                "action_space": action_space,
                "goal": e.goal,
                "memory": [i["content"] for i in  cc.activated_memory],
                "what_just_happened":["After "+cc.last_action_list[i]+", "+cc.re[i].field2text()+ ", you did the action: " + cc.last_action_list[i + 1] for i in  range(len(cc.re))],
                "d":cc.d
               }
        
        a=ga.get_action(arg,cc.pro,gpt_version=4, fuc = "prompt")

        print(e.oracle)
        print("Selected action: ",a)


        print(">>")

        if(a==e.oracle[0]):
            score+=1
        
        r=e.action(a)

        try:
            tag="trained" if trained else "untrained"
            logs = np.load('logs_'+tag+'_'+env_num+'.npy', allow_pickle='TRUE').item()["logs"]
        except:
            logs = []
            np.save('logs_'+tag+'_'+env_num+'.npy', {"logs": logs})

        logs.append({"args": arg, "pro": cc.pro, "action": a, "result_obs":r})
        np.save('logs_'+tag+'_'+env_num+'.npy', {"logs": logs})

        print("Observation: ",r)

        rate_tmp = (finish_len - len(e.oracle)) / finish_len
        if (rate_tmp > finish_rate):
            finish_rate = rate_tmp
        print("finish rate: ",finish_rate)

        if(e.score>=1):
            break

print(finish_rate)

#e.show()
e.stop()

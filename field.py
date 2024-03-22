import torch
import numpy as np
import json
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch 
import copy

import matplotlib as mpl 
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.mplot3d import proj3d 

vec={}
bert=None
#bert=Bert()
try:
    vec = np.load('vec.npy', allow_pickle='TRUE').item()
except:
    np.save('vec.npy', vec)
#print(vec)


class Arrow3D(FancyArrowPatch): 
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)
 
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

class Bert:
    def __init__(self):
        from transformers import BertTokenizer, BertModel
        self.tokenizer = BertTokenizer.from_pretrained('./bert')
        self.model = BertModel.from_pretrained('./bert')

    def get_vec(self,word):
        input_ids = torch.tensor(self.tokenizer.encode(word)).unsqueeze(0)
        outputs = self.model(input_ids)
        #print(outputs)
        sequence_output = (sum(outputs[0][0]).detach().numpy())/float(len(outputs[0][0]))
        x=sequence_output
        return x

class Monad:
    def __init__(self,name,wv,pv):
        self.obj_name=name
        self.wv = wv
        self.pv = pv #[r, st, fi], front is st=pi/2 & fi=0

class Field:
    def __init__(self,create="none",create_elements=None):
        self.front_distance=0.1
        self.front_angel=np.pi/6

        self.perception_range=np.pi*2/3

        self.front_r=np.sqrt(np.square(self.front_distance)*(np.square(np.tan(self.front_angel)+1)) )
        self.front_st=np.pi/2-self.front_angel

        self.monad=[]

        if(create=="circular_structure"):
            self.circular_structure(create_elements)
        elif (create == "dict_structure"):
            self.dict_structure(create_elements)
        elif(create=="copy"):
            self.monad = copy.deepcopy(create_elements.monad)

    def obj_movement(self,obj_name,p):
        n=self.get_obj(obj_name)
        self.monad[n].pv=p

    def copy(self):
        f=Field("copy",self)
        return f

    def add(self,data): #{name : pv}
        for name in data.keys():
            pv=data[name]
            if (pv == "front"):
                pv=[self.front_r, self.front_st, 0]

            mnd=Monad(name,vector(name),pv)
            self.monad.append(mnd)

    
    def circular_structure(self,obj_list):
        global vec
        n=len(obj_list)
        for i in range(len(obj_list)):
            mnd = Monad(obj_list[i], vector(obj_list[i]), [1,np.pi/2,2*np.pi/n*i])
            self.monad.append(mnd)

    def dict_structure(self,obj_list):
        global vec
        n=len(obj_list)

        for i in range(len(obj_list)):
            name = obj_list[i]["name"]
            mnd = Monad(name, vector(name), obj_list[i]["coordinate"])
            self.monad.append(mnd)

    def get_obj(self,obj): #name or word_vector
        n=-1
        max_sim=0
        if(str(type(obj))=="<class 'str'>"):
            for i in range(len(self.monad)):
                if(self.monad[i].obj_name==obj):
                    n=i
                    break
        else:
            for i in range(len(self.monad)):
                sim=cos_sim(self.monad[i].wv,obj)
                if(sim>max_sim):
                    n=i
                    max_sim=sim
                if(max_sim>=1):
                    break
        return n

    def spherical2rectangular(self,spherical_axes):
        rectangular_axes=[
                spherical_axes[0]*np.sin(spherical_axes[1])*np.cos(spherical_axes[2]),
                spherical_axes[0]*np.sin(spherical_axes[1])*np.sin(spherical_axes[2]),
                spherical_axes[0]*np.cos(spherical_axes[1])
                ]
        return rectangular_axes

    def rectangular2spherical(self,rectangular_axes):
        r=np.sqrt(np.power(rectangular_axes[0],2)+np.power(rectangular_axes[1],2)+np.power(rectangular_axes[2],2))
        y_x=rectangular_axes[1]/rectangular_axes[0]
        fi=np.arctan(y_x)
        if(rectangular_axes[0]<0):
            fi+=np.pi
        spherical_axes=[
                r,
                np.arccos(rectangular_axes[2]/r),
                fi
                ]
        spherical_axes=self.spherical_axes_standardization(spherical_axes)
        return spherical_axes

    def spherical_axes_standardization(self,spherical_axes):
        spherical_axes=spherical_axes.copy()
        if spherical_axes[1]<0:
            spherical_axes[1]=abs(spherical_axes[1])
            spherical_axes[2]=spherical_axes[2]+np.pi
        if spherical_axes[1]>np.pi:
            spherical_axes[1]=2*np.pi-spherical_axes[1]
            spherical_axes[2]=spherical_axes[2]+np.pi
        spherical_axes[2]=spherical_axes[2]+2*np.pi if spherical_axes[2]<0 else spherical_axes[2]
        spherical_axes[2]=spherical_axes[2]-2*np.pi if spherical_axes[2]>2*np.pi else spherical_axes[2]

        return spherical_axes
        

    def relocation_by_front_obj(self,obj):
        n=self.get_obj(obj)
        if(n==-1):
            return [0,0,0]
        # print(n)
        standard=self.monad[n].pv
        movement=[standard[0]-self.front_distance]+standard[1:]
        self.relocation(movement)
        return movement
        
    def relocation(self,standard):#standard=[r,st,pi]
        self.rotate(standard[1:])
        for i in range(len(self.monad)):
            spherical_axes=self.monad[i].pv
            if(spherical_axes[0]==0): #bring this thing
                continue

            rectangular_axes=self.spherical2rectangular(spherical_axes)
            
            rectangular_axes[0]-=standard[0]
            spherical_axes=self.rectangular2spherical(rectangular_axes)

            self.monad[i].pv=spherical_axes
    
    def rotate(self,standard):#standard=[st,pi]
        standard=[0]+standard
        for i in range(len(self.monad)):
            
            self.monad[i].pv[1]+=(np.pi/2-standard[1])
            self.monad[i].pv[2]-=standard[2]
            self.monad[i].pv=self.spherical_axes_standardization(self.monad[i].pv)

    def show(self,radius=2):
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['mathtext.default'] = 'regular'
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        point={}

        x=[]
        y=[]
        z=[]
        for p in range(len(self.monad)):
            a=self.spherical2rectangular(self.monad[p].pv)
            x.append(a[0])
            y.append(a[1])
            z.append(a[2])
            point[self.monad[p].obj_name]=a
        
        ax.plot(x, y, z, 'o', color='g', alpha=0.2)

        ax.scatter3D([0], [0], [0], c="red")
        
        for p in point.keys():
            ax.text(point[p][0], point[p][1], point[p][2], p, fontsize=9, fontdict=None)
        
        ax.set_title('Phenomenal Field')

        ax.set_xlim(-radius,radius)
        ax.set_ylim(-radius,radius)
        ax.set_zlim(-radius,radius)
        ax.set_xlabel('X', rotation=20)
        ax.set_ylabel('Y', rotation=-45)
        ax.set_zlabel('Z', rotation=0)

        arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0) 
 
        a = Arrow3D([0,0],[0,0],[-0.2,0.5],mutation_scale=10,  lw=1, arrowstyle="-|>", color="r",alpha=0.4) 
        ax.add_artist(a)
        a = Arrow3D([-0.2,0.5],[0,0],[0,0],mutation_scale=10,  lw=1, arrowstyle="-|>", color="r",alpha=0.8) 
        ax.add_artist(a)
        a = Arrow3D([-0.1,0.3],[-0.1,0.3],[0,0],mutation_scale=5,  lw=1, arrowstyle="-|>", color="b",alpha=0.4) 
        ax.add_artist(a)
        a = Arrow3D([-0.1,0.3],[0.1,-0.3],[0,0],mutation_scale=5,  lw=1, arrowstyle="-|>", color="b",alpha=0.4) 
        ax.add_artist(a) 

        plt.draw() 
        plt.show()

    def diffrence(self,f):
        wn=0.5
        wp=1-wn
        
        diff=0
        a=len(self.monad)
        b=len(f.monad)

        if(a==0 and b==0):
            return 0

        for i in range(a):
            _max=0
            threshold=0.9999
            choosen_num=-1
            for j in range(b):
                sim=cos_sim(self.monad[i].wv,f.monad[j].wv)
                if(sim>_max):
                    _max=sim
                    choosen_num=j
                if(_max>=threshold):
                    break
            if(choosen_num!=-1):
                n=_max
                p=spherical_sim(self.monad[i].pv,f.monad[choosen_num].pv)
            else:
                n = cos_sim(self.monad[i].wv,[1]*len(self.monad[i].wv))
                p = spherical_sim(self.monad[i].pv, [0]*len(self.monad[i].pv))
            diff+=(n*wn+p*wp)
        return diff/(max(a,b))

    def field2text(self,show=False):
        f=self.copy()
        text=""
        place="none"
        for i in range(len(f.monad)):
            if(show):
                print(f.monad[i].obj_name,f.monad[i].pv[0]-f.front_distance)
            if(f.monad[i].pv[0]<=f.front_distance+0.01 and f.monad[i].pv[0]!=0):
                place=f.monad[i].obj_name
                break
        if(place=="none"):
            text+="You are in the middle of a place. "
            n=len(f.monad)
            if(n==0):
                text += "Looking quickly around you, you see nothing."
                return text
            scan_gap=np.pi*2/n
            obj=[]
            take=[]
            for i in range(n):
                for j in range(len(f.monad)):

                    if (f.monad[j].pv[0]==0):
                        if(f.monad[j].obj_name not in take):
                            take.append(f.monad[j].obj_name)
                        else:
                            continue
                    elif(f.monad[j].pv[2]>=scan_gap*i and f.monad[j].pv[2]<scan_gap*(i+1)):
                        obj.append(f.monad[j].obj_name)
            text+="Looking quickly around you, you see "
            text+=", ".join(["a "+i for i in obj[:-1]])
            if(len(obj)>1):
                text+=", and a "+obj[-1]+"."
            else:
                if(len(obj)==1):
                    text+="a "+obj[-1]+"."
                else:
                    text+="nothing."
            if(len(take)!=0):
                text+=" You are holding "
                text += ", ".join(["a " + i for i in take[:-1]])
                if (len(take) > 1):
                    text += ", and a " + take[-1] + "."
                else:
                    text += "a " + take[-1] + "."
            else:
                text += " You are holding nothing in your hands."
        else:
            text+="On the "+place+", you see "
            obj=[]
            take = []
            for j in range(len(f.monad)):
                if (f.monad[j].pv[0] == 0):
                    if (f.monad[j].obj_name not in take):
                        take.append(f.monad[j].obj_name)
                    else:
                        continue
                if(f.monad[j].pv[2]>=-f.perception_range/2 and f.monad[j].pv[2]<f.perception_range/2 and f.monad[j].obj_name!=place):
                    obj.append(f.monad[j].obj_name)
            text+=", ".join(["a "+i for i in obj[:-1]])
            if(len(obj)>1):
                text+=", and a "+obj[-1]+"."
            else:
                if(len(obj)==1):
                    text+="a "+obj[-1]+"."
                else:
                    text+="nothing."
            if (len(take) != 0):
                text += " You are holding "
                text += ", ".join(["a " + i for i in take[:-1]])
                if (len(take) > 1):
                    text += ", and a " + take[-1] + "."
                else:
                    text += "a " + take[-1] + "."
            else:
                text += " You are holding nothing in your hands."
        return text

    def update(self,p):
        for i in range(len(p.monad)):
            searched=False
            for j in range(len(self.monad)):
                if(p.monad[i].obj_name==self.monad[j].obj_name):
                    self.monad[j].pv=p.monad[i].pv
                    searched = True
                    break
            if(not searched):
                self.monad.append(p.monad[i])

def spherical_sim(a, b):
    a=[a[0],a[1]/np.pi,a[2]/(2*np.pi)]
    b=[b[0],b[1]/np.pi,b[2]/(2*np.pi)]

    w=[3/7,1/7,3/7]
    sim=np.tanh(abs(a[0]-b[0]))*w[0]+abs(a[1]-b[1])*w[1]+abs(a[2]-b[2])*w[2]
    sim=1-sim/3
    return sim

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos

def vector(x):
    global vec,bert

    try:
        return vec[x]
    except:
        if(bert==None):
            bert=Bert()
        vec[x]=bert.get_vec(x)

        np.save('vec.npy', vec)
        return vec[x]
        


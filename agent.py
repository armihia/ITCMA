import requests
import json
import random
import re

from large_language_model import LLM

class Generative_Agent:
    def __init__(self):
        self.llm=LLM()

    def get_action_from_asw(self,action_space,asw):
        asw=asw.replace(",","").replace("** ","**").replace("the ","")
        result = ""
        try:
            result = re.compile(r"\*\*([A-Za-z0-9 ,\.])+\*\*").search(asw).group(0).replace(".", "").replace(",", "")
        except:
            try:
                result = re.compile(r"\*\*([A-Za-z0-9 ,\.])+\.").search(asw).group(0).replace(".","").replace(",","")
            except:
                pass

        if (result in action_space):
            return result

        result=""
        try:
            result = re.compile(r"\"([A-Za-z0-9 ,\.])+\"").search(asw).group(0).replace(".","").replace(",","")
        except:
            try:
                result = re.compile(r"\'([A-Za-z0-9 ,\.])+\'").search(asw).group(0).replace(".","").replace(",","")
            except:
                pass
        if(result in action_space):
            return result

        action = {}
        for act in action_space:
            if (act in asw):
                try:
                    action[act] += 1
                except:
                    action[act] = 1

        # print(action)

        try:
            max_n=max(list(action.values()))
            max_list = []
            for a in action:
                if (action[a] == max_n):
                    max_list.append(a)

            max_index = 1000000
            max_action = "look"
            for ml in max_list:
                if (asw.index(ml) < max_index):
                    max_index = asw.index(ml)
                    max_action = ml

            result = max_action
        except Exception as e:
            print("Exception: ",e)
            result = "look"


        return result

    def get_action(self,arg,pro=None, gpt_version=4, fuc = "prompt"):
        qst="In the current situation, combined with reference information, what action do you think is the most suitable for execution?\n" + \
        "Please note that the objects directly mentioned in the goal may not necessarily be the correct action objects. Please pay attention to the reference information and possible scenarios that may arise after the action, and make the correct judgment.\n" + \
        "Please break down the actions and try to think step by step."
        print("action generating")
        if (fuc == "prompt"):
            asw0, jdg = self.llm.prompt_chat(qst, arg, use_stream=False, pro=pro, gpt_version=gpt_version)
            if (not jdg):
                asw = self.llm.function_chat(qst, arg, use_stream=False, pro=pro)
                print("function:", asw, "\n", "simple:", asw0, "\n")
            else:
                asw = asw0
        else:
            try:
                asw = self.llm.function_chat(qst, arg, use_stream=False, pro=pro)
                print("function:", asw)
            except:
                pass
        #print(asw)

        result=self.get_action_from_asw(arg["action_space"], asw)
        if(result=="look"):
            result = self.get_action_from_asw(arg["action_space"], asw0)

        print("asw: ",asw)

        return result
    

import requests
import json
import random
from openai import OpenAI

class LLM:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000"

        self.chatgpt_url = "" #chatgpt_url
        self.gpt4_url = "" #gpt4_url 

        self.chatgpt_key = "" #chatgpt_key
        self.gpt4_key = "" #gpt4_key

    def create_chat_completion(self, model, messages, functions, use_stream=False):
        content0 = ""
        data = {
            "functions": functions,
            "model": model,  # model name
            "messages": messages,  # session history
            "stream": use_stream,
            "max_tokens": 100,
            "temperature": 0.2,
            "top_p": 0.8,
        }

        response = requests.post(f"{self.base_url}/v1/chat/completions", json=data, stream=use_stream)
        if response.status_code == 200:
            if use_stream:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')[6:]
                        try:
                            response_json = json.loads(decoded_line)
                            content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            content0 += content
                        except:
                            pass
            else:
                decoded_line = response.json()
                content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
                content0 = content
        else:
            print("Error:", response.status_code)
            return None
        return content0

    def tool_function(self, f_name, arguments,pro=None):

        function_list=["getAction","getProtention"]
        if (f_name not in function_list):
            result = {
                "role": "function",
                "description": "error",
                "name": f_name,
                "content": '{"error": "error"}',
            }
            return {}, result

        if(f_name=="getAction"):
            functions = {
                "name": "getAction",
                "description": "Get what to do later.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "what_just_happened": {
                            "type": "string",
                            "description": "What just happened and the situation just observed.",
                        },
                        "memory": {
                            "type": "list",
                            "description": "Memory for decision-making on how to act.",
                        },
                        "observation": {
                            "type": "string",
                            "description": "What is currently being observed.",
                        },
                        "action_space": {
                            "type": "list",
                            "description": "Actions allowed to be taken.",
                        },
                        "goal": {
                            "type": "string",
                            "description": "What must be done.",
                        },
                        "d": {
                            "type": "list",
                            "description": "Your emotions. They are pleasure level, arousal level, and dominance level.",
                        },
                    },
                    "required": ["observation", "action_space", "goal"],
                },
            }

            if(pro==None):
                r = self.get_possible_result(arguments)
            else:
                r=pro

        elif (f_name == "getProtention"):
            functions = {
                "name": "getProtention",
                "description": "Predicting how the scene will change after the action.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scene_that_have_occurred": {
                            "type": "list",
                            "description": "Scenes that have occurred.",
                        },
                        "ongoing _scene": {
                            "type": "string",
                            "description": "The current scene that is happening.",
                        },
                        "memory": {
                            "type": "list",
                            "description": "Scenes related to the current scene and their follow-up.",
                        },

                        "action": {
                            "type": "string",
                            "description": "Actions to be taken.",
                        },
                    },
                    "required": ["scene_that_have_occurred","ongoing _scene","memory","action"],
                },
            }
            r = arguments

        result = {
            "role": "function",
            "name": functions["name"],
            "content": str(r),
        }
        return functions, result

    def function_chat(self, qst, arguments, use_stream=True,f_name = "getAction",pro="None"):

        functions, result = self.tool_function(f_name, arguments,pro)

        a0 = []
        for i in arguments.keys():
            a0.append(i + "='" + str(arguments[i]) + "'")
        a0 = " , ".join(a0)

        chat_messages = [
            {
                "role": "user",
                "content": qst,
            },
            {
                "role": "assistant",
                "content": f_name + "\n ```python\ntool_call(" + a0 + ")\n```",
                "function_call": {
                    "name": f_name,
                },
            },
            result
        ]
        a = self.create_chat_completion("chatglm3-6b", messages=chat_messages, functions=[functions],
                                        use_stream=use_stream)

        return a

    def prompt_chat(self,qst, arguments, use_stream=True,f_name = "getAction",pro="None",gpt_version=3):
        jdg=True
        functions, result = self.tool_function(f_name, arguments, pro)

        if(f_name=="getAction"):
            msg=["Your current emotion is: "+str(arguments["d"])+" (They are pleasure level, arousal level, and dominance level)",

             "Here are some reference information, which comes from similar tasks you have performed before:\n"+ \
             str(arguments["memory"]).replace("'[", "'[\n").replace("',", "',\n"),

             "What just happened (in your memory) is: "+str(arguments["what_just_happened"]).replace("['", "[\n'").replace("',", "',\n"),

             "Here are the things that may happen if you take corresponding actions:\n"+ \
             result["content"].replace("'[", "'[\n").replace("',", "',\n").replace('",', '",\n'),

             "You are an excellent behavior planner who can extract effective information from the above that helps achieve goals in the current situation, and gradually decompose and plan actions.\n"+ \
             str(arguments["observation"])+" And you want to " + str(arguments["goal"]).replace("Your task is to: ","")+"\n"+\
             qst
             ]
            msg = "\n\n".join(msg)

            with open("prompt.txt", "w", encoding="utf-8") as f:
                f.write(msg)

        else:
            msg = [qst,
                   "scene_that_have_occurred=" + str(arguments["scene_that_have_occurred"]),
                   "ongoing_scene=" + str(arguments["ongoing_scene"]),
                   "action=" + str(arguments["action"]),
                   "memory=" + str(arguments["memory"]).replace("'[", "'[\n").replace("',", "',\n"),
                   ]
            msg = "\n\n".join(msg)

            with open("prompt_pro.txt", "w", encoding="utf-8") as f:
                f.write(msg)

        try:
            if(gpt_version==3):
                a = self.chatgpt_chat(msg)
            else:
                try:
                    a = self.gpt4_chat(msg)
                except Exception as e:
                    print("gpt4 error: ", e)
                    a = self.chatgpt_chat(msg)
        except Exception as e:
            jdg=False
            print("chatgpt error: ",e)
            a = self.simple_chat(msg)

        return a,jdg

    def simple_chat(self, qst, use_stream=True):
        functions = None
        sys_context = ""

        chat_messages = [
            {
                "role": "system",
                "content": sys_context,
            },
            {
                "role": "user",
                "content": qst
            }
        ]
        a = self.create_chat_completion("chatglm3-6b", messages=chat_messages, functions=functions,
                                        use_stream=use_stream)
        return a

    def get_possible_result(self,arguments):
        acts=arguments["action_space"]
        pro={}
        for a in acts:
            pro[a]="There's no telling what's going to happen."
        pro[list(pro.keys())[0]]="The goal will be achieved"
        r={"Possible_changes_in_the_scene_after_the_action":pro}
        return r

    def chatgpt_chat(self,qst, his=[]):
        msg = []
        for i in his:
            msg.append({"role": "user", "content": i[0]})
            msg.append({"role": "assistant", "content": i[1]})
        msg.append({"role": "user", "content": qst})


        client = OpenAI(
            base_url=self.chatgpt_url,
            api_key=self.chatgpt_key
        )

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=msg
        )

        response=completion
        r=response.choices[0].message.content

        return r.lower()

    def gpt4_chat(self,qst, his=[]):
        msg = []
        for i in his:
            msg.append({"role": "user", "content": i[0]})
            msg.append({"role": "assistant", "content": i[1]})
        msg.append({"role": "user", "content": qst})


        client = OpenAI(
            base_url=self.gpt4_url,
            api_key=self.gpt4_key
        )

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=msg
        )

        response=completion
        r=response.choices[0].message.content

        return r.lower()
    

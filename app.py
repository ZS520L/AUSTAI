import numpy as np
import cv2
import urllib.request
import openai
import gradio as gr
import random


user_contexts = {}

def get_assistant_response(user_question, context):
    context.append({"role": "user", "content": user_question+"Let's think step by step"})
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=context,
        temperature=0
    )
    assistant_response = response.choices[0].message['content']
    context.append({"role": "assistant", "content": assistant_response})
    return assistant_response

def generate_image_url(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1,  # 生成1张图片
        size="512x512",  # 图像大小
    )
    image_url = response["data"][0]["url"]
    return image_url

def greet(user_id, api_key, user_question, clear_history):
    openai.api_key = api_key
    global user_contexts
    if user_id not in user_contexts:
        user_contexts[user_id] = [
            {"role": "system", "content": "你是一个聪明的AI助手。"},
            {"role": "user", "content": "你会说中文吗？"},
            {"role": "assistant", "content": "是的，我可以说中文。"}
        ]

    context = user_contexts[user_id]

    if clear_history:
        context = [
            {"role": "system", "content": "你是一个聪明的AI助手。"},
            {"role": "user", "content": "你会说中文吗？"},
            {"role": "assistant", "content": "是的，我可以说中文。"}
        ]
        user_contexts[user_id] = context
        return '清空成功', '保持聊天记录', np.ones((5,5))
    else:
        # 如果user提问包含生成图像的特定指令（这里我们使用“生成图片：”作为示例）
        if user_question.startswith("生成图片：") or user_question.startswith("生成图片:"):
            image_prompt = user_question[5:]  # 提取用于生成图片的文本
            image_url = generate_image_url(image_prompt)
            resp = urllib.request.urlopen(image_url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # return image
            return '', '图片已生成', image
        get_assistant_response(user_question, context)
        prompt = ""

        for item in context[3:]:
            prompt += item["role"] + ": " + item["content"] + "\n"
        return '', prompt, np.ones((5,5))

demo = gr.Interface(
    fn=greet,
    inputs=[
        gr.Textbox(lines=1, label='请输入用户ID', placeholder='请输入用户ID'),
        gr.Textbox(lines=1, label='请输入你的OpenAI API密钥', placeholder='请输入你的OpenAI API密钥'),
        gr.Textbox(lines=15, label='请输入问题', placeholder='请输入您的问题'),
        gr.Checkbox(label='清空聊天记录', default=False)
    ],
    outputs=[
        gr.Textbox(lines=1, label='聊天记录状态', placeholder='等待清空聊天记录'),
        gr.Textbox(lines=20, label='AI回答', placeholder='等待AI回答'),
        gr.Image(label='等待图片生成')
    ],
    title="AI助手",
    description="""
1.使用说明：
请输入您的问题，AI助手会给出回答。
支持连续对话，可以记录对话历史。
重新开始对话勾选清空聊天记录，输出清空成功表示重新开启对话。
2.特别警告：
为了防止用户数据混乱，请自定义用户ID。
理论上如果被别人知道自己的ID，那么别人可以查看自己的历史对话，对此你可以选择在对话结束后清除对话记录。
3.图片生成示例：格式-【生成图片：xxxxxxxx】
生成图片：春天到了，万物复苏
    """
)

if __name__ == "__main__":
    demo.launch()
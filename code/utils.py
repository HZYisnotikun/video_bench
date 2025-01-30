from PIL import Image
import torch
import os
from torchvision import io
from torchvision.transforms import Resize
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import json
from tqdm import tqdm
import numpy as np
import pickle
import gzip

def lablize_jsonl(file_path, lable="qry", lable_name="qry_id"):
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 处理每一行
    updated_lines = []
    for idx, line in tqdm(enumerate(lines, start=1), desc="Processing", unit="lines"):
        # 解析 JSON 对象
        json_obj = json.loads(line.strip())
        # 添加新的字段
        json_obj[lable_name] = f"{lable}_{idx}"
        # 将修改后的 JSON 对象转换为字符串
        updated_lines.append(json.dumps(json_obj, ensure_ascii=False))

    # 覆盖原文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(updated_lines))

def write_big_pkl(file_path, batch_size, content):
    '''
        Input:
          content: 要写入的list
          file_path: pickle文件的路径
          batch_size: 一次写入的元素数
        Output:
          none
    '''
    # 打开文件，以追加模式写入
    with gzip.open(file_path, 'ab') as file:
        # 遍历 content，每次处理 batch_size 个元素
        for i in range(0, len(content), batch_size):
            batch = content[i:i + batch_size]
            pickle.dump(batch, file)
            print(f"Batch {i // batch_size + 1} written to {file_path}")
    print(f"finish writing into {file_path}")

def split_jsonl(file_path, split_size=1000, target_path="./"):
    '''
        Input:
          file_path: jsonl文件的路径
          split_size: 每个子json文件的json对象数
          target_path: 存储子文件的目标路径
    '''
    if not os.path.exists(target_path):
        os.makedirs(target_path)  # 如果目标路径不存在，创建目录

    # 获取源文件名
    base_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(base_name)

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    total_lines = len(lines)
    for i in range(0, total_lines, split_size):
        # 确定当前批次的起始和结束索引
        start = i
        end = min(i + split_size, total_lines)

        # 提取当前批次的行
        batch_lines = lines[start:end]

        # 生成子文件名
        sub_file_name = f"{file_name}_{i // split_size + 1}{file_extension}"
        sub_file_path = os.path.join(target_path, sub_file_name)

        # 写入子文件
        with open(sub_file_path, 'w', encoding='utf-8') as sub_file:
            sub_file.writelines(batch_lines)

        print(f"Written {len(batch_lines)} lines to {sub_file_path}")

def load_qwne2vl_model_and_processor(device, qwen2vl_path):
    model = Qwen2VLForConditionalGeneration.from_pretrained(qwen2vl_path, device_map="balanced_low_0")   
    processor = AutoProcessor.from_pretrained(qwen2vl_path)
    return model, processor

def load_images_from_path(video_id, base_path):
    images = []
    for i in range(1, 9):  # 遍历1到8
        image_name = f"{video_id}_{i}.jpg"  # 构造图片文件名
        image_path = os.path.join(base_path, image_name)  # 构造完整路径
        if os.path.exists(image_path):  # 检查文件是否存在
            image = Image.open(image_path)  # 加载图片
            images.append(image)  # 添加到列表
        else:
            raise FileNotFoundError(f"图片 {image_path} 不存在")
    return images

# Video
def fetch_video(ele: Dict, nframe_factor=2, target_height=240, target_width=320):
    if isinstance(ele['video'], str):
        def round_by_factor(number: int, factor: int) -> int:
            return round(number / factor) * factor

        video = ele["video"]
        if video.startswith("file://"):
            video = video[7:]

        video, _, info = io.read_video(
            video,
            start_pts=ele.get("video_start", 0.0),
            end_pts=ele.get("video_end", None),
            pts_unit="sec",
            output_format="TCHW",
        )
        assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
        if "nframes" in ele:
            nframes = round_by_factor(ele["nframes"], nframe_factor)
        else:
            fps = ele.get("fps", 1.0)
            nframes = round_by_factor(video.size(0) / info["video_fps"] * fps, nframe_factor)
        nframes=min(nframes,20)
        idx = torch.linspace(0, video.size(0) - 1, nframes, dtype=torch.int64)
        video = video[idx]

        # 等比例压缩
        original_height, original_width = video.size(2), video.size(3)
        aspect_ratio = original_width / original_height
        if target_height / target_width > aspect_ratio:
            # 保持宽度不变，调整高度
            new_height = int(target_width / aspect_ratio)
            new_width = target_width
        else:
            # 保持高度不变，调整宽度
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        # 使用 torchvision.transforms.Resize 进行等比例压缩
        resize_transform = Resize((new_height, new_width))
        video = resize_transform(video)
        
        return video

def batch_VidQA_to_embedding(conversations, video_list, model, processor, device, batch_size, temp_path):
    """
        embed video and its QAs to vectors using Qwen2VL
        """
    # print(len(conversations))
    embeddings = []  # List to store image embeddings
    # with open(temp_path, 'rb') as file:
    #     conversations = pickle.load(file)
    #     video_list = pickle.load(file)   //如果意外中断 从中间结果文件load预处理内容
    print("conv shape: ", np.array(conversations).shape)

    for i in tqdm(range(0, len(conversations), batch_size), desc="Extracting feature"):
        prompts = []
        batch_convs = conversations[i:i+batch_size]
        for conv in batch_convs:
            prompts.append(processor.apply_chat_template(conv,add_generation_prompt=False))
        print("prompts: ", prompts)
        if video_list!=[]:
            batch_video = video_list[i:i+batch_size]
            inputs = processor(videos=batch_video, text=prompts, padding=True, return_tensors="pt").to(model.device, torch.float16)
        else:
            inputs = processor(text=prompts, padding=True, return_tensors="pt").to(model.device, torch.float16)
        inputs['output_hidden_states'] = True #输出隐藏状态
        print("input token length: ", inputs["input_ids"].shape)

        # 提取多模态特征
        with torch.no_grad():
            outputs = model(**inputs)
            multimodal_features = outputs.hidden_states[-1]  # 提取最后一层的隐藏状态作为特征
        one_token_feature=multimodal_features[:,-1,:]
        embeddings = embeddings + one_token_feature.cpu().tolist()

    # 打印结果
    print("seq len: ", len(embeddings))
    print("dim size: ", len(embeddings[0]))

    return embeddings
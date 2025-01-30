from utils import batch_VidQA_to_embedding, fetch_video, load_qwne2vl_model_and_processor
import numpy as np

qwen_model, qwen_processor = load_qwne2vl_model_and_processor('cuda', '/global_data/sft_intern/lh/Qwen2-VL-7B-Instruct')

conversations_1=[
    [{
        "role": "user",
        "content": [
            {"type": "text", "text": "find a video along with its QA pair that match this sentence: "},
            {"type": "text", "text": "a man is opening a box up"},
        ]
    }],
    [{
        "role": "user",
        "content": [
            {"type": "text", "text": "find a video along with its QA pair that match this sentence: "},
            {"type": "text", "text": "two boys performing karate moves"},   
        ]
    }],
    [{
        "role": "user",
        "content": [
            {"type": "text", "text": "find a video along with its QA pair that match this sentence: "},
            {"type": "text", "text": "a man walking through wilderness"},
        ]
    }],
]

qry_embeddings=batch_VidQA_to_embedding(
    conversations=conversations_1,
    video_list=[],
    model=qwen_model,
    processor=qwen_processor,
    device='cuda',
    batch_size=1,
    temp_path=''
)

# print(f"qry seq len: {len(qry_embeddings)}")
# print(f"qry dim: {len(qry_embeddings[0])}")

conversations_2 = [
    [{
        "role": "user",
        "content": [
            {"type": "text", "text": "here is a video and its QA pair: "},
            {"type": "text", "text": "Question:what contains someone opens the lid of a corrugated cardboard?Answer:rifle"},
            {"type": "video"} #jfrrO5K_vKM_55_65
        ]
    }],
    [{
        "role": "user",
        "content": [
            {"type": "text", "text": "here is a video and its QA pair: "},
            {"type": "text", "text": "Question:how many men are giving a synchronized martial art performance in front of a crowd at a fun fair?Answer:two"},
            {"type": "video"} #jjl2ZMdFCsw_17_35
        ]
    }],
    [{
        "role": "user",
        "content": [
            {"type": "text", "text": "here is a video and its QA pair: "},
            {"type": "text", "text": "Question:who is walking down a pathway lined with greenery?Answer:man"},
            {"type": "video"} #jlahRlo4jlU_30_36
        ]
    }]
]

video_list=[]
video_infos = [
    {"type": "video", "video": f"/data_train/code/sft_intern/lh/hzy/vid_bench/MSVD_Zero_Shot_QA/videos/{video_id}.mp4", "fps": 1.0}
    for video_id in ['jfrrO5K_vKM_55_65', 'jjl2ZMdFCsw_17_35', 'jlahRlo4jlU_30_36']
]
for video_info in video_infos:
    video_list.append(fetch_video(video_info))

cand_embeddings=batch_VidQA_to_embedding(
    conversations=conversations_2,
    video_list=[],
    model=qwen_model,
    processor=qwen_processor,
    device='cuda',
    batch_size=1,
    temp_path=''
)

qry_embeddings=np.array(qry_embeddings)
cand_embeddings=np.array(cand_embeddings)

qry_norms = np.linalg.norm(qry_embeddings, axis=1, keepdims=True)
qry_embeddings_normalized = qry_embeddings / qry_norms  

cand_norms = np.linalg.norm(cand_embeddings, axis=1, keepdims=True)  
cand_embeddings_normalized = cand_embeddings / cand_norms  # 归一化

# 计算点积
scores = qry_embeddings_normalized @ cand_embeddings_normalized.T

print(scores)

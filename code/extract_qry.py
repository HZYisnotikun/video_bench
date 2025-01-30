# import chromadb
import json
from tqdm import tqdm
import pickle
import gzip
from utils import load_qwne2vl_model_and_processor, fetch_video, batch_VidQA_to_embedding, write_big_pkl
from vecdb import _save_embeddings_to_db
# import signal

# signal.signal(signal.SIGINT, signal.SIG_IGN)
# signal.signal(signal.SIGTSTP, signal.SIG_IGN)
# Load the model in half-precision
# Load the model in half-precision on the available device(s)

# # Get three different images
# url = "https://www.ilankelman.org/stopsigns/australia.jpg"
# image_stop = Image.open(requests.get(url, stream=True).raw)
def process_json(file_path, temp_path):
    # 打开 JSONL 文件
    with open(file_path["json_path"], 'r', encoding='utf-8') as file:
        # 获取文件总行数（用于 tqdm 的总进度）
        total_lines = sum(1 for _ in file)
        file.seek(0)  # 重置文件指针到开头
        captions=[]
        qry_ids=[]
        conversations=[]
        # batch_size=1
        # 使用 tqdm 包装循环
        for line in tqdm(file, total=total_lines, desc="Processing JSONL"):        
            # 解析每一行的JSON对象
            data = json.loads(line.strip())
            
            # question = data.get('question', 'Unknown')
            # answer = data.get('answer', 'Unknown')
            # video_id = data.get('video_id', 'Unknown')
            # qa_ids.append(data.get('qa_id', 'Unknown'))  #11111111111111111111注意换成qa_id
            # video_path=file_path["vid_base_path"]+video_id+file_path["suffix"]
            # video_paths.append(video_path)
            # qa_content="Question:"+question+"Answer:"+answer
            # qa_contents.append(qa_content)
            caption=data.get('caption','Unknown')
            captions.append(caption)
            qry_id=data.get('qry_id', 'Unknown')
            qry_ids.append(qry_id)
            conversation = [
                {
                    "role": "caption",
                    "content": [
                        {"type": "text", "text": caption},
                    ]
                }
            ]
            conversations.append(conversation)
            # video_info = {"type": "video", "video": video_path, "fps": 1.0}
            # # print(video_.shape)
            # video_list.append(fetch_video(video_info))
    print("processing jsonl file finished")
    # for item in [conversation,video_list,qa_ids]:
    #     write_big_pkl(temp_path,batch_size,item)
    # print("temp result written in ", temp_path)
    return conversations, qry_ids, captions

def extract_features(qwen_path, dataset_path, output_file, device, batch_size, split, vecdb_path, vecdb_name, temp_path=''):
    qwen_model, qwen_processor = load_qwne2vl_model_and_processor(device, qwen_path)
    dataset_base_path = dataset_path["json_path"]
    for i in range(1,2):
        dataset_path["json_path"] = dataset_base_path+".jsonl"
        conversations, qry_ids, captions = process_json(dataset_path,temp_path)
        embeddings = batch_VidQA_to_embedding(conversations=conversations,
                                            video_list=[],
                                            model=qwen_model,
                                            processor=qwen_processor,
                                            device=device,
                                            batch_size=batch_size,
                                            temp_path=temp_path)
        metadata=[{"caption":caption} for caption in captions]
        ids=qry_ids
        # metadata = [{"video_paths": video_path, "qa_content": qa_content} for video_path, qa_content in zip(video_paths, qa_contents)]
        _save_embeddings_to_db(vecdb_path, vecdb_name, embeddings, metadata, ids)
        print(f"split {i} embeddings stored in {vecdb_name}. Path:{vecdb_path}")
        # 写不下
        # with gzip.open(output_file["embeddings"], 'ab') as file:
        #     pickle.dump(embeddings,file)
        #     print(f"Embeddings is saved at {output_file["embeddings"]}")
        # with gzip.open(output_file["qa_ids"], 'ab') as file:
        #     pickle.dump(qa_ids,file)
        #     print(f"qa_ids is saved at {output_file["qa_ids"]}")


# client=chromadb.PersistentClient(path="./chroma/")
# MSRVTT_C2QA_cand = client.get_collection(name="MSRVTT_C2QA_cand")


#     # MSRVTT_C2QA_cand.add(
#     #     embeddings = one_token_feat,
#     #     ids = qa_ids
#     # )
#     # print("all items stored in chroma")

if __name__ == "__main__":
    extract_features(qwen_path="/global_data/sft_intern/lh/Qwen2-VL-7B-Instruct",
                     dataset_path={
                        "json_path": "/data_train/code/sft_intern/lh/hzy/vid_bench/TGIF_Zero_Shot_QA/test_qry", #no suffix
                        # "vid_base_path": "/data_train/code/sft_intern/lh/hzy/vid_bench/MSRVTT_Zero_Shot_QA/videos/all/",
                        # "suffix": ".mp4"
                     },
                     device="cuda",
                     batch_size=10,
                    #  temp_path="../temp/test.pkl",
                     split=1,
                     vecdb_path='../chroma_store',
                     vecdb_name='tgif_c2qa_test_qry',
                     output_file='')


# # import torch
# # print("可用的GPU数量:", torch.cuda.device_count())
# # print("当前默认GPU:", torch.cuda.current_device())
# # for i in range(torch.cuda.device_count()):
# #     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# # import pickle

# # with open('../MSVD_Zero_Shot_QA/caption2QA/msvd_c2qa_cand.pkl', 'rb') as file:
# #     loaded_list1 = pickle.load(file)
# #     loaded_list2 = pickle.load(file)

# # print(loaded_list1)
# # print(loaded_list2)

# import pickle
# import gzip

# def read_big_pkl(file_path):
#     '''
#         Input:
#           file_path: pickle文件的路径
#         Output:
#           一个包含所有写入的列表的列表
#     '''
#     result = []
#     with gzip.open(file_path, 'rb') as file:
#         while True:
#             try:
#                 batch = pickle.load(file)
#                 result.append(batch)
#             except EOFError:
#                 break
#     return result

# # 示例用法
# if __name__ == "__main__":
#     file_path = "test.pkl"  # 临时文件路径

#     # 读取文件
#     lists = read_big_pkl(file_path)

#     # 打印读取的列表
#     for i, lst in enumerate(lists):
#         print(f"List {i+1}: {lst[:10]}... (length: {len(lst)})")

# # import pickle
# # with gzip.open('test.pkl', 'wb') as f:
# # 	pickle.dump([1,2,3],f)
# # 	pickle.dump([4,5,6],f)
# # 	pickle.dump([7,8],f)
# # 	pickle.dump([1],f)
# # print(len(a))
# from utils import split_jsonl

# split_jsonl("/data_train/code/sft_intern/lh/hzy/vid_bench/TGIF_Zero_Shot_QA/mergedQA.jsonl",5000,"/data_train/code/sft_intern/lh/hzy/vid_bench/TGIF_Zero_Shot_QA/split_test/")
# import chromadb
# 
# client=chromadb.PersistentClient(path="../chroma_store")
# collection=client.get_collection("tgif_c2qa_test_cand")
# client.delete_collection(name="aaa")
# collection.modify(name="msvd_c2qa_test_cand")
# print(client.list_collections())
# print("embeddings num: ",len(aaa["embeddings"]))
# print("embeddings dim: ", len(aaa["embeddings"][0]))
# print("collection record num: ", collection.count())
# print(collection.peek())
# ll=collection.get(
#     ids="qry_1"
# )
# print(ll)
from utils import lablize_jsonl
lablize_jsonl("/data_train/code/sft_intern/lh/hzy/vid_bench/TGIF_Zero_Shot_QA/test_qry.jsonl")

# import json

# def json_to_jsonl(json_file_path, jsonl_file_path):
#     """
#     将 JSON 文件（包含一个 JSON 列表）转换为 JSONL 文件
#     :param json_file_path: 输入的 JSON 文件路径
#     :param jsonl_file_path: 输出的 JSONL 文件路径
#     """
#     # 读取 JSON 文件
#     with open(json_file_path, 'r', encoding='utf-8') as json_file:
#         data = json.load(json_file)  # 加载 JSON 数据

#     # 将 JSON 列表写入 JSONL 文件
#     with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
#         for item in data:
#             # 将每个 JSON 对象转换为字符串并写入文件
#             jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')

# # 示例调用
# json_to_jsonl('/data_train/code/sft_intern/lh/hzy/vid_bench/TGIF_Zero_Shot_QA/test_qry.json', '/data_train/code/sft_intern/lh/hzy/vid_bench/TGIF_Zero_Shot_QA/test_qry.jsonl')
import chromadb
import json

def eval(vecdb_path,vecdb_cand,vecdb_qry,recall,qry_file_path):
    client=chromadb.PersistentClient(path=vecdb_path)
    cand_collection=client.get_collection(name=vecdb_cand)
    qry_collection=client.get_collection(name=vecdb_qry)

    qry_id_list=[f"qry_{i}" for i in range(200,206)]
    qry_list=qry_collection.get(ids=qry_id_list,include=['embeddings'])
    embeddings=qry_list['embeddings']
    cand_list=cand_collection.query(query_embeddings=embeddings,n_results=recall)

    pos_cands=[]
    recall_rates=[]

    for qry_id in qry_id_list:
        with open(qry_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_object = json.loads(line)
                if json_object['qry_id'] == qry_id:
                    pos_cands.append(json_object['pos_cand'])
                    break
    print(pos_cands[0])
    print(cand_list['ids'])

    for i, id_list in enumerate(cand_list['ids']):    
        n_pos=0
        for id in id_list:
            if id in pos_cands[i]:
                n_pos += 1
        print(f"n_pos: {n_pos}")
        recall_rate=n_pos/len(pos_cands[i])
        print(f"recall rate: {recall_rate}")  
        recall_rates.append(recall_rate)
    print(f"average recall rate: {sum(recall_rates)/len(recall_rates)}")

if __name__ == '__main__':
    eval(
        vecdb_path='../chroma_store',
        vecdb_cand='tgif_c2qa_test_cand',
        vecdb_qry='tgif_c2qa_test_qry',
        recall=10,
        qry_file_path='/data_train/code/sft_intern/lh/hzy/vid_bench/TGIF_Zero_Shot_QA/test_qry.jsonl'
    )
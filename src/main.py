import re
from llm_api import api_call
from event_log_preprocessing import *
from rag_config import *
from vector_api import *
from pathlib import Path

def main():
    datasets_xes = [
        "/home/apost/projects/Diplomatiki/data/xes_logs/BPI_Challenge_2012.xes",
        "/home/apost/projects/Diplomatiki/data/xes_logs/BPI_Challenge_2013_ClosedProblems.xes",
        "/home/apost/projects/Diplomatiki/data/xes_logs/BPI_Challenge_2013_Incidents.xes",
        "/home/apost/projects/Diplomatiki/data/xes_logs/BPI_Challenge_2019.xes",
        "/home/apost/projects/Diplomatiki/data/xes_logs/BPI_Challenge_2020_InternationalDeclarations.xes",
        "/home/apost/projects/Diplomatiki/data/xes_logs/Sepsis.xes",
        "/home/apost/projects/Diplomatiki/data/xes_logs/HospitalBilling.xes",
        "/home/apost/projects/Diplomatiki/data/xes_logs/BPI_Challenge_2017.xes"
  ]
    collections = [
        # "sentence-transformers_all-MiniLM-L12-v2_b1_g3_BPI_Challenge_2020_InternationalDeclarations",
        "sentence-transformers_all-MiniLM-L12-v2_b1_g3_BPI_Challenge_2017",
        # "sentence-transformers_all-MiniLM-L12-v2_b1_g3_BPI_Challenge_2019",
        # "sentence-transformers_all-MiniLM-L12-v2_b1_g3_BPI_Challenge_2013_Incidents",
        # "sentence-transformers_all-MiniLM-L12-v2_b1_g3_BPI_Challenge_2013_ClosedProblems",
        # "sentence-transformers_all-MiniLM-L12-v2_b1_g3_BPI_Challenge_2012",
        # "sentence-transformers_all-MiniLM-L12-v2_b1_g3_Sepsis",
        # "sentence-transformers_all-MiniLM-L12-v2_b1_g3_HospitalBilling"
    ]
    
    for dataset in datasets_xes:
        process_log(dataset_xes=dataset, m=5)
    
    # retrieval_csv_sets = Path("/home/apost/projects/Diplomatiki/data/retrieval_csv_prefixes")
    
    # config = RAGConfig(dataset="/home/apost/projects/Diplomatiki/data/retrieval_csv_prefixes/b1_g3_BPI_Challenge_2012.csv", model_id="sentence-transformers/all-MiniLM-L12-v2")
    # for dataset in retrieval_csv_sets.iterdir():
    #     config.dataset=dataset
    #     config.store_embeddings()
        
    # test_df = pd.read_csv(config.test_set)
    # random_row = test_df.sample(n=1)
    # random_prefix = random_row["prefix"].iloc[0]
    # cntx, hits = config.retrieve_similar_prefixes(random_prefix, 20)
    # real_next_activity = random_row["prediction"].iloc[0]
    # print(random_row.to_string())
    # for key, value in cntx.items():
    #     print(key)
    #     print(value["prefix"])
    #     print(value["prediction"])
    #     print(value["score"])
    #     print()
        
    # output_text = api_call(cntx, random_prefix)
    # match = re.search(r"\\boxed\{([^}]*)\}", output_text)
    # prediction = match.group(1) if match else None
    
    # print(output_text)
    # print(f"Prediction: {prediction}, Real next activity: {real_next_activity}")
    


    # csv_prefixes = "/home/apost/projects/Diplomatiki/data/test_csv_prefixes/b1_g3_BPI_Challenge_2012.csv"
    # config = RAGConfig("sentence-transformers/all-MiniLM-L12-v2", csv_prefixes)
    # store_embeddings(config, csv_prefixes)
    # ctxt, hits = retrieve_similar_prefixes(config, qprefix)
    # print(ctxt)
    # print(f"Done. Collection: {[c.name for c in config.client.get_collections().collections]}")

    # qprefix = "A_SUBMITTED,A_PARTLYSUBMITTED,A_PREACCEPTED,W_Completeren aanvraag,W_Completeren aanvraag,W_Completeren aanvraag,W_Completeren aanvraag,W_Completeren aanvraag,W_Completeren aanvraag,W_Completeren aanvraag,A_ACCEPTED,O_SELECTED,A_FINALIZED,O_CREATED,O_SENT,W_Completeren aanvraag,W_Nabellen offertes,W_Nabellen offertes,W_Nabellen offertes,O_SENT_BACK,W_Nabellen offertes,W_Valideren aanvraag,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Valideren aanvraag,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Valideren aanvraag,W_Valideren aanvraag,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Valideren aanvraag,W_Valideren aanvraag,A_APPROVED,O_ACCEPTED - Values: {'orre': '10609', 'litr': 'COMPLETE', 'titi': '2011-11-15 12:50:28.812000+00:00'}"
    # collection_name = "sentence-transformers_all-MiniLM-L12-v2_BPI_Challenge_2012"
    # config = RAGConfig(collection_name=collection_name)
    # ctxt, hits = retrieve_similar_prefixes(config, qprefix)
    # print("########Context Ready#########")
    # print(api_call(ctxt, qprefix))
    # sort_xes_by_trace_and_timestamp_external(dataset,"/home/apost/projects/Diplomatiki/data/xes_logs/BPI_Challenge_2020_sorted.xes")
if __name__ == "__main__":
    main()

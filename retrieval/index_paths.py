THE_SPARSE_INDEX = {
    'dl19': 'msmarco-v1-passage',
    'dl20': 'msmarco-v1-passage',
    'dl21': 'msmarco-v2-passage',
    'dl22': 'msmarco-v2-passage',
    'dl23': 'msmarco-v2-passage',
    }

THE_TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'dl21': 'dl21',
    'dl22': 'dl22',
    'dl23': 'dl23',
}

def load_queries_qids(corpus_name):
    from pyserini.search import get_qrels, get_topics

    topics = get_topics(THE_TOPICS[corpus_name] if corpus_name != 'dl20' else 'dl20')
    if corpus_name in ['dl21', 'dl22', 'dl23']:
        qrels = get_qrels(f'{THE_TOPICS[corpus_name]}-passage')
    else:
        qrels = get_qrels(THE_TOPICS[corpus_name])
    test_only_qids_queries = set(qrels.keys())
    topics_qids = [(key, topics[key]['title'])  for key in topics if key in test_only_qids_queries]
    qids = [i[0] for i in topics_qids]
    queries = [i[1] for i in topics_qids]
    return qids, queries

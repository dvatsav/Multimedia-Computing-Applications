import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    
    rf_sim = sim
    iters = 3
    
    for iter_idx in range(iters):
        vec_docs = vec_docs.toarray()
        vec_queries = vec_queries.toarray()
        num_queries = vec_queries.shape[0]
        num_docs = vec_docs.shape[0]

        alpha = 1/n
        beta = 1/(num_docs-n)
        for i in range(num_queries):
            relevant_doc_idx = np.argsort(-rf_sim[:, i])[:n]
            non_relevant_doc_idx = np.argsort(rf_sim[:, i])[:num_docs-n]
            relevant_docs = vec_docs[relevant_doc_idx]
            non_relevant_docs = vec_docs[non_relevant_doc_idx]
            vec_queries[i] += alpha * np.sum(relevant_docs, axis=0) - beta*np.sum(non_relevant_docs, axis=0)
        vec_docs = csr_matrix(vec_docs)
        vec_queries = csr_matrix(vec_queries)
        rf_sim = cosine_similarity(vec_docs, vec_queries)
    
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    rf_sim = sim
    iters = 3
    k = 10 # number of terms for query expansion
    
    for iter_idx in range(iters):
        vec_docs = vec_docs.toarray()
        vec_queries = vec_queries.toarray()

        num_queries = vec_queries.shape[0]
        num_docs = vec_docs.shape[0]

        alpha = 1/n
        beta = 1/(num_docs-n)
        for i in range(num_queries):
            relevant_doc_idx = np.argsort(-rf_sim[:, i])[:n]
            non_relevant_doc_idx = np.argsort(rf_sim[:, i])[:num_docs-n]
            relevant_docs = vec_docs[relevant_doc_idx]
            non_relevant_docs = vec_docs[non_relevant_doc_idx]
            vec_queries[i] += alpha * np.sum(relevant_docs, axis=0) - beta*np.sum(non_relevant_docs, axis=0)

            # Query Expansion 
            

            best_terms_in_query_idx = np.argsort(-vec_queries[i])[:k]
            best_terms_in_docs = np.argsort(-relevant_docs[:, best_terms_in_query_idx]) 
            
            
            for k_idx in range(k):
                doc_terms_based_update = np.mean(np.sort(-relevant_docs[:, best_terms_in_query_idx[k_idx]]))

                vec_queries[i][best_terms_in_query_idx[k_idx]] += doc_terms_based_update
            

        vec_docs = csr_matrix(vec_docs)
        vec_queries = csr_matrix(vec_queries)
        rf_sim = cosine_similarity(vec_docs, vec_queries)
    return rf_sim
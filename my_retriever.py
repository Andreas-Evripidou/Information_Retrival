
import math


class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. ​(You can extend this method, as required.)
    def __init__(self,index,term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.cf_dict = self.compute_cf()
        self.df_dict = self.compute_df()
        # self.idf_dict = self.compute_idf()
        # write self.index to file
        open("help/index.txt", "w").write(str(self.index))
        # write self.num_docs to file
        open("help/num_docs.txt", "w").write(str(self.num_docs))
        
        
    def compute_number_of_documents(self): 
        self.doc_ids = set() 
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). ​Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        query_set = set(query)
        matrix = self.choose_weight_scheme(query, query_set)
        scores = {}
        query_vector = self.compute_query_vector(query, query_set)
        
        for i, doc_vector in enumerate(matrix):
            # print("For document:", i, "and vector:", doc_vector)
            scores[i] = self.cosine_similarity(query_vector, doc_vector)
        


        ranked_docs = self.rank_documents(scores)
        print("Ranked docs:", ranked_docs)
        # add 1 to each doc id to match the qrels file
        ranked_docs = [doc_id + 1 for doc_id in ranked_docs]

        return ranked_docs
    
    def choose_weight_scheme(self, query, query_set):
        match self.term_weighting:
                case 'binary':
                    return self.compute_binary_matrix(query_set)
                case 'tf':
                    return self.tf_model(query)
                case 'tfidf':
                    return self.tfidf_model(query)
                case _:
                    return self.boolean_model(query)

    # Method performing binary comperation for a single query
    # Returns a m x n document-term matrix where:
    # m is the number of documents in the collection
    # n is the number of terms in the query
    # The values in the matrix are the frequency of the term in the document
    def compute_binary_matrix(self, query_set):
        doc_term_matrix = []
        # For each document in the collection
        for doc_id in self.doc_ids:
            doc = []
            # For each term in the query
            for term in query_set:
                # If the term is not in the index, skip it
                if term not in self.index:
                    doc.append(0)
                # If the term is in the document, add the frequency
                elif doc_id in self.index[term]:
                    doc.append(self.index[term][doc_id])
                # If the term is not in the document, add 0
                else:
                    doc.append(0)
            doc_term_matrix.append(doc)
        return doc_term_matrix
    
    # Method for finding the cosine similarity between a query and a document
    # Returns the cosine similarity between the two vectors
    def cosine_similarity(self, query_vec, doc_vec):
        # If there is no query terms in the document, return -100
        if sum(doc_vec) == 0:
            return -100
        
        if sum(query_vec) == 0:
            print("Query vector is 0 -------------------------------------------------")

        numerator = 0
        doc_denominator = 0
        for i in range(len(query_vec)):
            numerator += query_vec[i] * doc_vec[i]
            doc_denominator += doc_vec[i]**2
        denominator = math.sqrt(doc_denominator)
        return numerator / denominator

    # Method for finding the term frequency for a single query
    # Returns t vectore of term frequencies:
    def compute_query_vector(self, query, query_set):
        query_vector = []
        for term in query_set:
            query_vector.append(query.count(term))
        return query_vector
    
    # Method ranking the documents based on the cosine similarity
    # Returns a list of document ids in rank order
    def rank_documents(self, cosine_simularity):
        return sorted(cosine_simularity, key= cosine_simularity.get,  reverse=True)[:10]

    
    # Method to compute the icollection frequency
    # Returns a list of collection frequencies for each term
    def compute_cf(self):
        cf = {}
        for term in self.index:
            cf[term] = sum(self.index[term].values())
        return cf
    
    # Method to compute the document frequency
    # Returns a list of document frequencies for each term
    def compute_df(self):
        df = {}
        for term in self.index:
            df[term] = len(self.index[term])
        return df

    
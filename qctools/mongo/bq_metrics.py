from mongoengine import EmbeddedDocument, EmbeddedDocumentField, FloatField, StringField, BooleanField, IntField, ListField

# class ClusteredVoteResult(EmbeddedDocument):
#     k = IntField(required=True)
#     alpha = FloatField()  # optional, for Bayesian cases
#     bitstring = StringField()
#     matching_bits = IntField()
#     best = BooleanField(default=False)

class BQJobMetric(EmbeddedDocument):

    job_id = StringField(required=True)
    time = FloatField(required=True)
    N = IntField(required=True)
    
    chi = IntField()
    
    target_found = BooleanField()
    target_in_counts = BooleanField()
    
    # max count
    max_count = IntField()
    max_bs = StringField()
    max_match = IntField()

    # majority vote
    majority_vote_bs = StringField()
    majority_vote_match = IntField()

    # clustered vote sweep (plain cluster vote)
    # cluster_vote_results = ListField(EmbeddedDocumentField(ClusteredVoteResult))
    cluster_vote_bs = StringField() # best, and smallest k
    cluster_vote_match = IntField() # best
    cluster_k = IntField() # best

    # Bayesian weighted vote sweep
    # bayes_vote_results = ListField(EmbeddedDocumentField(ClusteredVoteResult))
    bayes_vote_bs = StringField() # best, and smallest k
    bayes_vote_match = IntField() # best
    bayes_k = IntField() # best
    bayes_alpha = IntField() # best

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        target_found = False
        N = self.N
        if 'max_match' in kwargs:
            target_found = target_found or (N == kwargs['max_match'])

        if 'majority_vote_match' in kwargs:
            target_found = target_found or (N == kwargs['majority_vote_match'])

        if 'cluster_vote_match' in kwargs:
            target_found = target_found or (N == kwargs['cluster_vote_match'])

        if 'bayes_vote_match' in kwargs:
            target_found = target_found or (N == kwargs['bayes_vote_match'])

        self.target_found = target_found
    
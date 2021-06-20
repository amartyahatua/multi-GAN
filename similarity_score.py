from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import jaccard_score
from matplotlib import pyplot as plt

# Calculating the similarity

class SimilarityScore:
    def score(self, fake, real):
        fake = fake.cpu().detach().numpy()
        real = real.cpu().detach().numpy()

        cos_score = sum(sum(cosine_similarity(fake, real)))/100
        man_score = sum(sum(manhattan_distances(fake, real)))/100
        euc_score = sum(sum(euclidean_distances(fake, real)))/100
        x = fake[0]
        y =  real[0]
        dict = {}

        dict['cos_score'] = cos_score
        dict['man_score'] = man_score
        dict['euc_score'] = euc_score
        return dict
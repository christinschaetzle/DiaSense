# DiaSense
 
DiaSense was developed in the context of SemEval-2020 Task 1 `Unsupervised Lexical Semantic Change Detection' (Details: https://competitions.codalab.org/competitions/20948#learn_the_details) for the investigation of lexical semantic change.   

DiaSense assesses lexical semantic change via the calculation of metrics which have originally been suggested for the human annotation of diachronic meaning relatedness (see Schlechtweg et al., 2018). These metrics are calculated on the basis of pre-trained BERT embeddings (Devlin et al. 2018) in DiaSense. The BERT embeddings are generated using bert-as-service (Xiao 2018). The system is implemented in Python, with diasense.py being the main file while words_to_bert.py handles the mapping between target words and BERT embeddings.

More details on the methodology and ideas behind DiaSense will be published soon.



References:

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Dominik Schlechtweg, Sabine Schulte im Walde, and Stefanie Eckmann. 2018. Diachronic usage relatedness (DURel): A framework for the annotation of lexical semantic change. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 169-174, New Orleans, Louisiana, June. Association for Computational Linguistics.

Han Xiao. 2018. bert-as-service. https://github.com/hanxiao/bert-as-service.

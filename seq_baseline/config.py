# dataset params
def get_params(model_name):
    if model_name=='SEQ2SEQ':
        return seq_params
    else:
        raise ValueError("Params of %s not found"%model_name)

seq_params = {
        'hidden_size':   128,
        'word2vec':   '../wordvec/glove.6B.300d.txt',
        'embed_size':   300,
        'max_q_len':   30,
        'max_a_len':   30,
        }


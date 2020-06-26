from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model, load_model
from sklearn.metrics import classification_report

from tensorflow.python.keras.layers import Input, Dense, Embedding, SpatialDropout1D, Conv1D, concatenate
from tensorflow.python.keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import joblib
from tensorflow.python.keras.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
from sklearn.utils import class_weight  # weights for the classes
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils import np_utils

import matplotlib.pyplot as plt

print('importing done')


def read_data(fname):
    """
    Reading the training or testing data
    :param fname: file name only without extension (because that is how it works)
    :return Data: data frame after removing missing data
    """
    # Loading data
    Data = pd.read_csv(fname+'.csv')
    # Removing the missing rows from the data. There are some bad rows where values are missing
    Data = Data.dropna(subset=['Title'])

    return Data


# UNUSED
def get_ngrams(text, n=2):
    """
    cleaning & Tokenizing the text and returning N-gram tokens found in the text
    :param text: input text
    :param n: number of grams (length of the token?)
    :return grams: list of generated grams
    """
    # Removing punctuation
    text = re.sub(r'[\,\.\;\(\)\[\]\_\+\#\@\^]', ' ', text)
    # Tokenizaton
    tokens = [token for token in text.split(" ") if token.strip() != ""]
    # generating N-grams from tokens
    ngs = [ng for ng in ngrams(tokens, n)]

    return ["_".join(ng) for ng in ngs if len(ng) > 0] + tokens


def get_rnn_model(MAX_NB_WORDS, embedding_matrix_2):
    """
    Creating the RNN model
    :param MAX_NB_WORDS: maximum length of the seuquence 
    :param embedding_matrix_2: embedding matrix for the tokens
    """
    # defining input shape of the data
    inp = Input(shape=(50, ))

    # the layers
    # -------------------------------------------
    # defining Embedding layer
    x = Embedding(input_dim=MAX_NB_WORDS, ouput_dim=300, input_length=50, weights=[
                  embedding_matrix_2], trainable=False)(inp)
    # ----------------------------------------------------
    # dropout layer
    x = SpatialDropout1D(0.2)(x)
    # -------------------------------------------------------------------------
    # defining RRN part
    # two successive bidirectional GRU layers followed by a 1D Conv layer improved the performance
    """ after some trial and error this combination was found to be the best. dono really why :(
        GRU chosen over LSTM since it is simpler and faster
    """
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    x = Conv1D(512, kernel_size=1, padding="valid",
               kernel_initializer="he_uniform")(x)
    # --------------------------------------------------------------------------
    # defining two pooling layers average and maximum. so we reduce the dimensionality
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    # concatenating the two pooling layers
    conc = concatenate([avg_pool, max_pool])
    # -------------------------------------------------------------------------
    # applying batch normalization to speed the weights learning
    """
        standardizes the input values. mean = 0, std = 1
    """
    conc = BatchNormalization()(conc)
    # -------------------------------------------------------------------------
    conc = LeakyReLU()(conc)
    # -------------------------------------------------------------------------
    # defining the ouput layer
    outp = Dense(10, activation='softmax')(conc)
    # --------------------------------------------------------------------

    # defining the model
    model = Model(inputs=inp, outputs=outp)

    # if we want to load pre-trained weights
    # .load_weights("/weights-improvement-01-0.69.hdf5")

    # Compiling the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def get_coefs(word, *arr):
    """
    extracting from the embedding for each word its equivalent vector
    :param word: input word
    :param *arr: embedding matrix
    :return word: input word
    :return np.asarray(arr, dtype='float32'): vector of the word
    """
    try:
        return word, np.asarray(arr, dtype='float32')
    except:
        return None, None


def load_Embeddings(embedding_dir, embed_size):
    """
    loading embeddings and create embeddings matrix
    input:
    :embedding_dir: directory to the fasttext embedding vectors
    :embed_size: the length of the vectors
    output:
    :return all_embs: Embedding vectors
    :return emb_mean: Mean embedding vectors 
    :return emb_std: standard deviation of the embedding vectors
    :return embeddings_index: Embedding dictionary
    """

    # read rows of the embeddings ( containing token and embedding vector)
    embeddings_index = dict(get_coefs(*o.strip().split())
                            for o in tqdm_notebook(open(embedding_dir)))

    # defining the embedding size (length of the vectors)
    embed_size = 300
    # looping over the embedding index to load its value
    for k in tqdm_notebook(list(embeddings_index.keys())):
        v = embeddings_index[k]
        try:
            # ensure the embeddings values have same embeddings size
            if v.shape != (embed_size, ):
                embeddings_index.pop(k)
        except:
            pass

    # comment if it throws an error
    # embeddings_index.pop(None)

    # extract values (embedding vectors)
    values = list(embeddings_index.values())

    # concatenate all resulting lists into one so we can get mean and std
    all_embs = np.stack(values)
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    return all_embs, emb_mean, emb_std, embeddings_index


def Tokenize_Text(Data, tokenizer, MAX_LENGTH=50):
    """
    tokenizing and padding the sequences
    :param Data: text array
    :param tokenizer: tokenizer object
    :param MAX_LENGTH: sequnence sequence length
    :return pads :padded sequences
    """
    seqs = tokenizer.texts_to_sequences(Data)

    pads = pad_sequences(seqs, maxlen=MAX_LENGTH)
    return pads


def Create_EmbeddingMatrix(emb_mean, emb_std, embed_size, MAX_NB_WORDS, word_index, embeddings_index):
    """
    Initializing and creating the Embedding matrix
    :param emb_mean: mean of embeddings vectors
    :param emb_std: std of embeddings vectors
    :param MAX_NB_WORDS: maximum size of the vocabulary
    :param word_index: index of the tokenizer of the vocabulary
    :embeddings_index: embedding matrix
    :return embedding_matrix_2: final embedding matrix of your problem (mapping of data tokens 
    to the embedding vectors of fasttext model)
    """
    # Initializing the embedding matrix using maximum number of the vocab and embedding vector length
    embedding_matrix_2 = np.random.normal(
        emb_mean, emb_std, (MAX_NB_WORDS, embed_size))
    # total number of unique tokens
    N = 0

    # number of out of vocabulary tokens (tokens which does not exist in embedding_index)
    oov = 0
    # lookup for the embedding vectors of your vocab
    for word, i in tqdm_notebook(word_index.items()):
        if i >= MAX_NB_WORDS:
            continue
        N += 1
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:  # if the token exist in embedding index
            embedding_matrix_2[i] = embedding_vector
        else:
            oov += 1
    print(oov)
    print(N)
    return embedding_matrix_2


def Encoder(labels, encoder):
    """
    Encoding your labels 
    :param labels: target labels
    :param encoder: Encoder object
    :return dummy_y: encoded labels
    """
    encoded_Y = encoder.transform(labels)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y


def Train_Test_Split(filename):
    """
    Splitting data into Testing and training data
    :param filename: data file name
    """
    Data = pd.read_csv(filename+'.tsv', delimiter='\t')
    # removing duplicates in the data
    Data = Data.drop_duplicates(subset="Title", keep='first')
    # using test_test_split function to split the data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(
        Data['Title'], Data['Class'], test_size=0.1, random_state=42)
    Train_Data = pd.DataFrame()
    Test_Data = pd.DataFrame()
    Train_Data['Title'] = X_train
    Train_Data['Class'] = y_train
    Test_Data['Title'] = X_test
    Test_Data['Class'] = y_test
    # change name for english
    Train_Data.to_csv('DeTrainingData_2.csv', index=False)
    Test_Data.to_csv('DeTestingData_2.csv', index=False)


def predict(filename):
    """
    inference function to predict the output class of unseen data
    :param: filename
    """
    MAX_LENGTH = 50
    # defining classes (first layer)
    # class_names = ['0', '1', '2','3','4','5','6','7','8','9']
    # defining mapping dictionary to make dummy classes to the above classes name
    # id_name_map = dict(zip(range(len(class_names)),class_names))

    # loading model
    model = load_model("/Models/De_1_0.67.hdf5")

    # loading tokenizer
    tokenizer = joblib.load('/Models/tok_De_1.pickle')

    Data = read_data(filename)

    # tokenize the Data and pad to equal sizes
    padded_sequences = Tokenize_Text(Data['Message'], tokenizer, MAX_LENGTH)

    # predict
    y_pred = np.argmax(model.predict(padded_sequences, batch_size=128), axis=1)
    Data = Data.drop(['Title'], axis=1)
    Data['Class'] = y_pred
    #Data = Data.replace({"Class": id_name_map})
    #Data = Data.drop_duplicates()
    Data.to_csv('Book_Title_Classification.csv', index=False)


def asses_test(filename):
    MAX_LENGTH = 50
    #class_names = ['0', '1', '3','4','5','6','7','8','9']
    model = load_model("/Models/De_1_0.67.hdf5")
    tokenizer = joblib.load('/Models/tok_De_1.pickle')
    encoder = joblib.load('/Models/encoder_De_1.pickle')
    Data = read_data(filename)
    padded_sequences = Tokenize_Text(Data['Title'], tokenizer, MAX_LENGTH)
    y_pred = np.argmax(model.predict(padded_sequences, batch_size=128), axis=1)
    y_true = encoder.transform(Data['Class'])
    print(classification_report(y_true, y_pred))


def main():
    MAX_LENGTH = 50  # size of the input sequence
    MAX_NB_WORDS = 250000
    embed_size = 300
    # german data
    Data = read_data('DeTrainingData_1')
    # english data
    #Data = read_data('EnTrainingData_1')

    # load fasttext embeddings
    # fasttext embeddings:
    # download link: https://fasttext.cc/docs/en/crawl-vectors.html
    # english
    # embedding_dir = './fasttext embeddings/german/cc.en.300.vec'
    # german
    embedding_dir = './fasttext embeddings/german/cc.de.300.vec'

    # break training data into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(
        Data['Title'], Data['Class'], test_size=0.1, random_state=42)

    all_embs, emb_mean, emb_std, embeddings_index = load_Embeddings(
        embedding_dir, embed_size)
    # ------------------------------------------------------------
    # Tokenizer for our data
    # define tokenizer
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

    # fit tokenizer on data. it will give a dictionary of words and their assigned values in word_index
    tokenizer.fit_on_texts(Data['Title'])

    # save tokenizer
    #joblib.dump(tokenizer, '/Models_test/tok_DE_1.pickle')

    # use the tokenizer to transform the input text data into sequence of integers. each token (word) is mapped into its
    # corresponding integer value in word_index
    padded_train_sequences = Tokenize_Text(X_train, tokenizer, MAX_LENGTH)
    padded_val_sequences = Tokenize_Text(X_val, tokenizer, MAX_LENGTH)

    # gives a dictionary of words and their uniquely assigned integers (this is actually the internal state of the tokenizer
    # which can be updated with more text)
    word_index = tokenizer.word_index
    # ------------------------------------------------------------
    # make embedding matrix from fasttex model. we take all corresponding embedding rows from the model
    # word_index are the tokens from our text,
    # embedding_index are the rows from the fasttex embedding model
    embedding_matrix_2 = Create_EmbeddingMatrix(
        emb_mean, emb_std, embed_size, MAX_NB_WORDS, word_index, embeddings_index)
    # ------------------------------------------------------------
    print('defining the simple network ...')
    # embedding matrix is fed to the first layer (embedding layer)
    rnn_simple_model = get_rnn_model(MAX_NB_WORDS, embedding_matrix_2)
    print(rnn_simple_model.summary())
    # -------------------------------------------------------------
    # giving the classes weights based on their occurences/frequencies ( n_samples / (n_classes * np.bincount(y)) )
    # second input argument is the list of labels. third is all of them with repetitions
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train),
                                                      y_train)
    # -------------------------------------------------------------
    #### Converting Classes into label Encoding ######
    """
    it is kind of a similar procedure to tokenizing the input data. we first define an
    encoder (which encodes the class labels to values between 0 and n_class-1) and train
    it on the labels. then this encoder is used to tranform train and validation labels 
    into normalized values. they are then converted into one-hot encoded vectors
    """
    encoder = LabelEncoder()
    encoder.fit(y_train)
    #joblib.dump(encoder, '/models_test/encoder_DE_1.pickle')
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y_train = Encoder(y_train, encoder)
    dummy_y_val = Encoder(y_val, encoder)
    # -------------------------------------------------------------
    # callbacks
    # these two callbacks monitor the network's improvements
    # if the specified metric reaches a plateau, reduces the learning rate
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', cooldown=0, min_lr=0.0000001)
    # if the specified metric is not improving, stops the training early (before finishing the epochs)
    es = EarlyStopping(monitor='val_loss', min_delta=0.00005,
                       patience=5, verbose=0, mode='auto')

    # callback for checkpoint. saves the model after each epoch
    # german
    filepath = "./Models_test/DE-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    # english
    # filepath="./Models_test/EN-{epoch:02d}-{val_accuracy:.2f}.hdf5
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # ----------------------------------------------------------------------------
    """
        now we run our netwrok. 
        - train and validation sets are transformed into sequences of integers
        - output labels are one-hot encoded
        - the weights are also given. so the network considers if some classes is appearing a lot more than the others
    """
    history = rnn_simple_model.fit(padded_train_sequences, dummy_y_train, validation_data=(padded_val_sequences, dummy_y_val), batch_size=128, callbacks=[lr_scheduler, es, checkpoint], class_weight=class_weights  # so the networks considers the frequency of the classes
                                   , epochs=50)
    # --------------------------------------------------------------------------
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # Saving Accuracy Figure
    plt.savefig('DE_1_Accuracy.png')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # Saving Loss Figure
    plt.savefig('DE_1_Loss.png')
    plt.show()


main()

# asses_test('DeTestingData_1')
# Train_Test_Split('german_2')

#import timeit

#start = timeit.default_timer()

# predict('Vodafone')

#stop = timeit.default_timer()

#print('Time: ', stop - start)

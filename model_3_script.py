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
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils import np_utils

import matplotlib.pyplot as plt

print('importing done')


def read_data(fname):
    """
    Reading the training or testing data
    :param fname: file name only without extension
    :return Data: data frame after removing missing data
    """
    # Loading data
    Data = pd.read_csv(fname+'.csv')\
        # Removing the missing rows from the data
    Data = Data.dropna(subset=['Title'])

    return Data


def get_ngrams(text, n=2):
    """
    Removing & Tokenizing the text and returning N-gram tokens found in the text
    :param text: input text
    :param n: number of grams
    :return grams: list of generated grams
    """
    # Removing punctuation
    text = re.sub(r'[\,\.\;\(\)\[\]\_\+\#\@\^]', ' ', text)
    # Tokenizaton
    tokens = [token for token in text.split(" ") if token.strip() != ""]
    # generating N-grams
    ngs = [ng for ng in ngrams(tokens, n)]

    return ["_".join(ng) for ng in ngs if len(ng) > 0]+tokens


def get_rnn_model(MAX_NB_WORDS, embedding_matrix_2):
    """
    Creating the RNN model
    :param MAX_NB_WORDS: maximum length of the seuquence 
    :param embedding_matrix_2: embedding matrix
    """

    # -----------------------------------------------------
    """
    both previous models are used here. predictions from both models are used 
    as metadata in the network
    """
    # defining input shape of the data
    inp = Input(shape=(50, ))
    # defining input shape of the level_1 predictions
    meta_input_1 = Input(shape=(1,))
    # defining input shape of the level_2 predictions
    meta_input_2 = Input(shape=(1,))
    # -----------------------------------------------------
    # defining Embedding layer
    x = Embedding(MAX_NB_WORDS, 300, input_length=50, weights=[
                  embedding_matrix_2], trainable=False)(inp)
    # -----------------------------------------------
    # defining spatial dropout
    x = SpatialDropout1D(0.2)(x)
    # ----------------------------------------------------------
    # defining RRN part
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)

    # defining the convolutional layer
    x = Conv1D(512, kernel_size=1, padding="valid",
               kernel_initializer="he_uniform")(x)
    # --------------------------------------------------------
    # defing two pooling layers average and maximum
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    # concating the two pooling layers
    conc = concatenate([avg_pool, max_pool])
    # -----------------------------------------------------
    # applying batch normalization to speed the weights learning
    conc = BatchNormalization()(conc)
    # ----------------------------------------------------
    """
    both predictions are concatenated with the new input data
    and fed into a dense layer
    """
    # concating the numerical and embedding feature
    conc = concatenate([conc, meta_input_1, meta_input_2])
    # applying dense layer on the concatenation
    conc = Dense(512)(conc)
    # ---------------------------------------------------
    # increases the learning speed
    conc = BatchNormalization()(conc)
    # ---------------------------------
    # applying leakyRelu
    conc = LeakyReLU()(conc)
    # --------------------------------------------------
    # defining the ouput layer
    outp = Dense(477, activation='softmax')(conc)
    # --------------------------------------------------
    # 3 inputs
    model = Model(inputs=[inp, meta_input_1, meta_input_2], outputs=outp)

    # if you to load pre-trained weights
    # .load_weights("weights-improvement-01-0.69.hdf5")

    # Compiling the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def get_coefs(word, *arr):
    """
    extracting from the embedding for each word it equivalent vector 
    :param word: input word
    :param *arr: embedding matrix
    :return word: input word
    :return np.asarray(arr, dtype='float32'): vector of the word
    """
    try:
        return word, np.asarray(arr, dtype='float32')
    except:
        return None, None


def load_Embeddings(MAX_LENGTH=50, MAX_NB_WORDS=250000):
    """
    loading embeddings and create embeddings matrix
    :param MAX_LENGTH: maximum length of the sequence
    :param MAX_NB_WORDS: maximum size of the vocabulary
    :return all_embs: Embedding vectors
    :return emb_mean: Mean embedding vectors 
    :return emb_std: standard deviation of the embedding vectors
    :return embeddings_index: Embedding dictionary
    """
    embeddings_index = dict(get_coefs(*o.strip().split())
                            for o in tqdm_notebook(open('./fasttext embeddings/german/cc.de.300.vec')))

    # defining the embedding size
    embed_size = 300
    # looping on the embedding index to load its value
    for k in tqdm_notebook(list(embeddings_index.keys())):
        v = embeddings_index[k]
        try:
            # insure the embeddings values have same embeddings size
            if v.shape != (embed_size, ):
                embeddings_index.pop(k)
        except:
            pass
    # comment of you get error
    # embeddings_index.pop(None)
    values = list(embeddings_index.values())
    all_embs = np.stack(values)

    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    return all_embs, emb_mean, emb_std, embeddings_index


def Text_to_Sequence(Data, tokenizer, MAX_LENGTH=50):
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
    :return embedding_matrix_2: final embedding matrix of your problem
    """
    # Initializing the embedding matrix using maximum number of the vocab and embedding vector length
    embedding_matrix_2 = np.random.normal(
        emb_mean, emb_std, (MAX_NB_WORDS, embed_size))

    # total number of unique tokens
    N = 0
    # mumber of out of vocabulary
    oov = 0
    # lookup for the embedding vectors of your vocab
    for word, i in tqdm_notebook(word_index.items()):
        if i >= MAX_NB_WORDS:
            continue
        N += 1
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
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
    Data = pd.read_csv(filename+'.csv')

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
    Train_Data.to_csv('DeTrainingData_3.csv', index=False)
    Test_Data.to_csv('DeTestingData_3.csv', index=False)


def load_models(lan):
    if lan == 'en':
        # 1st level model:
        model_1 = load_model("./Models/EN_1_0.77.hdf5")
        # 2nd level model
        model_2 = load_model("./Models/EN_2_0.65.hdf5")
        # 3rd level model
        model = load_model("./Models/EN_2_0.29.hdf5")
        # loading tokenizer
        tokenizer = joblib.load('./Models/tok_EN_3.pickle')
    elif lan == 'de':
        model_1 = load_model("./Models/De_1_0.67.hdf5")
        model_2 = load_model("./Models/DE_2_0.53.hdf5")
        model = load_model("./Models/DE_3_0.33.hdf5")
        tokenizer = joblib.load('./Models/tok_DE_3.pickle')

    return model_1, model, tokenizer


def predict(filename):
    """
    inference function to predict the output class of unseen data
    :param: filename
    """
    MAX_LENGTH = 50

    # defining classes
    #class_names = ['0', '1', '2','3','4','5','6','7','8','9']
    # defining mapping dictionary to make dummy classes to the above classes name
    #id_name_map = dict(zip(range(len(class_names)),class_names))

    # load models
    # german
    model_1, model_2, model, tokenizer = load_models('de')
    # english
    #model_1, model_2, model, tokenizer = load_model('en')

    Data = read_data(filename)

    # applying pipeline on the Data to generate the padded sequence
    padded_sequences = Text_to_Sequence(Data['Message'], tokenizer, MAX_LENGTH)

    # -------------------------------------------------------------------
    # predicting 1st level titles
    level_1_num = np.argmax(model_1.predict(
        padded_sequences, batch_size=256), axis=1)
    # predicting 2nd level titles
    level_2_num = np.argmax(model_2.predict(
        padded_sequences, batch_size=256), axis=1)
    # predicting 2nd level titles
    y_pred = np.argmax(model.predict([padded_sequences, np.float32(
        level_1_num), np.float32(level_2_num)], batch_size=256), axis=1)
    # --------------------------------------------------------------------

    Data = Data.drop(['Title'], axis=1)
    Data['Class'] = y_pred
    #Data = Data.replace({"Class": id_name_map})
    #Data = Data.drop_duplicates()
    Data.to_csv('Book_Title_Classification.csv', index=False)


def asses_test(filename):
    MAX_LENGTH = 50
    #class_names = ['0', '1', '3','4','5','6','7','8','9']

    # load models
    # german
    model_1, model_2, model, tokenizer = load_models('de')
    # english
    #model_1, model_2, model, tokenizer = load_models('en')

    # german
    encoder = joblib.load('./Models/encoder_De_3.pickle')
    # english
    # encoder = joblib.load('./Models/encoder_En_3.pickle')

    Data = read_data(filename)
    padded_sequences = Text_to_Sequence(Data['Title'], tokenizer, MAX_LENGTH)

    # predictions using first level model
    level_1_num = np.argmax(model_1.predict(
        padded_sequences, batch_size=256), axis=1)
    # predictions using second level model
    level_2_num = np.argmax(model_2.predict(
        [padded_sequences, np.float32(level_1_num)], batch_size=256), axis=1)

    # predicting the third level model with metadata
    y_pred = np.argmax(model.predict([padded_sequences, np.float32(
        level_1_num), np.float32(level_2_num)], batch_size=256), axis=1)
    # real labels
    y_true = encoder.transform(Data['Class'])

    print(classification_report(y_true, y_pred))


def main():
    MAX_LENGTH = 50
    MAX_NB_WORDS = 250000
    embed_size = 300
    # german
    Data = read_data('DeTrainingData_3')
    # english
    # Data = read_data('EnTrainingData_3')

    X_train, X_val, y_train, y_val = train_test_split(
        Data['Title'], Data['Class'], test_size=0.1, random_state=42)

    all_embs, emb_mean, emb_std, embeddings_index = load_Embeddings(
        MAX_LENGTH=50, MAX_NB_WORDS=250000)

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(Data['Title'])

    # german
    joblib.dump(tokenizer, './Models_test/tok_DE_3.pickle')
    # english
    # joblib.dump(tokenizer, './Models_test/tok_EN_3.pickle')

    # to numerical vectors
    padded_train_sequences = Text_to_Sequence(X_train, tokenizer, MAX_LENGTH)
    padded_val_sequences = Text_to_Sequence(X_val, tokenizer, MAX_LENGTH)
    # --------------------------------------------------------------------
    """
    - The data is first fed into the first-level model and predictions are acquired.
    - Then the same data is concatenated with the acquired predictions (as metadata) 
      and are fed into the second level model. 
    - Finally the original data is again concatenated with both predictions from both
      first and second models and fed into the network
    """
    # Predictions by first-level model
    # german
    model_lev_1 = load_model("./Models_test/DE_1_0.67.hdf5")
    # english
    # model_lev_1 = load_model("./Models_test/EN_1_0.77.hdf5")
    level_1_num_train = np.argmax(model_lev_1.predict(
        padded_train_sequences, batch_size=256), axis=1)
    level_1_num_val = np.argmax(model_lev_1.predict(
        padded_val_sequences, batch_size=256), axis=1)

    # Predictions by second-level model with first level predictions as metadata
    # german
    model_lev_2 = load_model("./Models/DE_2_0.53.hdf5")
    # english
    # model_lev_2 = load_model("./Models/EN_2_0.65.hdf5")
    level_2_num_train = np.argmax(model_lev_2.predict(
        [padded_train_sequences, np.float32(level_1_num_train)], batch_size=256), axis=1)
    level_2_num_val = np.argmax(model_lev_2.predict(
        [padded_val_sequences, np.float32(level_1_num_val)], batch_size=256), axis=1)
    # -----------------------------------------------------------------------------
    # get embedding matrix (is fed into the first layer => embedding layer)
    word_index = tokenizer.word_index
    embedding_matrix_2 = Create_EmbeddingMatrix(
        emb_mean, emb_std, embed_size, MAX_NB_WORDS, word_index, embeddings_index)
    # -----------------------------------------------------------------------------
    # compile the network
    rnn_model = get_rnn_model(MAX_NB_WORDS, embedding_matrix_2)
    print(rnn_model.summary())
    # ----------------------------------------------------------------------------------
    # take care of inbalance
    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(y_train), y_train)
    # -------------------------------------------------------------------------------------------
    #### Converting Classes into label Encoding ######
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train)
    joblib.dump(encoder, './Models/encoder_DE_3.pickle')
    # convert integers to one-hot encoding
    dummy_y_train = Encoder(y_train, encoder)
    dummy_y_val = Encoder(y_val, encoder)
    # ----------------------------------------------------------------------------------------
    # callbacks
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', cooldown=0, min_lr=0.0000001)
    es = EarlyStopping(monitor='val_loss', min_delta=0.00005,
                       patience=5, verbose=0, mode='auto')

    # save checkpoint
    filepath = "./Models_test/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # -----------------------------------------------------------------------------------------------
    # run the network
    """
    train and validation sets = [data, predictions from the first model, predictions from the second model]
    """
    history = rnn_model.fit([padded_train_sequences, level_1_num_train, level_2_num_train], dummy_y_train, validation_data=(
        [padded_val_sequences, level_1_num_val, level_2_num_val], dummy_y_val), batch_size=256, callbacks=[lr_scheduler, es, checkpoint], class_weight=class_weights, epochs=50)

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # Saving Accuracy Figure
    plt.savefig('DE_3_Accuracy.png')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # Saving Loss Figure
    plt.savefig('DE_3_Loss.png')
    plt.show()


main()
# asses_test('EnTestingData_2')
# Train_Test_Split('german_3_excl_Minority')

#import timeit
#
#start = timeit.default_timer()
#
# predict('Vodafone')
#
#stop = timeit.default_timer()
#
#print('Time: ', stop - start)

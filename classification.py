import sklearn
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def classification(processed_df):

    # TF-IDF
    X_tf = TF_IDF(processed_df)
    # split into train and test sets
    # classification
    X_train_tf = X_tf[:490, :]
    X_test_tf = X_tf[490:, :]
    # Your code here
    # Answer begin
    y_tf = processed_df["industry"].values  # conver object series to ndarray
    y_train = y_tf[:490]
    y_test = y_tf[490:]
    clf = MultinomialNB()  # create an example of Multinomial Naive Bayes classificator
    clf.fit(X_train_tf, y_train)  # To train
    y_test_predict = clf.predict(X_test_tf)  # To predict
    results = metrics.classification_report(y_test, y_test_predict)  # To measure the performance
    # Answer end
    return results


def TF_IDF(train):
    # Your code here
    # Answer begin
    train_text_list = list(train["business_scope"])  # To convert the object series to list
    vectorizer = CountVectorizer() # To create example of the two classes
    transformer = TfidfTransformer()
    train_text_vector = vectorizer.fit_transform(train_text_list)  # To obtain tf matrix
    X_train_tf = transformer.fit_transform(train_text_vector)  # Text vectorization by tf-idf
    X_train_tf = X_train_tf.toarray()  # Convert to ndarray
    # Answer end
    return X_train_tf

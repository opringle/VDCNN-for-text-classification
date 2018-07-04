import pandas as pd

def preprocess_df(df, name):

    index_to_intent = {1: 'Society & Culture', 2: 'Science & Mathematics', 3: 'Health', 4: 'Education & Reference',
                       5: 'Computers & Internet', 6: 'Sports', 7: 'Business & Finance', 8: 'Entertainment & Music',
                       9: 'Family & Relationships', 10: 'Politics & Government'}

    df.intent = df.intent.map(index_to_intent.get)
    df.drop_duplicates(inplace=True, subset=['utterance'])

    print("{} {} records".format(df.shape[0], name))

    return df[['intent', 'utterance']]

test_df = pd.read_csv('./data/yahoo_answers_csv/test.csv', names=['intent', 'utterance', 'question_content', 'best_answer'])
train_df = pd.read_csv('./data/yahoo_answers_csv/train.csv', names=['intent', 'utterance', 'question_content', 'best_answer'])

test_df = preprocess_df(test_df, 'test')
train_df = preprocess_df(train_df, 'train')

train_df.to_pickle('./data/yahoo_train.pickle')
test_df.to_pickle('./data/yahoo_test.pickle')

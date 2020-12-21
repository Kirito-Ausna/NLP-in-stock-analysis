import tushare as ts
import pandas as pd
import jieba
import jieba.analyse


def get_data(token, N):
    # get_data
    pro = ts.pro_api(token)
    pd.set_option('max_colwidth', 120)
    df0 = pro.stock_company(exchange='SZSE', fields='ts_code, business_scope')
    df1 = df0.dropna(axis=0, how='any')
    df2 = pro.stock_basic(exchange='SZSE', fields='ts_code,industry')
    df2 = df2.dropna(axis=0, how='any')  # get rid of the line if there exits NA
    # merge
    # Your code here
    # Answer begin
    df1.applymap(str.strip)  # As a result of changing the data source, there is no need to rename
    df2.applymap(str.strip)  # get rid of \t\d\n white space etc from the begin and the end of string
    # Answer end
    df = pd.merge(df1, df2, how='right')  # join df1,df2 with respect to the column of df2
    # filter by number of records
    nonan_df = df.dropna(axis=0, how='any')
    vc = nonan_df['industry'].value_counts()
    pat = r'|'.join(vc[vc > N].index)
    merged_df = nonan_df[nonan_df['industry'].str.contains(pat)]

    return merged_df


def word_process(business_scope):
    seg_list = jieba.cut(business_scope)  # use jieba to cut the text
    keyword = jieba.analyse.extract_tags(business_scope, topK=20, withWeight=False)  # use jieba to gain the tags
    combinantion = " ".join(seg_list) + " " + " ".join(keyword)  # join seg_list and cut, each word separated by space
    return combinantion


def text_preprocess(merged_df):
    # word segmentation + extract keywords (using jieba)
    # Your code here
    # Answer begin
    processed_text_list = merged_df["business_scope"].apply(word_process)  # apply word_process function to the series
    # of business_scope
    processed_df = pd.concat([processed_text_list, merged_df["industry"]], axis=1)  # link the process_text_list
    # and series of industry by column
    return processed_df

# Answer end

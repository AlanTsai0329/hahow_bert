# %%
from datetime import datetime, timezone
import re
import ast
import xlsxwriter
import numpy as np
import pandas as pd
from lxml import etree
from pathlib import Path
from lib.Data_Processer import DataProcessTools

DATA_DIR = Path('data')
dataprocesstools = DataProcessTools()

# %%
# 讀入資料：課程基本資料、課程描述、價格資料、評論資料
course_df = pd.read_csv(f'{DATA_DIR}/all_course_page.csv', encoding='utf-8-sig') 
content_df = pd.read_csv(f'{DATA_DIR}/all_course_page_content.csv', encoding='utf-8-sig') 
price_df = pd.read_csv(f'{DATA_DIR}/all_course_page_price.csv', encoding='utf-8-sig') 
comment_df = pd.read_csv(f'{DATA_DIR}/all_course_page_comment.csv', encoding='utf-8-sig') 

# %%
"""
clean course_df 
"""
course_processed_df = (
    course_df
    # 清理值是 dict 的資料
    ## owner clean
    .pipe(dataprocesstools.join_dict_columns, "owner")
    ## successCriteria clean
    .pipe(dataprocesstools.join_dict_columns, "successCriteria")
    ## campaign clean
    .pipe(dataprocesstools.join_dict_columns, "campaign")
    ## contentRating clean
    .pipe(dataprocesstools.join_dict_columns, "contentRating")
    
    # status clean
    .assign(status=course_df['status'].map(dataprocesstools.status_map))
    # get each course url
    .assign(course_url=lambda df: df['_id'].apply(lambda x: f"https://hahow.in/courses/{x}/main"))
    # 去除價格是免費的課程
    .query("price != 0")
)

cleaned_course_df = (
    course_processed_df
    # 保留需要的資料
    [[
        # 課程資訊
        '_id', 'title', 'metaDescription', 'status', 'uniquename', 'tags', 
        'averageRating', 'numRating', 'bookmarkCount', 
        'video', 'totalVideoLengthInSeconds',  
        'assignment', '課程類別', 'course_url',
        # owner
        'owner__id', 'owner_name', 
        #  time
        'estimatedCourseStartTime', 'createdAt', 'incubateTime', 'publishTime', 'proposalDueTime',
        # ???
        'isReject', 'modules', 'includedKnowledgeCollections', 'bestDiscount', 'successCriteria_numSoldTickets',
    ]]
)

cleaned_course_df.to_csv(f"{DATA_DIR}/cleaned_all_course_page.csv", encoding='utf-8-sig', index=False)

# %%
"""
clean content_df 
"""
cleaned_content_df = (
    content_df[[
        # 課程資訊
        'title', 'description', 'requiredTools', 'recommendedBackground', 'willLearn', 'targetGroup',
        'viewCount', 'isTop10Percent', 'contentRating.brief', 'contentRating.content',
        # owner
        'owner.username', 'owner.name', 'owner.metaDescription', 'owner.description', 'owner.skills', 'owner.interests', 
        'owner.links.facebook', 'owner.links.website', 'owner.links.instagram',  'owner.links.behance',  'owner.links.googlePlus',
        'owner.links.pinterest', 'owner.links.facebookPixel',  'owner.links.youtube',
        'owner.publishedProductStatistics.courseCount', 'owner.publishedProductStatistics.articleCount', 'owner.publishedProductStatistics.studentCount',
        # group
        'group.title', 'group.subGroup.title',
        # time
        'draftRevisionUpdateTime', 'revisionReleaseTime', 'latestCourseStartTime',
        # ???
        'incubationSchedule.owner', 'incubationSchedule.preferredTime', 'skipIncubating', 'migratedToWistia', 'wistiaStatus', 'videoBackupStatus',
        'estimatedCourseStartIntervalInDays', 'estimatedCourseVideoLengthInMins',
        ]]
)

# %%
"""
clean price_df 
"""
# 刪掉重複 columns、清理 contentRating
cleaned_price_df = (
    price_df
    [[
        'title', 'preOrderedPrice', 'price', 'contentRating',  
        'numSoldTickets', '開課後銷售額',
       ]]
    .pipe(dataprocesstools.join_dict_columns, "contentRating")
)

# %%
"""
clean comment_df 
"""
cleaned_comment_df = (
    comment_df
    [['title', 'comment']]
    # .assign(comment = comment_df['comment'].apply(ast.literal_eval))
    # .assign(all_comment = lambda df: df['comment'].apply(lambda x: dataprocesstools.clean_comment(x, "all_comment")))
    # .assign(all_comment_usefulcount = lambda df: df['comment'].apply(lambda x: clean_comment(x, "usefulCount")))
    # .assign(all_comment_rating = lambda df: df['comment'].apply(lambda x: clean_comment(x, "rating")))
    # .astype(str)
    # .drop_duplicates(subset=['title'], keep='first')
)

cleaned_comment_df.to_csv("comment_data_for_tp.csv", encoding='utf-8-sig', index=False)



# %%
"""
合併處理後的資料
"""
merge_df = (
    cleaned_course_df
    .merge(cleaned_price_df, on="title", how='left')
    .merge(cleaned_comment_df, on="title", how='left')
    .merge(cleaned_content_df, on="title", how='left')
    .drop_duplicates(subset=['title'], keep='first')
)

merge_df.to_csv(f'{DATA_DIR}/merged_course_data.csv', encoding='utf-8-sig', index=False)
merge_df.to_excel(f'{DATA_DIR}/merged_course_data.xlsx', engine='xlsxwriter')

# %%
"""
處理文字資料
"""
processed_course_df = (
    merge_df
    # fill 文字的 na
    .assign(
        recommendedBackground=merge_df['recommendedBackground'].fillna("未提供推薦背景"),
        targetGroup=merge_df['targetGroup'].fillna("未提供哪些人適合這堂課"),
        willLearn=merge_df['willLearn'].fillna("未提供你可以學到"),
        requiredTools=merge_df['requiredTools'].fillna("未提供上課前的準備"),
        owner_metaDescription=merge_df['owner.metaDescription'].fillna("未提供講師簡介"),
        fillna_owner_description=merge_df['owner.description'].fillna("未提供講師描述"),
        owner_skills=merge_df['owner.skills'].fillna("未提供講師技能"),
        owner_interests=merge_df['owner.interests'].fillna("未提供講師興趣"),
        )
    # 將 html 元素去除
    .assign(
        description=lambda df: df['description'].apply(dataprocesstools.etree_clean),
        owner_description=lambda df: df['fillna_owner_description'].apply(dataprocesstools.etree_clean),
        )
    # 去除標點符號
    .pipe(dataprocesstools.remove_punc)
    .rename({
        "owner_skills": "owner.skills",
        "owner_interests": "owner.interests",
    })
)

# %%
"""
處理totalVideoLengthInSeconds，因有些課程未上架，所以沒有完整的課程資訊，
所以用 estimatedCourseVideoLengthInMins 來代替
"""
processed_course_df = (
    processed_course_df
    .assign(totalVideoLengthInSeconds=processed_course_df['totalVideoLengthInSeconds'].replace(0, np.nan))
    .assign(totalVideoLengthInSeconds=lambda df: df['totalVideoLengthInSeconds'].fillna(df['estimatedCourseVideoLengthInMins']*60))

)


# %%
"""
組出銷售額資料：
numSoldTickets - 完整銷售額
開課後銷售額 - 開課後銷售額
presoldtickets - 募資期銷售額 = 完整銷售額 - 開課後銷售額
"""
processed_course_df = (
    processed_course_df
    .astype({'numSoldTickets': 'float',
             "開課後銷售額":"float"})
    .assign(presoldtickets=lambda df: df['numSoldTickets'] - df['開課後銷售額'])
)
# processed_course_df.query("presoldtickets < 0")[['numSoldTickets', '開課後銷售額', 'presoldtickets']]

# %%
"""
組出課程期間資料
"""

processed_course_df.loc[processed_course_df['title']=='24單元入門投資科學一次搞懂投資必備知識', "incubateTime"] = datetime(2020, 8, 18, 12, 0, 0)

add_time_course_df = (
    processed_course_df
    .assign(
        proposalDueTime=pd.to_datetime(processed_course_df['proposalDueTime'].fillna(pd.Timestamp("20000101"))),
        incubateTime=pd.to_datetime(processed_course_df['incubateTime'].fillna(pd.Timestamp("20000101"))),
        publishTime=pd.to_datetime(processed_course_df['publishTime'].fillna(datetime.now(timezone.utc)))
        )
    .assign(
        # 募資月數>30天就是2個月，否則往下一層判斷，小於30天
        募資月數m=lambda df: np.where(
            (df['proposalDueTime'] - df['incubateTime']).dt.days > 30, 
            2,
            np.where(
                (df['proposalDueTime'] - df['incubateTime']).dt.days <= 30, 
                1, 
                0) 
                ),
        開課月數m=lambda df: round((datetime.now(timezone.utc) - df['publishTime']).dt.days / 30, 0)
    )
)




# %%
"""
組成銷售額、募資期間的資料
"""
unselected_course_df = (
    add_time_course_df   
    .assign(prices=lambda df: list(df[['preOrderedPrice', "price"]].values))
    .assign(sales=lambda df: list(df[['presoldtickets', "開課後銷售額"]].values))
    .assign(months=lambda df: list(df[['募資月數m', "開課月數m"]].values))
    .explode(['prices', 'sales', 'months'])
    .drop(columns=['presoldtickets', '開課後銷售額', 'preOrderedPrice', "price", '募資月數m', "開課月數m"])
)




# %%
final_course_df = (
    unselected_course_df
    [[
"_id",
# 文字
"title",
"metaDescription",
"description",
"requiredTools",
"recommendedBackground",
"willLearn",
"targetGroup",
"owner.metaDescription",
"owner.skills",
"owner.interests",
"owner_description",
# label encoding
"owner.name",
"課程類別",
"group.subGroup.title",
"status",
"includedKnowledgeCollections",
"successCriteria_numSoldTickets",
"isTop10Percent",
# one hot encoding
"tags",
# standardscaler
"averageRating",
"numRating",
"bookmarkCount",
"totalVideoLengthInSeconds",
"viewCount",
"prices",
"sales",
# 不用處理
"owner.publishedProductStatistics.courseCount",
"owner.publishedProductStatistics.articleCount",
# 暫保留
"publishTime",
]]
)

final_course_df.to_csv(f'{DATA_DIR}/processed_merged_course_data.csv', encoding='utf-8-sig', index=False)

# %%

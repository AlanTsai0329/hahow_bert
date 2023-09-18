import re
import ast
import pandas as pd
from lxml import etree
from pathlib import Path

class DataProcessTools:
    def __init__(self) -> None:
        self.status_map = {
            "PUBLISHED": "已開課",
            "SUCCESS": "募資成功",
            "INCUBATING": "募資中"
        }
        self.text_col = [
            "title", "metaDescription", 'description',
            'requiredTools', 'recommendedBackground', 'willLearn', 'targetGroup',
            'owner.metaDescription', 'owner_description', 'owner.skills', 'owner.interests',
        ]


    @staticmethod
    def join_dict_columns(df, col_name):
        """dict columns clean：
        owner、successCriteria、campaign、contentRating、
        coverImage、basePricingInfo、squareCoverImage

        Args:
            df (DataFrame): 要處理的 dataframe
            col_name (str): 要處理的 column names

        Returns:
            DataFrame: 處理後的df
        """
        try:
            # change data type to dict
            df[col_name] = df[col_name].apply(ast.literal_eval)
        except Exception as e:print(e)

        # transfer dict values to pd.Series
        join_df = df[col_name].apply(pd.Series)

        # join to original df
        df = (
            df
            .join(join_df.add_prefix(f"{col_name}_"))
            .drop(columns=[col_name])
            )
        return df
    
    @staticmethod
    def etree_clean(x):
        """處理具有HTML元素的欄位：將HTML元素去除

        Args:
            x (str): _description_

        Returns:
            str: 去除HTML元素的資料
        """
        if x != "":
            tree = etree.HTML(x)
            description = tree.xpath('/html/body//text()')
            # 去除 \n 跟空白
            newline_re = re.compile(r"(\n|\s)", re.S)
            des_no_nl_space = [newline_re.sub("", des) for des in description]
            # 去除 \
            quote_re = re.compile(r'[\"\']', re.S)
            des_no_nl_space_quote = [quote_re.sub("", des).strip() for des in des_no_nl_space]
            # 去除完全沒有的 element
            processed_des = [des for des in des_no_nl_space_quote if len(des) != 0]
            final_description = "".join(processed_des)
            return final_description
        return ""
    
    @staticmethod
    def clean_comment(x, col):
        """清理comment

        Args:
            x (_type_): comment
            col (_type_): 要命名的欄位

        Returns:
            _type_: _description_
        """
        if x != []:
            sigle_comment_df = pd.json_normalize(x)
            sigle_comment_df["all_comment"] = sigle_comment_df['title'] + " " + sigle_comment_df['description']
            return sigle_comment_df[col].dropna().tolist()
        return []
    
    def remove_punc(self, df):
        """去除標點符號

        Args:
            df (DataFrame): 須處理的df

        Returns:
            DataFrame: 須處理的df
        """
        df[self.text_col] = df[self.text_col].apply(lambda x: x.str.replace('[^\w\s\"\']', ''))
        df[self.text_col] = df[self.text_col].apply(lambda x: x.str.replace(r"(\n|\s)", ''))
        return df
    
    

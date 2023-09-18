import time
import random
import requests
from tqdm import tqdm
from typing import Dict
from lib.Basic_crawler_setting import Cralwer_Setting

class Course_Cate(Cralwer_Setting):
    def __init__(self):
        # 取得 random user agents
        self.user_agents = self.get_user_agent
        self.my_headers = {
            "user-agent": self.user_agents
            }        

    def __call__(self) -> Dict[str, str]:
        """取得課程的類別

        Returns:
            Dict[str, str]: {名稱:unique 名稱}
        """
        time.sleep(random.randint(3, 6))
        url = "https://api.hahow.in/api/groups/index"

        resp = requests.get(
            url, 
            headers=self.my_headers,
            )
        
        home_json = resp.json()['groups']
        home_category = {item['title']:item['uniquename'] for item in home_json}

        return home_category

class Course_Need_Page(Cralwer_Setting):
    def __init__(self, course_categories):
        # 取得 random user agents
        self.user_agents = self.get_user_agent
        self.my_headers = {
            "user-agent": self.user_agents
            }  
        self.course_categories = course_categories
    
    def __call__(self) -> Dict[str, Dict[str, str]]:
        """爬取各種類型課程、要爬的頁數

        Returns:
            Dict[str, Dict[str, str]]: 回傳每種類課程的 unique name 跟其所需要爬的頁數
        """
        for name, uniquename in tqdm(self.course_categories.items()):
            time.sleep(random.randint(2, 5))

            # get api page
            url = f"https://api.hahow.in/api/products/search?category=COURSE&groups={uniquename}&limit=100&page=0&sort=NUM_OF_STUDENT"
            resp = requests.get(
                url, 
                headers=self.my_headers,
                )
            
            # count need crawler pages
            metadata_json = resp.json()['_metadata']
            need_crawler_page = metadata_json['count'] // metadata_json['limit'] + 1

            # update dict
            self.course_categories[name] = {
                "unique_name":uniquename,
                "crawler_page":need_crawler_page
                }
            
        return self.course_categories
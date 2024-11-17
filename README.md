# Readme 
In this project,we mainly use the **random forest** after comparing with other model,which is include Linear Regression,Decision Tree,Adaboost, Gradient Boosting.
After compared with it's score,we choose the best one to be our project's main model.
The picture is the random forest's schematic diagram:
![Random-Forest-Algorithm](https://github.com/user-attachments/assets/47a4f1bf-e351-45dd-9f06-3ff11f1e4c1e)
We will explore this principle in more detail in a later section.
# Dataset
## Form
We choose the website https://sz.58.com/ershouche/?PGTID=0d30001d-0000-4622-24da-a55ad27728f0&ClickID=57 and crawl the data.
The below code is the method how we crawl it:
```
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}
base_url = 'https://www.che168.com/china/list/#pvareaid=100943/'
get_page_data(base_url)
```

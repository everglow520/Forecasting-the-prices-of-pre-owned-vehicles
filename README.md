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
def parse_page(soup):
     cars = soup.find_all('li', class_='cards-li')
     for car in cars:
        try:
            title = car.find('h4', class_='card-name').get_text(strip=True)
            price = car.find('span', class_='pirce').get_text(strip=True)
            str = car.find('p', class_='cards-unit').get_text(strip=True)
            box = str.split('')
            mileage = box[0]
            year = box[1]
            city = box[2]
            
            car_data.append([title, price, mileage, year, city])
        except AttributeError:
            continue
```
Then we get 3641 rows and 8 columns data.
## preprocessing
In this project,we mainly use **one-hot encoder** and **standard scaler** to preprocessing our data to make it more comprehendible for the computer.
And what is it? Why we use it?
The reason is that the computer can't comprehend the data we crwal directly.We must convert it to computer language.
### one-hot encoder
One-hot encoding, also known as one-bit effective encoding, mainly uses N-bit status registers to encode N states, each state is composed of its own independent register bits, and only One bit is valid at any time.
It is a representation of categorical variables as binary vectors. This first requires mapping class values to integer values. Each integer value is then represented as a binary vector that is zero except for the index of the integer, which is labeled 1.




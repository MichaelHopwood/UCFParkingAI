import requests
import pandas as pd

base_year_url = "https://api.ucfgarages.com/all?year="
years = [2019,2020,2021]
namer = {'Garage A': 'A', 'Garage B': 'B', 'Garage C': 'C', 'Garage D': 'D', 'Garage H': 'H', 'Garage I': 'I', 'Garage Libra': 'Libra'}

def get_data():
    formatted_data = []
    for y in years:
        data = requests.get(base_year_url + str(y)).json()
        df = pd.DataFrame(data["data"])
        for ind,row in df.iterrows():
            d = row['garages']
            formatted_dict = {}
            for garage_dict in d:
                GarageName = namer[garage_dict['name']]
                keys = list(garage_dict.keys())
                keys.remove('name')
                for key in keys:
                    formatted_dict[GarageName+'_'+key] = garage_dict[key]
            formatted_dict['date'] = row['date']
            formatted_data.append(formatted_dict)
    return pd.DataFrame(formatted_data)


if __name__ == "__main__":
    df = get_data()
    print(df)

    #get_citation_data()
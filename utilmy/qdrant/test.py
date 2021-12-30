import json
import numpy as np
from dbvector import ClientQdrant




def get_data(filename="startups.json"):
    data =[]
    with open(filename) as f:
        for jsonObj in f:
            dd = {}
            dd['payload'] = json.loads(jsonObj)
            dd['vector']  = np.random.rand(768)
            data.append(dd)
    return data



def main():
    print("Connecting to Qdrant")
    client = Client('localhost', 6333)
    client.connect()


    print("Creating a new table")
    client.create_collection('startups')

    #### Insert data 
    item_list = get_data()
    client.put_multi(item_list)
    client.table_view()


    query_list = [ {"vector": np.random.rand(768), 'filter' :{  'category_id': 90} } ]
    client.get_multi(query_list)


if __name__ == "__main__":
	main()



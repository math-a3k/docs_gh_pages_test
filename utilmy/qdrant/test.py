"""


https://blog.qdrant.tech/neural-search-tutorial-3f034ab13adc



from qdrant_openapi_client.models.models import Filter

    ...

    city_of_interest = "Berlin"

    # Define a filter for cities
    city_filter = Filter(**{
        "must": [{
            "key": "city", # We store city information in a field of the same name 
            "match": { # This condition checks if payload field have requested value
                "keyword": city_of_interest
            }
        }]
    })

    search_result = self.qdrant_client.search(
        collection_name=self.collection_name,
        query_vector=vector,
        query_filter=city_filter,
        top=5
    )
    ...


"""
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



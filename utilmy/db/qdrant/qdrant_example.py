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



docker pull generall/qdrant

docker run -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  generall/qdrant



  pip install qdrant-client



import numpy as np
import json

fd = open('./startups.json')

# payload is now an iterator over startup data
payload = map(json.loads, fd)

# Here we load all vectors into memory, numpy array works as iterable for itself.
# Other option would be to use Mmap, if we don't want to load all data into RAM
vectors = np.load('./startup_vectors.npy')

# And the final step - data uploading
qdrant_client.upload_collection(
  collection_name='startups',
  vectors=vectors,
  payload=payload,
  ids=None,  # Vector ids will be assigned automatically
  batch_size=256  # How many vectors will be uploaded in a single request?
)






 File: neural_searcher.py

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class NeuralSearcher:

    def __init__(self, collection_name):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cpu')
        # initialize Qdrant client
        self.qdrant_client = QdrantClient(host='localhost', port=6333)

    # The search function looks as simple as possible:

    def search(self, text: str):
        # Convert text query into vector
        vector = self.model.encode(text)

        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,  # We don't want any filters for now
            top=5  # 5 the most closest results is enough
        )

        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # In this function we are interested in payload only
        payloads = [payload for point, payload in search_result]
        return payloads



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





# File: service.py

from fastapi import FastAPI

# That is the file where NeuralSearcher is stored
from neural_searcher import NeuralSearcher

app = FastAPI()

# Create an instance of the neural searcher
neural_searcher = NeuralSearcher(collection_name='startups')

@app.get("/api/search")
def search_startup(q: str):
    return {
        "result": neural_searcher.search(text=q)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





    
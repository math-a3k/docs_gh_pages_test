#!/usr/bin/env python
from qdrant_client import QdrantClient
import numpy as np
import json
import os
from tempfile import mkdtemp
import uuid
import random

from qdrant_openapi_client.models.models import PointInsertOperationsAnyOf1, PointOperationsAnyOf, PointStruct

class Client:
	def __init__(self, host = 'localhost', port = 6333, table='default'):
		self.PORT = port
		self.HOST = host
		self.qdrant_client = None
		self.table = table
		#self.DIM = 768
		#self.NUM_VECTORS = 1_000


	def connect(self, table):
		self.qdrant_client = QdrantClient(host=self.HOST, port=self.PORT)
		self.collection_name = table


	def table_create(self, table, vector_size=768):
		self.qdrant_client.recreate_collection(collection_name=table, vector_size=vector_size)

	def table_view(self, table):
		collection_info = self.qdrant_client.http.collections_api.get_collection(table)
		return collection_info.dict()


	def get_multi(self, vect_list, query_filter=None, topk=5):
		topk_filtered = self.qdrant_client.search(
					        collection_name=self.collection_name, 
					        query_vector=vect_list,
					        query_filter=query_filter,
					        append_payload=True,
					        top=topk)
        return topk_filtered


	def put_multi(self, item_list):
		nitems = len(item_list)
		for i, ddict in enumerate(item_list):
			vector  = ddict['vector']
            payload = dddict['payload']  
            pt = PointStruct(id= i, payload= payload, vector=  vector )
            pt_list.append(pt)

            if len(pt_list) < 10 and i < nitems-1 : continue
			self.qdrant_client.http.points_api.update_points( name=self.table, wait=True, 
		                collection_update_operations=PointOperationsAnyOf(
		                upsert_points= PointInsertOperationsAnyOf1(
		                points= pt_list)) 
		            )

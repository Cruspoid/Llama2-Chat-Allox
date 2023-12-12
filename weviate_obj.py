import weaviate

client = weaviate.Client("https://test-1-rth920qm.weaviate.network")  
some_objects = client.data_object.get()
print(some_objects)
import weaviate


YOUR_OPENAI_KEY = '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'
YOUR_WEAVIATE_CLUSTER = 'https://allox-22-xo7y4fwt.weaviate.network'

WEAVIATE_URL = 'https://allox-22-xo7y4fwt.weaviate.network'
client = weaviate.Client(url=WEAVIATE_URL,additional_headers={"X-OpenAI-Api-Key": YOUR_OPENAI_KEY},startup_period=10)
client.schema.delete_all()
client.schema.get()
schema = {
    "classes": [
        {
            "class": "Chatbot",
            "description": "Documents for chatbot",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {"text2vec-openai": {"model": "ada", "type": "text"}},
            "properties": [
                {
                    "dataType": ["text"],
                    "description": "The content of the paragraph",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "content",
                },
            ],
        },
    ]
 }

client.schema.create(schema)
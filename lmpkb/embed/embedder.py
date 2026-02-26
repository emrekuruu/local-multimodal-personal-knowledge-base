import voyageai
from PIL.Image import Image
from dotenv import load_dotenv

load_dotenv()

class ImageEmbedder():

    def __init__(self):
        self.client = voyageai.Client()
        self.model_name = "voyage-multimodal-3.5"

    def embed_document(self, document) -> list[float]:
        result = self.client.multimodal_embed(
            inputs=[[document]],
            model=self.model_name,
            input_type="document"
        )
        return result.embeddings[0]

    def embed_query(self, query: str) -> list[float]:
        result = self.client.multimodal_embed(
            inputs=[[query]],
            model=self.model_name,
            input_type="query"
        )
        return result.embeddings[0]


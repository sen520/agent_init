import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, List
from sentence_transformers import SentenceTransformer

app = FastAPI()


class EmbData(BaseModel):
    texts: Union[str, List[str]] = None


model = SentenceTransformer("./models/bge-small-zh-v1.5")


@app.post("/emb")
async def emb(emb_body: EmbData):
    embeddings = model.encode(emb_body.texts).tolist()
    return {"embeddings": embeddings}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)

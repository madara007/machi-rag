from langchain_community.document_loaders import PyMuPDFLoader#
loader = PyMuPDFLoader(r"C:\Users\masar\Desktop\machi-rag\backend\zentai_kiban_c.pdf")
docs = loader.load()
from dataclasses import dataclass
# =========== Document ================

@dataclass
class doc:
  text: str
  meta: dict[str, any]

docs: list[doc] = []
from langchain_community.document_loaders import PyMuPDFLoader
loader: list  = PyMuPDFLoader(file_path=r"C:\Users\masar\Desktop\machi-rag\backend\zentai_kiban_c.pdf").load()
for d in loader:
  docs.append(doc(text=d.page_content.strip(), meta={"page": d.metadata["page"], "source": d.metadata["source"]}))

print(type("hoge".strip()))



#print(loader.load()[0])

#print(loader.load()[0].metadata["page"])


#from sentence_transformers import SentenceTransformer
#
#
#
#model_name = "intfloat/multilingual-e5-base"
#model_path = f"path_to_model/{model_name}"
#model = SentenceTransformer(model_path)
#embedding = model.encode('ベクトル化したい文字列')
#
#print(embedding)
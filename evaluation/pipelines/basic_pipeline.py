from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from overrides import overrides

from evaluation.pipelines.pipeline import Pipeline


class BasicPipeline(Pipeline):
    retrieval_chain: Runnable

    @overrides
    def __init__(self, llm: LanguageModelLike, prompt_template: PromptTemplate,
                 vector_db_retriever: VectorStoreRetriever) -> None:
        combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)

        self.retrieval_chain = create_retrieval_chain(vector_db_retriever, combine_docs_chain)

    @overrides
    def query(self, query: str) -> str:
        return self.retrieval_chain.invoke({"input": query})

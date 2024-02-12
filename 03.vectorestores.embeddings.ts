import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { similarity } from "ml-distance";

import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { 
    RecursiveCharacterTextSplitter
} from "langchain/text_splitter";

import { MemoryVectorStore } from "langchain/vectorstores/memory";

// Vectorstores and embeddings
// Vectorstore ingestion

const embeddings = new OpenAIEmbeddings();
const response = await embeddings.embedQuery("This is some sample text");
// console.log('response - ', response)

const vector1 = await embeddings.embedQuery(
  "What are vectors useful for in machine learning?"
);
const unrelatedVector = await embeddings.embedQuery(
  "A group of parrots is called a pandemonium."
);

const response1 = similarity.cosine(vector1, unrelatedVector);
console.log('response1 - ', response1)

const similarVector = await embeddings.embedQuery(
  "Vectors are representations of information."
);

const response2 = similarity.cosine(vector1, similarVector);
console.log('response2 - ', response2)

const loader = new PDFLoader("eBook.pdf");

const rawCS229Docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 128,
  chunkOverlap: 0,
});

const splitDocs = await splitter.splitDocuments(rawCS229Docs)


// in memory vector store 
const vectorstore = new MemoryVectorStore(embeddings);

await vectorstore.addDocuments(splitDocs);

const retrievedDocs = await vectorstore.similaritySearch(
  "What are microfrontends?", 
  6
);

const pageContents = retrievedDocs.map(doc => doc.pageContent);

console.log('pageContents: ', pageContents);

//Retrievers

const retriever = vectorstore.asRetriever();
const retrieverResponse = await retriever.invoke("What is deep learning?")
console.log('retrieverResponse - ', retrieverResponse)
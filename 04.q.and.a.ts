import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RunnableSequence } from "@langchain/core/runnables";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableMap } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";

// #1 - Load the PDF and Split Chunks
const loader = new PDFLoader("eBook.pdf");
const eBookPDF = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1536,
  chunkOverlap: 128,
});

const splitDocs = await splitter.splitDocuments(eBookPDF);

// #2 - Initialize VectorStore - In-Memory version
// TODO: Enahnce this to connect to a real VectorStore
const embeddings = new OpenAIEmbeddings();
const vectorstore = new MemoryVectorStore(embeddings);

await vectorstore.addDocuments(splitDocs);

const retriever = vectorstore.asRetriever();

// #3  Document retrieval in a chain
const convertDocsToString = (documents: Document[]): string => {
  return documents
    .map((document) => {
      return `<doc>\n${document.pageContent}\n</doc>`;
    })
    .join("\n");
};

const documentRetrievalChain = RunnableSequence.from([
  (input) => input.question,
  retriever,
  convertDocsToString,
]);

const results = await documentRetrievalChain.invoke({
  question: "What are the prerequisites for this course?",
});
// console.log(results);

// #4 Formatting Response using a Chat Template:

const TEMPLATE_STRING = `You are an experienced researcher, 
expert at interpreting and answering questions based on provided sources.
Using the provided context, answer the user's question 
to the best of your ability using only the resources provided. 
Be verbose!

<context>

{context}

</context>

Now, answer this question using the above context:

{question}`;

const answerGenerationPrompt = ChatPromptTemplate.fromTemplate(TEMPLATE_STRING);

const runnableMap = RunnableMap.from({
  context: documentRetrievalChain,
  question: (input) => input.question,
});

// const runnableMapRespones = await runnableMap.invoke({
//     question: "What are the prerequisites for learning micro-frontends?"
// })

// console.log('runnableMapRespones - ', runnableMapRespones)

// #5 Agumented Generation :
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-1106",
});

const retrievalChain = RunnableSequence.from([
  {
    context: documentRetrievalChain,
    question: (input) => input.question,
  },
  answerGenerationPrompt,
  model,
  new StringOutputParser(),
]);

const answer = await retrievalChain.invoke({
  question: "What are the prerequisites for this course?"
});

console.log("What are the prerequisites for this course?")
console.log(answer);
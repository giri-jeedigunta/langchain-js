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
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { RunnablePassthrough } from "@langchain/core/runnables";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";


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

// const answer = await retrievalChain.invoke({
//   question: "What are the prerequisites for this course?"
// });

// console.log("What are the prerequisites for this course?")
// console.log('original q and a - ', answer);


// WE do not have history 

// const followupAnswer = await retrievalChain.invoke({
//   question: "Can you list them in bullet point form?"
// });

// console.log("Can you list them in bullet point form?");
// console.log('followupAnswer - ', followupAnswer);


// console.log("Can you list them in bullet point form?");
// const docs = await documentRetrievalChain.invoke({
//   question: "Can you list them in bullet point form?"
// });

// console.log(' documentRetrievalChain - ', docs);




// #6 Adding history to make it more conversational: 
const REPHRASE_QUESTION_SYSTEM_TEMPLATE = 
  `Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.`;

const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human", 
    "Rephrase the following question as a standalone question:\n{question}"
  ],
]);


const rephraseQuestionChain = RunnableSequence.from([
  rephraseQuestionChainPrompt,
  new ChatOpenAI({ temperature: 0.1, modelName: "gpt-3.5-turbo-1106" }),
  new StringOutputParser(),
])

const originalQuestion = "What are the prerequisites for this course?";

const originalAnswer1 = await retrievalChain.invoke({
  question: originalQuestion
});

console.log('originalAnswer - ', originalAnswer1);

const chatHistory = [
  new HumanMessage(originalQuestion),
  new AIMessage(originalAnswer1),
];

await rephraseQuestionChain.invoke({
  question: "Can you list them in bullet point form?",
  history: chatHistory,
});


const ANSWER_CHAIN_SYSTEM_TEMPLATE = `You are an experienced researcher, 
expert at interpreting and answering questions based on provided sources.
Using the below provided context and chat history, 
answer the user's question to the best of 
your ability 
using only the resources provided. Be verbose!

<context>
{context}
</context>`;

const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human", 
    "Now, answer this question using the previous context and chat history:\n{standalone_question}"
  ]
]);


// await answerGenerationChainPrompt.formatMessages({
//   context: "fake retrieved content",
//   standalone_question: "Why is the sky blue?",
//   history: [
//     new HumanMessage("How are you?"),
//     new AIMessage("Fine, thank you!")
//   ]
// });

const conversationalRetrievalChain = RunnableSequence.from([
  RunnablePassthrough.assign({
    standalone_question: rephraseQuestionChain,
  }),
  RunnablePassthrough.assign({
    context: documentRetrievalChain,
  }),
  answerGenerationChainPrompt,
  new ChatOpenAI({ modelName: "gpt-3.5-turbo" }),
  new StringOutputParser(),
]);

const messageHistory = new ChatMessageHistory();

const finalRetrievalChain = new RunnableWithMessageHistory({
  runnable: conversationalRetrievalChain,
  getMessageHistory: (_sessionId) => messageHistory,
  historyMessagesKey: "history",
  inputMessagesKey: "question",
});

// const originalQuestion = "What are the prerequisites for this course?";

const originalAnswer = await finalRetrievalChain.invoke({
  question: originalQuestion,
}, {
  configurable: { sessionId: "test_1" }
});

const finalResult = await finalRetrievalChain.invoke({
  question: "Can you list them in bullet point form?",
}, {
  configurable: { sessionId: "test_1" }
});

console.log(finalResult);
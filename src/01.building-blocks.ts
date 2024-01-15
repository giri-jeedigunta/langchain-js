
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { 
  SystemMessagePromptTemplate, 
  HumanMessagePromptTemplate 
} from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";


//Language model
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-1106", 
  openAIApiKey: '' // your openai api key
});

const langModel = await model.invoke([
  new HumanMessage("Tell me a joke.")
]);

console.log('langModel', langModel);



//## Prompt template
const prompt = ChatPromptTemplate.fromTemplate(
  `What are three good names for a company that makes {product}?`
)

const prompt1 = await prompt.format({
  product: "samosas"
});

console.log('prompt1 - ', prompt1)

const promptFromMessages = ChatPromptTemplate.fromMessages([
  SystemMessagePromptTemplate.fromTemplate(
    "You are an expert at picking company names."
  ),
  HumanMessagePromptTemplate.fromTemplate(
    "What are three good names for a company that makes {product}?"
  )
]);

const prompt2 = await promptFromMessages.formatMessages({
    product: "shiny objects"
});

console.log('prompt2 - ', prompt2)


//LangChain Expression Language (LCEL)
const chain = prompt.pipe(model);

const prompt3 = await chain.invoke({
  product: "colorful socks"
});

console.log('prompt3 - ', prompt3)

//Output parser



const outputParser = new StringOutputParser();

const nameGenerationChain = prompt.pipe(model).pipe(outputParser);

const response4 = await nameGenerationChain.invoke({
  product: "fancy cookies"
});

console.log('response4 - ', response4)


//RunnableSequence

const nameGenerationChain1 = RunnableSequence.from([
  prompt,
  model,
  outputParser
])

const response5 = await nameGenerationChain1.invoke({
  product: "fancy cookies"
});

console.log('response5 - ', response5);


//Stream
const stream = await nameGenerationChain.stream({
  product: "really cool robots",
});

for await (const chunk of stream) {
    console.log(chunk);
}

//batch

const inputs = [
  { product: "large calculators" },
  { product: "alpaca wool sweaters" }
];

const batchresponse = await nameGenerationChain.batch(inputs);

console.log('batchresponse - ', batchresponse)
import { GithubRepoLoader } from "langchain/document_loaders/web/github";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

const loader = new GithubRepoLoader(
  "https://github.com/langchain-ai/langchainjs",
  { recursive: false, ignorePaths: ["*.md", "yarn.lock"] }
);

const docs = await loader.load();

console.log(docs.slice(0, 3));


const pdfLoader = new PDFLoader("eBook.pdf");
const rawCS229Docs = await pdfLoader.load();

console.log(rawCS229Docs.slice(0, 5));
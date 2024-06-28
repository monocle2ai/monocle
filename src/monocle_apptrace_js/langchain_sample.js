
const { setupMonocle } = require("./instrumentations.js")
const exporter = setupMonocle()
const { ChatOpenAI, OpenAIEmbeddings } = require("@langchain/openai")

const { HNSWLib } = require("@langchain/community/vectorstores/hnswlib")
const { formatDocumentsAsString } = require("langchain/util/document");
const { PromptTemplate } = require("@langchain/core/prompts");
const {
  RunnableSequence,
  RunnablePassthrough,
} = require("@langchain/core/runnables");
const { StringOutputParser } = require("@langchain/core/output_parsers");

const model = new ChatOpenAI({});

HNSWLib.fromTexts(
  ["mitochondria is the powerhouse of the cell"],
  [{ id: 1 }],
  new OpenAIEmbeddings()
).then((vectorStore) => {
  const retriever = vectorStore.asRetriever();

  const prompt =
    PromptTemplate.fromTemplate(`Answer the question based only on the following context:
{context}

Question: {question}`);

  const chain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocumentsAsString),
      question: new RunnablePassthrough(),
    },
    prompt,
    model,
    new StringOutputParser(),
  ]);

  chain.invoke("What is the powerhouse of the cell?").then(
    (res) => {
      console.log("result:" + res)
    }
  )
})
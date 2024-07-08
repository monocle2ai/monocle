const { langchainPackages } = require("../langchain/packages")
const { llamaindexPackages } = require("../llamaindex/packages")

exports.combinedPackages = [
    langchainPackages,
    llamaindexPackages
]
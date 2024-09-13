

import os

from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.retrievers.in_memory.embedding_retriever import (
    InMemoryDocumentStore,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from monocle_apptrace.instrumentor import setup_monocle_telemetry
from monocle_apptrace.wrap_common import llm_wrapper
from monocle_apptrace.wrapper import WrapperMethod


def haystack_app():

    setup_monocle_telemetry(
            workflow_name="haystack_app_1",
            span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
            wrapper_methods=[
                WrapperMethod(
                    package="haystack.components.retrievers.in_memory",
                    object_name="InMemoryEmbeddingRetriever",
                    method="run",
                    span_name="haystack.retriever",
                    wrapper=llm_wrapper),

            ])

    # initialize
    api_key = os.getenv("OPENAI_API_KEY")
    generator = OpenAIGenerator(
        api_key=Secret.from_token(api_key), model="gpt-3.5-turbo"
    )

    # initialize document store, load data and store in document store
    document_store = InMemoryDocumentStore()
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(docs_with_embeddings["documents"])

    # embedder to embed user query
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # get relevant documents from embedded query
    retriever = InMemoryEmbeddingRetriever(document_store)

    # use documents to build the prompt
    template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    prompt_builder = PromptBuilder(template=template)

    basic_rag_pipeline = Pipeline()
    # Add components to your pipeline
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", generator)

    # Now, connect the components to each other
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder", "llm")

    question = "What does Rhodes Statue look like?"

    response = basic_rag_pipeline.run(
        {"text_embedder": {"text": question}, "prompt_builder": {"question": question}}
    )

    print(response["llm"]["replies"][0])


haystack_app()

# {
#     "name": "haystack.retriever",
#     "context": {
#         "trace_id": "0xee46d79f09814a638517cd05f26ab21f",
#         "span_id": "0x79a2f252baf2b721",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xad97601a1ba8388d",
#     "start_time": "2024-09-13T11:58:06.064937Z",
#     "end_time": "2024-09-13T11:58:06.074810Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "type": "vector_store",
#         "provider_name": "InMemoryDocumentStore",
#         "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "haystack_app_1"
#         },
#         "schema_url": ""
#     }
# }
# The Colossus of Rhodes depicted the Greek sun-god Helios and was approximately 33 meters (108 feet) tall. The head likely had curly hair with spikes of bronze or silver flame radiating, similar to the images found on contemporary Rhodian coins. The statue's exact appearance is unknown, but it was described as having a standard rendering of the time.
# {
#     "name": "haystack.openai",
#     "context": {
#         "trace_id": "0xee46d79f09814a638517cd05f26ab21f",
#         "span_id": "0xf57731cd5cfe3f8e",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xad97601a1ba8388d",
#     "start_time": "2024-09-13T11:58:06.075424Z",
#     "end_time": "2024-09-13T11:58:08.299323Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "llm_input": "\n    Given the following information, answer the question.\n\n    Context:\n    \n        Within it, too, are to be seen large masses of rock, by the weight of which the artist steadied it while erecting it.[22][23]\nDestruction of the remains[edit]\nThe ultimate fate of the remains of the statue is uncertain. Rhodes has two serious earthquakes per century, owing to its location on the seismically unstable Hellenic Arc. Pausanias tells us, writing ca. 174, how the city was so devastated by an earthquake that the Sibyl oracle foretelling its destruction was considered fulfilled.[24] This means the statue could not have survived for long if it was ever repaired. By the 4th century Rhodes was Christianized, meaning any further maintenance or rebuilding, if there ever was any before, on an ancient pagan statue is unlikely. The metal would have likely been used for coins and maybe also tools by the time of the Arab wars, especially during earlier conflicts such as the Sassanian wars.[9]\nThe onset of Islamic naval incursions against the Byzantine empire gave rise to a dramatic account of what became of the Colossus. \n    \n        Construction[edit]\nTimeline and map of the Seven Wonders of the Ancient World, including the Colossus of Rhodes\nConstruction began in 292\u00a0BC. Ancient accounts, which differ to some degree, describe the structure as being built with iron tie bars to which brass plates were fixed to form the skin. The interior of the structure, which stood on a 15-metre-high (49-foot) white marble pedestal near the Rhodes harbour entrance, was then filled with stone blocks as construction progressed.[14] Other sources place the Colossus on a breakwater in the harbour. According to most contemporary descriptions, the statue itself was about 70 cubits, or 32 metres (105 feet) tall.[15] Much of the iron and bronze was reforged from the various weapons Demetrius's army left behind, and the abandoned second siege tower may have been used for scaffolding around the lower levels during construction.\n\n    \n        The Colossus of Rhodes (Ancient Greek: \u1f41 \u039a\u03bf\u03bb\u03bf\u03c3\u03c3\u1f78\u03c2 \u1fec\u03cc\u03b4\u03b9\u03bf\u03c2, romanized:\u00a0ho Koloss\u00f2s Rh\u00f3dios Greek: \u039a\u03bf\u03bb\u03bf\u03c3\u03c3\u03cc\u03c2 \u03c4\u03b7\u03c2 \u03a1\u03cc\u03b4\u03bf\u03c5, romanized:\u00a0Koloss\u00f3s tes Rh\u00f3dou)[a] was a statue of the Greek sun-god Helios, erected in the city of Rhodes, on the Greek island of the same name, by Chares of Lindos in 280\u00a0BC. One of the Seven Wonders of the Ancient World, it was constructed to celebrate the successful defence of Rhodes city against an attack by Demetrius Poliorcetes, who had besieged it for a year with a large army and navy.\nAccording to most contemporary descriptions, the Colossus stood approximately 70 cubits, or 33 metres (108 feet) high \u2013 approximately the height of the modern Statue of Liberty from feet to crown \u2013 making it the tallest statue in the ancient world.[2] It collapsed during the earthquake of 226 BC, although parts of it were preserved. In accordance with a certain oracle, the Rhodians did not build it again.[3] John Malalas wrote that Hadrian in his reign re-erected the Colossus,[4] but he was mistaken.[5] According to the Suda, the Rhodians were called Colossaeans (\u039a\u03bf\u03bb\u03bf\u03c3\u03c3\u03b1\u03b5\u1fd6\u03c2), because they erected the statue on the island.\n    \n        Silver tetradrachm of Rhodes showing Helios and a rose (205\u2013190 BC, 13.48 g)\nWhile scholars do not know what the statue looked like, they do have a good idea of what the head and face looked like, as it was of a standard rendering at the time. The head would have had curly hair with evenly spaced spikes of bronze or silver flame radiating, similar to the images found on contemporary Rhodian coins.[29]\n\nPossible locations[edit]\nThe old harbour entrance from inner embankment. The Fortress of St Nicholas is on right\nWhile scholars generally agree that anecdotal depictions of the Colossus straddling the harbour's entry point have no historic or scientific basis,[29] the monument's actual location remains a matter of debate. As mentioned above the statue is thought locally to have stood where two pillars now stand at the Mandraki port entrance.\nThe floor of the Fortress of St Nicholas, near the harbour entrance, contains a circle of sandstone blocks of unknown origin or purpose. \n    \n        To you, O Sun, the people of Dorian Rhodes set up this bronze statue reaching to Olympus, when they had pacified the waves of war and crowned their city with the spoils taken from the enemy. Not only over the seas but also on land did they kindle the lovely torch of freedom and independence. For to the descendants of Herakles belongs dominion over sea and land.\nCollapse (226\u00a0BC)[edit]\nArtist's conception from the Grolier Society's 1911 Book of Knowledge\nFurther information: 226 BC Rhodes earthquake\nThe statue stood for 54 years until a 226\u00a0BC earthquake caused significant damage to large portions of Rhodes, including the harbour and commercial buildings, which were destroyed.[19] The statue snapped at the knees and fell over onto land. Ptolemy III offered to pay for the reconstruction of the statue, but the Oracle of Delphi made the Rhodians fear that they had offended Helios, and they declined to rebuild it.[citation needed]\n\nFallen state (226\u00a0BC to 653\u00a0AD)[edit]\nThe remains lay on the ground for over 800 years, and even broken, they were so impressive that many travelled to see them.\n\n    \n        [6]\nIn 653, an Arab force under Muslim general Muawiyah I conquered Rhodes, and according to the Chronicle of Theophanes the Confessor,[7] the statue was completely destroyed and the remains sold;[8] this account may be unreliable.[9]\nSince 2008, a series of as-yet-unrealized proposals to build a new Colossus at Rhodes Harbour have been announced, although the actual location of the original monument remains in dispute.[10][11]\n\nSiege of Rhodes[edit]\nMain article: Siege of Rhodes (305\u2013304\u00a0BC)\nIn the early fourth century BC, Rhodes, allied with Ptolemy I of Egypt, prevented a mass invasion staged by their common enemy, Antigonus I Monophthalmus.\nIn 304\u00a0BC a relief force of ships sent by Ptolemy arrived, and Demetrius (son of Antigonus) and his army abandoned the siege, leaving behind most of their siege equipment. To celebrate their victory, the Rhodians sold the equipment left behind for 300 talents[12] and decided to use the money to build a colossal statue of their patron god, Helios. Construction was left to the direction of Chares, a native of Lindos in Rhodes, who had been involved with large-scale statues before. His teacher, the sculptor Lysippos, had constructed a 22-metre-high (72-foot)[13] bronze statue of Zeus at Tarentum.\n\n\n    \n        Seeking to outdo their Athenian rivals, the Eleans employed sculptor Phidias, who had previously made the massive statue of Athena Parthenos in the Parthenon.[2]\nThe statue occupied half the width of the aisle of the temple built to house it. The geographer Strabo noted early in the 1st century BC that the statue gave \"the impression that if Zeus arose and stood erect he would unroof the temple.\"[3] The Zeus was a chryselephantine sculpture, made with ivory and gold panels on a wooden substructure. No copy in marble or bronze has survived, though there are recognizable but only approximate versions on coins of nearby Elis and on Roman coins and engraved gems.[4]\nThe 2nd-century AD geographer and traveler Pausanias left a detailed description: the statue was crowned with a sculpted wreath of olive sprays and wore a gilded robe made from glass and carved with animals and lilies. Its right hand held a small chryselephantine statue of crowned Nike, goddess of victory; its left a scepter inlaid with many metals, supporting an eagle. The throne featured painted figures and wrought images and was decorated with gold, precious stones, ebony, and ivory.\n    \n        [5] Zeus' golden sandals rested upon a footstool decorated with an Amazonomachy in relief. The passage underneath the throne was restricted by painted screens.[6]\nPausanias also recounts that the statue was kept constantly coated with olive oil to counter the harmful effect on the ivory caused by the \"marshiness\" of the Altis grove. The floor in front of the image was paved with black tiles and surrounded by a raised rim of marble to contain the oil.[7] This reservoir acted as a reflecting pool which doubled the apparent height of the statue.[8]\nAccording to the Roman historian Livy, the Roman general Aemilius Paullus (the victor over Macedon) saw the statue and \"was moved to his soul, as if he had seen the god in person\",[9] while the 1st-century\u00a0AD Greek orator Dio Chrysostom declared that a single glimpse of the statue would make a man forget all his earthly troubles.\n    \n        Also, the fallen statue would have blocked the harbour, and since the ancient Rhodians did not have the ability to remove the fallen statue from the harbour, it would not have remained visible on land for the next 800 years, as discussed above. Even neglecting these objections, the statue was made of bronze, and engineering analyses indicate that it could not have been built with its legs apart without collapsing under its own weight.[29]\nMany researchers have considered alternative positions for the statue which would have made it more feasible for actual construction by the ancients.[29][30] There is also no evidence that the statue held a torch aloft; the records simply say that after completion, the Rhodians kindled the \"torch of freedom\". A relief in a nearby temple shows Helios standing with one hand shielding his eyes, similar to the way a person shields their eyes when looking toward the sun, and it is quite possible that the colossus was constructed in the same pose.\n\n\n    \n        The Statue of Zeus at Olympia was a giant seated figure, about 12.4\u00a0m (41\u00a0ft) tall,[1] made by the Greek sculptor Phidias around 435 BC at the sanctuary of Olympia, Greece, and erected in the Temple of Zeus there.  Zeus is the sky and thunder god in ancient Greek religion, who rules as king of the gods of Mount Olympus.\nThe statue was a chryselephantine sculpture of ivory plates and gold panels on a wooden framework. Zeus sat on a painted cedarwood throne ornamented with ebony, ivory, gold, and precious stones. It was one of the Seven Wonders of the Ancient World.\nThe statue was lost and destroyed before the end of the 5th century AD, with conflicting accounts of the date and circumstances. Details of its form are known only from ancient Greek descriptions and representations on coins.\n\nCoin from Elis district in southern Greece illustrating the Olympian Zeus statue (Nordisk familjebok)\n\nHistory[edit]\nThe statue of Zeus was commissioned by the Eleans, custodians of the Olympic Games, in the latter half of the fifth century BC for their newly constructed Temple of Zeus.  \n    \n\n    Question: What does Rhodes Statue look like?\n    Answer:\n    ",
#         "model_name": "gpt-3.5-turbo",
#         "completion_tokens": 74,
#         "prompt_tokens": 2464,
#         "total_tokens": 2538
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "haystack_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "haystack_pipeline.workflow",
#     "context": {
#         "trace_id": "0xee46d79f09814a638517cd05f26ab21f",
#         "span_id": "0xad97601a1ba8388d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-09-13T11:58:06.053639Z",
#     "end_time": "2024-09-13T11:58:08.299411Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "input": "What does Rhodes Statue look like?",
#         "workflow_name": "haystack_app_1",
#         "workflow_type": "workflow.haystack",
#         "output": "The Colossus of Rhodes depicted the Greek sun-god Helios and was approximately 33 meters (108 feet) tall. The head likely had curly hair with spikes of bronze or silver flame radiating, similar to the images found on contemporary Rhodian coins. The statue's exact appearance is unknown, but it was described as having a standard rendering of the time."
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "haystack_app_1"
#         },
#         "schema_url": ""
#     }
# }

from sentence_transformers import SentenceTransformer, util


# Load the Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Business-related documents
documents = [
    "Businesses can improve customer satisfaction by providing fast and reliable customer support.",
    "Customer retention increases when companies provide personalized services and respond quickly to customer complaints.",
    "Companies can increase sales by offering discounts and special promotions to customers.",
    "Employee productivity improves when organizations provide proper training and development programs.",
    "Businesses can reduce operational costs by optimizing their supply chain processes.",
    "Companies can build customer loyalty by offering rewards and loyalty programs.",
    "Effective digital marketing helps businesses reach a larger target audience.",
    "Businesses can improve their brand reputation by consistently delivering high-quality products.",
    "Companies can increase revenue by expanding their products into new markets.",
    "Providing multiple payment options can improve the customer purchasing experience.",
    "Businesses can improve employee satisfaction by creating a positive workplace environment.",
    "Data analysis helps companies make better business decisions.",
    "Companies can improve productivity by automating repetitive tasks.",
    "Businesses can attract new customers through social media marketing campaigns.",
    "Companies can reduce employee turnover by providing competitive salaries and career growth opportunities.",
    "Efficient inventory management helps businesses avoid product shortages and excess stock.",
    "Businesses can improve profitability by reducing unnecessary expenses.",
    "Companies can understand customer needs by collecting and analyzing customer feedback.",
    "Strong cybersecurity practices help businesses protect customer and company data.",
    "Businesses can improve their online presence by developing an easy-to-use website.",
    "Companies can increase customer engagement by sending personalized email campaigns.",
    "Effective leadership helps organizations achieve their business goals.",
    "Businesses can improve sales performance by training employees in communication and negotiation skills.",
    "Companies can make better financial decisions by regularly analyzing their revenue and expenses."
]


# Convert all documents into embeddings
document_embeddings = model.encode(
    documents,
    convert_to_tensor=True
)


# Five required search queries
queries = [
    "How can a company keep its customers loyal?",
    "How can businesses improve employee performance?",
    "What can companies do to increase sales?",
    "How can businesses reduce their operating costs?",
    "How can a company improve customer satisfaction?"
]


def search(query):
    """
    Perform semantic search and return the top 3 results.
    """

    # Convert the query into an embedding
    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    # Find the top 3 most similar documents
    search_results = util.semantic_search(
        query_embedding,
        document_embeddings,
        top_k=3
    )[0]

    print("\n" + "=" * 80)
    print("Query:", query)
    print("=" * 80)

    # Display the top 3 results
    for rank, result in enumerate(search_results, start=1):

        document_index = result["corpus_id"]
        similarity_score = result["score"]

        print(f"\n{rank}. Similarity Score: {similarity_score:.4f}")
        print(f"   Document: {documents[document_index]}")


# -------------------------------------------------------------------
# Part 1: Run the five required assignment queries
# -------------------------------------------------------------------

print("\n" + "#" * 80)
print("SEMANTIC SEARCH - BUSINESS DOCUMENTS")
print("#" * 80)

print("\nRunning the five required search queries...")

for query in queries:
    search(query)


# -------------------------------------------------------------------
# Part 2: Allow the user to enter their own search query
# -------------------------------------------------------------------

print("\n" + "#" * 80)
print("CUSTOM SEARCH")
print("#" * 80)

print("\nYou can now enter your own business-related query.")
print("Type 'exit' to stop the program.")

while True:

    user_query = input("\nEnter your search query: ").strip()

    if user_query.lower() == "exit":
        print("\nThank you for using the semantic search system!")
        break

    if user_query == "":
        print("Please enter a valid search query.")
        continue

    search(user_query)
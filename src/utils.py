from openai import AzureOpenAI
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import json


# ----------------- openai -----------------

api_key = "dd9dd719aeee4118bd601f4310318671"
endpoint = "https://oh-ai-openai-scu.openai.azure.com/"

client = AzureOpenAI(api_key=api_key, api_version="2023-05-15", azure_endpoint=endpoint)
deployment_name = "gpt-35-turbo"

# load demonstration data
demo_file = "/Users/Pengyun_Wang/Projects/HSBC/data/demonstration.json"
with open(demo_file, "r") as f:
    demo_data = json.load(f)


def construct_prompt(demo_data, query):
    """
    Construct a prompt for the model to generate a human-readable summary of the website structure.
    Few shot learning is used to provide the model with context about the task.
    """
    start_phrase = f"""
    Task: Use the following examples to generate a human-readable summary \
        of the website structure. Make sure to follow the examples and provide \
        a step by step descritption. Most importantly, make sure to provide a \
        stand-alone overall summary as the last paragraph.
    Prompt:
    Summarize the DOM data in human-readable way:
    """

    # add demo
    for _, val in demo_data.items():
        start_phrase += f"DOM: {val['web_data']}\n"
        start_phrase += f"Summary: {val['summary']}\n"
    # add query
    start_phrase += f"DOM: {query}\n"
    start_phrase += f"Summary:"

    return start_phrase


def generate_response(web_data):

    # construct prompt
    prompt = construct_prompt(demo_data, web_data)

    # generate response
    try:
        response = client.completions.create(
            model=deployment_name, prompt=prompt, temperature=0, max_tokens=1000
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(e)
        return None


def get_summary(response):
    output = response.split("\n")[-1]
    if output.startswith("Overall"):
        return output
    else:
        return None


# ----------------- web data -----------------

# # Function to compute the CSS style of an element
# def compute_style(driver, element_id):
#     # if no such element in the driver, return None
#     try:
#         target_element = driver.find_element(By.ID, element_id)
#     except:
#         return None
#     # Execute JavaScript to compute the computed style of the element
#     script = """
#     var element = arguments[0];
#     var computedStyle = window.getComputedStyle(element);
#     return {
#         'width': computedStyle.getPropertyValue('width'),
#         'height': computedStyle.getPropertyValue('height'),
#         'margin': computedStyle.getPropertyValue('margin'),
#         'padding': computedStyle.getPropertyValue('padding'),
#         'border': computedStyle.getPropertyValue('border'),
#         'position': computedStyle.getPropertyValue('position'),
#         'top': computedStyle.getPropertyValue('top'),
#         'left': computedStyle.getPropertyValue('left'),
#         'font-family': computedStyle.getPropertyValue('font-family'),
#         'font-size': computedStyle.getPropertyValue('font-size'),
#         'font-weight': computedStyle.getPropertyValue('font-weight'),
#         'color': computedStyle.getPropertyValue('color'),
#         'text-align': computedStyle.getPropertyValue('text-align'),
#         'display': computedStyle.getPropertyValue('display'),
#         'visibility': computedStyle.getPropertyValue('visibility'),
#         'box-sizing': computedStyle.getPropertyValue('box-sizing'),
#         'overflow': computedStyle.getPropertyValue('overflow')
#     };
#     """
#     style = driver.execute_script(script, target_element)
#     return style

DEPTH = 5
KEEP_ATTRS = [
    "class",
    "id",
    "role",
    "style",
    "width",
    "height",
    "content",
    "rel",
    "size",
    "type",
    "alt",
    "title",
]


# Function to recursively extract DOM structure and store in a string
def extract_dom_structure(element, indent=0):
    # Initialize an empty string to store the DOM structure
    dom_structure = ""

    # Get the tag name and attributes of the element
    name = element.name
    attrs = element.attrs

    # Compute CSS style for the element, so far not included
    # css_style = compute_style(driver, element_id=element.attrs.get('id'))

    # keep certain attributes
    keep_attrs = KEEP_ATTRS + [
        key for key, value in attrs.items() if key.startswith("data-")
    ]

    attrs = {
        key: value
        for key, value in attrs.items()
        if key in keep_attrs and len(value) < 100
    }  # to avoid too long value

    # Append the current element's information to the DOM structure string, script excluded
    if name != "script":
        dom_structure += "--" * indent + f"Tag: {name}, Attributes: {attrs}" + "\n"

    # Recursively call the function for each child element and append their structures
    for child in element.children:
        if child.name and indent < DEPTH:
            dom_structure += extract_dom_structure(child, indent + 1)

    # Return the accumulated DOM structure string
    return dom_structure


def get_web_data(url):
    """
    Return web DOM structure of the given URL

    """
    try:
        # Fetch the webpage content
        response = requests.get(url)
        html_content = response.content

        # Parse the HTML content
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract the DOM structure starting from the root element (<html>)
        html_element = soup.find("html")

        # Call the function to extract the DOM structure
        dom_data = extract_dom_structure(html_element)

        return dom_data
    except Exception as e:
        print(e)
        return None


# ----------------- embeddings -----------------


# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
model = AutoModel.from_pretrained("thenlper/gte-base")


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def generate_embeddings(input_texts):
    """
    Generate embeddings for the given input texts
    """
    batch_dict = tokenizer(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    embeddings = embeddings.detach().numpy()
    return embeddings


# ----------------- clustering -----------------


def train_cluster_model(n_clusters, embeddings):
    """
    Train a KMeans clustering model with the given number of clusters
    """
    kmeans = KMeans(n_clusters=n_clusters)
    pipeline = Pipeline([("scaling", StandardScaler()), ("clustering", kmeans)])
    pipeline.fit(embeddings)
    return pipeline

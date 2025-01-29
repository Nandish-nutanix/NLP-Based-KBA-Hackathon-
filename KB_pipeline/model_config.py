from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama
import json
import warnings
from json import JSONDecodeError

warnings.filterwarnings("ignore", category=DeprecationWarning)

def format_json_response(text):
    """Convert text response to proper JSON format."""
    try:
        return json.loads(text)
    except JSONDecodeError:
        try:
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except JSONDecodeError:
            print(f"Warning: Could not parse JSON response. Using default structure.\nResponse was: {text}")
            return {"error": "Could not parse response", "raw_text": text}

def load_faiss_vector_store(path, embedding_model_name="NovaSearch/stella_en_1.5B_v5"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def get_context(vector_store, query, k=5):
    """Retrieve the top k most relevant text chunks for the query."""
    docs = vector_store.similarity_search(query, k=k)
    return " ".join([doc.page_content for doc in docs])

def query_ollama_with_context(context, query, system_prompt=""):
    print(f"Querying Ollama with context for: {query}")
    response = ollama.chat(
        model="llama3.2:latest",
        messages=[
            {
                "role": "system", 
                "content": f"{system_prompt}\nIMPORTANT: Your response must be valid JSON. Do not include any explanatory text."
            },
            {
                "role": "user", 
                "content": f"Context: {context}\n\nQuestion: {query}\nResponse must be valid JSON."
            }
        ]
    )
    return response["message"]["content"]

def generate_system_configuration(vector_store, system_prompt):
    try:
        nodes_query = """Extract and return a JSON array of nodes with this exact structure:
        {
            "nodes": [
                {
                    "node_1": {
                        "ip": "x.x.x.x",
                        "username": "nutanix",
                        "description": "Primary Controller VM"
                    }
                }
            ]
        }"""
        
        clusters_query = """Extract and return a JSON array of clusters with this exact structure:
        {
            "clusters": [
                {
                    "name": "Primary Cluster",
                    "description": "Cluster running NCC health checks"
                }
            ]
        }"""
        
        nodes_context = get_context(vector_store, nodes_query)
        clusters_context = get_context(vector_store, clusters_query)
        
        nodes_response = format_json_response(query_ollama_with_context(nodes_context, nodes_query, system_prompt))
        clusters_response = format_json_response(query_ollama_with_context(clusters_context, clusters_query, system_prompt))
        
        return {
            "nodes": nodes_response.get("nodes", []),
            "clusters": clusters_response.get("clusters", [])
        }
    except Exception as e:
        print(f"Error in generate_system_configuration: {e}")
        return {"nodes": [], "clusters": []}

def safe_json_query(vector_store, query, system_prompt, default_value=[]):
    """Safely execute a query and return JSON response with fallback to default value."""
    try:
        context = get_context(vector_store, query)
        response = query_ollama_with_context(context, query, system_prompt)
        return format_json_response(response)
    except Exception as e:
        print(f"Error in query execution: {e}")
        return default_value

def generate_actions(vector_store, system_prompt):
    query = """Extract and return a JSON array of actions with this exact structure:
    [
        {
            "action_name": "Action Name",
            "description": "Description of the action",
            "commands": ["command1", "command2"]
        }
    ]"""
    return safe_json_query(vector_store, query, system_prompt)

def generate_sequence(vector_store, system_prompt):
    query = """Extract and return a JSON array of sequences with this exact structure:
    [
        {
            "name": "Sequence Name",
            "description": "Description of sequence",
            "tasks": ["task1", "task2"]
        }
    ]"""
    return safe_json_query(vector_store, query, system_prompt)

def generate_dependencies(vector_store, system_prompt):
    query = """Extract and return a JSON array of dependencies with this exact structure:
    [
        {
            "task": "Task Name",
            "depends_on": ["dependency1", "dependency2"]
        }
    ]"""
    return safe_json_query(vector_store, query, system_prompt)

def generate_validations(vector_store, system_prompt):
    query = """Extract and return a JSON array of validations with this exact structure:
    [
        {
            "action": "Action Name",
            "expected_result": "Expected result description",
            "validation_steps": ["step1", "step2"]
        }
    ]"""
    return safe_json_query(vector_store, query, system_prompt)

def extract_ordered_commands(actions):
    """Extract commands from actions in sequential order."""
    ordered_commands = []
    try:
        for action in actions:
            if isinstance(action, dict) and "commands" in action:
                ordered_commands.extend(action["commands"])
    except Exception as e:
        print(f"Error extracting commands: {e}")
    return ordered_commands

def generate_metadata(vector_store, system_prompt):
    """Extract the KB article title and summary."""
    query = """Extract and return a JSON object with the KB article title and summary with this exact structure:
    {
        "title": "The KB article title or heading",
        "summary": "A brief summary of the main issue and resolution"
    }"""
    return safe_json_query(vector_store, query, system_prompt, default_value={"title": "N/A", "summary": "N/A"})

def generate_config(vector_store):
    system_prompt = """You are an expert assistant extracting structured information from technical documents.
    Return only valid JSON formatted responses that exactly match the structure requested in the question.
    Do not include any explanatory text or markdown formatting."""
    
    try:
        config = {
            "metadata": generate_metadata(vector_store, system_prompt),
            "system_configuration": generate_system_configuration(vector_store, system_prompt),
            "actions": generate_actions(vector_store, system_prompt),
            "sequence": generate_sequence(vector_store, system_prompt),
            "dependencies": generate_dependencies(vector_store, system_prompt),
            "validations": generate_validations(vector_store, system_prompt),
        }
        
        config["commands"] = {
            "ordered_commands": extract_ordered_commands(config["actions"])
        }
        
        return config
    except Exception as e:
        print(f"Error generating config: {e}")
        return {}

def save_config_to_json(config, output_file):
    try:
        with open(output_file, "w") as json_file:
            json.dump(config, json_file, indent=4)
        print(f"Successfully saved configuration to {output_file}")
    except Exception as e:
        print(f"Error saving config to JSON: {e}")

if __name__ == "__main__":
    try:
        faiss_path = "KB-2_vector_store.faiss"
        output_config_file = "/Users/nandish.chokshi/Downloads/NLP-Based-KBA-Hackathon-/execution/KB-2_config.json"
        vector_store = load_faiss_vector_store(faiss_path)
        config = generate_config(vector_store)
        save_config_to_json(config, output_config_file)

    except Exception as e:
        print(f"Error in main execution: {e}")
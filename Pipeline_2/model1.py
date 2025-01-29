import os
import yaml
import PyPDF2
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datetime import date
import logging
import spacy
import torch

class KBArticleConverter:
    def __init__(self, pdf_path, output_dir="./output_yaml", log_file="conversion.log"):
        """Initialize the converter with spaCy and LLM models."""
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger()

        try:
            # Load spaCy model
            self.logger.info("Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_sm")
            
            # Load the LLM model
            self.logger.info("Loading LLM model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                token=os.getenv('HUGGINGFACE_TOKEN')
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                torch_dtype=torch.float16,
                token=os.getenv('HUGGINGFACE_TOKEN')
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.95
            )
            
            self.logger.info("All models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    def extract_system_configuration(self, text):
        """Extract system configuration using LLM."""
        try:
            prompt = """Based on the following text, extract the system configuration details including:
            - Nodes (with IP, username, and description)
            - Clusters (with name and description)
            Format the response as a structured list.
            
            Text:
            {text}
            """
            
            config_text = self.extract_llm_section(text, prompt)
            return self._parse_system_config(config_text)
        except Exception as e:
            self.logger.error(f"Error extracting system configuration: {str(e)}")
            raise

    def extract_inputs_arguments(self, text):
        """Extract inputs and arguments using both spaCy and LLM."""
        try:
            # First pass with spaCy
            doc = self.nlp(text)
            inputs_args = []
            
            # Look for command parameters and arguments
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ in ["dobj", "pobj"] and token.head.lemma_ in ["input", "require", "accept", "need", "parameter", "argument"]:
                        command_context = self._get_command_context(token.sent.text)
                        inputs_args.append({
                            "parameter_name": token.text,
                            "description": token.sent.text,
                            "required": any(req in token.sent.text.lower() for req in ["required", "must", "necessary"]),
                            "command": command_context
                        })

            # Second pass with LLM for validation and enrichment
            llm_prompt = """Based on the following text, identify all command parameters, arguments, and inputs:
            - Parameter name/identifier
            - Description/purpose
            - Whether it's required
            - Associated command (if any)
            - Default value (if any)
            Format as a clear list.
            """
            
            llm_extracted = self.extract_llm_section(text, llm_prompt)
            combined_inputs = self._merge_input_sources(inputs_args, llm_extracted)
            
            return combined_inputs

        except Exception as e:
            self.logger.error(f"Error extracting inputs and arguments: {str(e)}")
            raise

    def _get_command_context(self, sentence):
        """Extract command context from a sentence."""
        doc = self.nlp(sentence)
        for token in doc:
            if token.text.startswith("ncc") or token.text.startswith("ncli") or token.text.startswith("cluster"):
                return token.text
        return None

    def _merge_input_sources(self, spacy_inputs, llm_inputs):
        """Merge and deduplicate inputs from spaCy and LLM."""
        # Convert LLM text to structured format
        llm_structured = self._parse_llm_inputs(llm_inputs)
        
        # Combine and deduplicate
        merged = {input_item["parameter_name"]: input_item for input_item in spacy_inputs}
        for llm_input in llm_structured:
            if llm_input["parameter_name"] not in merged:
                merged[llm_input["parameter_name"]] = llm_input
            else:
                # Enrich existing entry with LLM data
                merged[llm_input["parameter_name"]].update(llm_input)
        
        return list(merged.values())

    def _parse_llm_inputs(self, llm_text):
        """Parse LLM output into structured format."""
        # Implementation would depend on the exact format of LLM output
        # This is a placeholder for the parsing logic
        structured_inputs = []
        # Parsing logic here
        return structured_inputs

    def extract_llm_section(self, text, prompt):
        """
        Extracts specific information from the text using the LLM pipeline.

        Args:
            text (str): The text to analyze.
            prompt (str): The prompt instructing the LLM on what to extract.

        Returns:
            str: The response from the LLM.
        """
        try:
            # Combine the prompt with the input text
            full_input = prompt.format(text=text)

            # Generate response using the LLM pipeline
            response = self.pipeline(full_input)[0]["generated_text"]
            return response
        except Exception as e:
            self.logger.error(f"Error extracting LLM section: {str(e)}")
            raise




    def parse_configuration(self, text):
        """Parse text into structured YAML sections."""
        try:
            config = {
                "knowledge_base": {
                    "source_file": os.path.basename(self.pdf_path),
                    "extraction_date": date.today().isoformat(),
                },
                "system_configuration": self.extract_system_configuration(text),
                "inputs_arguments": self.extract_inputs_arguments(text),
                "actions": self.extract_actions(text),
                "sequences": self.extract_llm_section(
                    text, 
                    "What are the main sequences or steps outlined in this document? List them in chronological order."
                ),
                "validations": self.extract_llm_section(
                    text, 
                    "What are the validation steps or checks described in this document? List them clearly."
                ),
                "dependencies": self._extract_dependencies(text),
                "commands": self._extract_commands(text),
                "alerts": self._extract_alerts(text),
                "references": self._extract_references(text)
            }

            self.logger.info("Configuration parsed successfully")
            return config

        except Exception as e:
            self.logger.error(f"Error parsing configuration: {str(e)}")
            raise

    # Rest of the class implementation remains the same...

    def save_to_yaml(self, config):
        """Save configuration to YAML file."""
        try:
            base_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
            output_file = os.path.join(self.output_dir, f"{base_name}_config.yaml")

            with open(output_file, "w") as file:
                yaml.dump(config, file, default_flow_style=False)

            return output_file

        except Exception as e:
            self.logger.error(f"Error saving YAML: {str(e)}")
            raise

    def convert(self):
        """Complete conversion process."""
        try:
            text = self.extract_pdf_text()
            config = self.parse_configuration(text)
            yaml_path = self.save_to_yaml(config)
            return config

        except Exception as e:
            self.logger.error(f"Error in conversion process: {str(e)}")
            raise
    def extract_pdf_text(self):
        """Extract text from the given PDF file."""
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

def main():
    try:
        pdf_files = [
            "/Users/nandish.chokshi/Downloads/NLP-Based-KBA-Hackathon-/data/raw/KB-1.pdf",
            "/Users/nandish.chokshi/Downloads/NLP-Based-KBA-Hackathon-/data/raw/KB-2.pdf",
        ]
        output_dir = "./output_yaml"

        for pdf_file in pdf_files:
            converter = KBArticleConverter(pdf_file, output_dir)
            config = converter.convert()
            print(f"Processed {os.path.basename(pdf_file)}: Configuration extracted successfully.")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        logging.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
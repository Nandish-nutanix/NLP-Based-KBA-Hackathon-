import os
import yaml
import PyPDF2
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datetime import date
import logging
import torch

class KBArticleConverter:
    def __init__(self, pdf_path, output_dir="./output_yaml", log_file="conversion.log"):
        """
        Initialize the converter with a PDF path and logging setup.

        Args:
            pdf_path (str): Full path to the PDF file.
            output_dir (str): Directory to save output YAML files.
            log_file (str): Path for the log file.
        """
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
            # Load the LLaMA model with updated authentication
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                token=os.getenv('HUGGINGFACE_TOKEN')  # Using token instead of use_auth_token
            )
            
            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                torch_dtype=torch.float16,  # Use half precision to save memory
                token=os.getenv('HUGGINGFACE_TOKEN')  # Using token instead of use_auth_token
            )

            # Create pipeline without specifying device
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.95
            )
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def extract_pdf_text(self):
        """
        Extract text from PDF file.

        Returns:
            str: Extracted text from PDF.
        """
        try:
            with open(self.pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                full_text = ""

                for page in reader.pages:
                    full_text += page.extract_text() + "\n"

                self.logger.info("PDF text extraction complete.")
                return full_text
        except Exception as e:
            self.logger.error(f"Error extracting PDF text: {str(e)}")
            raise

    def extract_section(self, text, prompt):
        """
        Extract specific section details using LLaMA model.

        Args:
            text (str): The input text to analyze.
            prompt (str): The specific question or task.

        Returns:
            str: Extracted answer or section from the text.
        """
        try:
            # Truncate input text if too long
            max_context_length = 2048
            if len(text) > max_context_length:
                text = text[:max_context_length]

            input_prompt = f"""Based on the following context, {prompt}
            Please provide a clear and concise response.

            Context:
            {text}

            Response:"""

            response = self.pipeline(
                input_prompt,
                do_sample=True,
                num_return_sequences=1,
                clean_up_tokenization_spaces=True
            )
            
            # Extract the generated text after the prompt
            generated_text = response[0]['generated_text']
            answer = generated_text[len(input_prompt):].strip()
            
            return answer

        except Exception as e:
            self.logger.error(f"Error in section extraction: {str(e)}")
            raise

    def parse_configuration(self, text):
        """Parse text into structured YAML sections."""

        try:
            config = {
                "system_configuration": {
                    "nodes": self.extract_section(
                        text, 
                        "Provide a structured list of nodes, including IP addresses, usernames, and descriptions (e.g., Primary Controller VM, Secondary Controller VM)."
                    ),
                    "clusters": self.extract_section(
                        text, 
                        "List the clusters mentioned, including their names and descriptions (e.g., Primary Cluster running NCC health checks, Backup Cluster for failover)."
                    ),
                },
                "actions": self.extract_section(
                    text, 
                    "List the actions or operations mentioned, including names, descriptions, and associated commands. Ensure commands are listed sequentially."
                ),
                "sequence": self.extract_section(
                    text, 
                    "Outline the sequence of tasks or operations in a structured format. Include names, descriptions, and chronological tasks."
                ),
                "dependencies": self.extract_section(
                    text, 
                    "Describe the dependencies mentioned, mapping tasks to their prerequisites or required steps."
                ),
                "validations": self.extract_section(
                    text, 
                    "List the validation steps described, including the expected results and detailed validation steps for each action."
                ),
            "commands": self.extract_section(
                text,
                "Provide a list of all commands mentioned in the document in the order they should be executed."
            ),
                }

            self.logger.info("Configuration parsed successfully")
            return config

        except Exception as e:
            self.logger.error(f"Error parsing configuration: {str(e)}")
            raise

    def save_to_yaml(self, config):
        """
        Save configuration to YAML file with clean formatting.
        """
        try:
            base_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
            output_file = os.path.join(self.output_dir, f"{base_name}_config.yaml")

            # Clean and structure the YAML data
            cleaned_config = {
                "system_configuration": {
                    "nodes": self.clean_text(config.get("system_configuration", {}).get("nodes", "")),
                    "clusters": self.clean_text(config.get("system_configuration", {}).get("clusters", "")),
                },
                "actions": self.clean_text(config.get("actions", "")),
                "sequence": self.clean_text(config.get("sequence", "")),
                "dependencies": self.clean_text(config.get("dependencies", "")),
                "validations": self.clean_text(config.get("validations", "")),
                "commands": self.clean_text(config.get("commands", "")),
            }

            with open(output_file, "w") as file:
                yaml.dump(cleaned_config, file, default_flow_style=False, sort_keys=False, width=80)

            self.logger.info(f"Configuration saved to {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Error saving YAML: {str(e)}")
            raise

    def clean_text(self, text):
        """
        Clean redundant and unnecessary text content.
        Args:
            text (str): Input text to clean.
        Returns:
            str: Cleaned and concise text.
        """
        try:
            lines = text.split("\n")
            unique_lines = []
            seen = set()
            for line in lines:
                stripped_line = line.strip()
                if stripped_line and stripped_line not in seen:
                    unique_lines.append(stripped_line)
                    seen.add(stripped_line)
            return "\n".join(unique_lines)
        except Exception as e:
            self.logger.error(f"Error cleaning text: {str(e)}")
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

def main():
    try:
        # Specify PDF files and output directory
        pdf_files = [
            "/Users/nandish.chokshi/Downloads/NLP-Based-KBA-Hackathon-/data/raw/KB-1.pdf",
            "/Users/nandish.chokshi/Downloads/NLP-Based-KBA-Hackathon-/data/raw/KB-2.pdf",
        ]
        output_dir = "./output_yaml"

        # Process each PDF file
        for pdf_file in pdf_files:
            converter = KBArticleConverter(pdf_file, output_dir)
            config = converter.convert()
            print(f"Processed {os.path.basename(pdf_file)}: Configuration extracted successfully.")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        logging.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
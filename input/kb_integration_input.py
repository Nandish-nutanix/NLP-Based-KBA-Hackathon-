import os
import yaml
import PyPDF2
import spacy
from datetime import date
import logging
import re

class KBArticleConverter:
    def __init__(self, pdf_path, log_file="conversion.log"):
        self.pdf_path = pdf_path

        # Initialize logging
        logging.basicConfig(filename=log_file, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("KBArticleConverter initialized.")

        # Load SpaCy NLP model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            from spacy.cli import download
            download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

    def extract_pdf_text(self):
        with open(self.pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
            logging.info("PDF text extracted successfully.")
            return full_text.strip()

    def parse_inputs_arguments(self, text):
        config = {
            'knowledge_base': {
                'source_file': self.pdf_path,
                'extraction_date': date.today().isoformat()
            },
            'inputs_arguments': [],
            'inputs_count': 0  # Add field for count of inputs
        }

        # Patterns
        patterns = {
            'ip_pattern': r'\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:xx\.){3}xx\b',  # IP addresses or placeholders
            'kb_pattern': r'\bKB[-_ ]?\d+\b',  # KB references
            'precheck_pattern': r'test_hosts_in_maintenance_mode',  # Precheck term
            'error_pattern': r'\b(error|failure|exception|issue|problem|ERR_\d{4}|Code: \d+)\b',  # Errors
            'file_path_pattern': r'/[\w/.-]+',  # File paths
            'maintenance_pattern': r'\b(recovery|maintenance|backup|restore|failover|repair)\b'  # Maintenance terms
        }

        # Extract matches for each pattern
        matches = {key: re.findall(pattern, text) for key, pattern in patterns.items()}

        # Combine inputs and ensure uniqueness
        combined_inputs = []
        for key, values in matches.items():
            combined_inputs.extend(values)

        # Remove duplicates and update count
        unique_inputs = list(set(combined_inputs))
        config['inputs_arguments'] = unique_inputs
        config['inputs_count'] = len(unique_inputs)  # Store the count of unique inputs

        logging.info(f"Inputs and arguments parsed successfully. Total unique inputs: {config['inputs_count']}")
        return config

    def save_to_yaml(self, config, output_path="/home/rangu.ushasri/src/main/views/hack1/NLP-Based-KBA-Hackathon-/input/KB1_Config.yaml"):
        """
        Save configuration to YAML file.
        
        Args:
            config (dict): Configuration dictionary.
            output_path (str): Path to save YAML file.
        
        Returns:
            str: Path to saved YAML file.
        """
        with open(output_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        logging.info(f"Configuration saved to {output_path}.")
        return output_path

    def convert(self):
        """
        Complete conversion process.
        
        Returns:
            dict: Extracted configuration.
        """
        # Extract text
        text = self.extract_pdf_text()

        # Parse inputs and arguments
        config = self.parse_inputs_arguments(text)

        # Save to YAML
        self.save_to_yaml(config)

        return config


def main():
    # Path to PDF
    pdf_path = "/home/rangu.ushasri/src/main/views/hack1/NLP-Based-KBA-Hackathon-/data/raw/KB-1.pdf"

    # Initialize converter
    converter = KBArticleConverter(pdf_path)

    # Perform conversion
    config = converter.convert()

    # Output summary
    print("Conversion complete!")
    print(f"Inputs/Arguments: {config['inputs_arguments']}")
    print(f"Total Input/Arguments: {config['inputs_count']}")


if __name__ == "__main__":
    main()

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

    def parse_validations_and_dependencies(self, text):
        config = {
            'knowledge_base': {
                'source_file': self.pdf_path,
                'extraction_date': date.today().isoformat()
            },
            'validations': [],
            'dependencies': [],
        }

        # Common patterns for validations and dependencies
        validation_patterns = [
            r'(?i)(check|verify|ensure|validate)\s*[\w\s]+',
            r'\b(Precheck|Test)[\w\s]+',
            r'\b(error|failure|exception|issue)\b',  # Errors are a form of validation
        ]
        
        dependency_patterns = [
            r'\b(dependency|requirement|pre-requisite|needs)\b',  # Common dependency terms
            r'\b(related|linked|requires)\b',  # Related terms for dependencies
        ]

        # Extract matches for validations and dependencies
        validations = []
        for pattern in validation_patterns:
            validations.extend(re.findall(pattern, text))
        
        dependencies = []
        for pattern in dependency_patterns:
            dependencies.extend(re.findall(pattern, text))

        # Remove duplicates
        config['validations'] = list(set(validations))
        config['dependencies'] = list(set(dependencies))

        logging.info(f"Validations parsed successfully. Found {len(config['validations'])} validations.")
        logging.info(f"Dependencies parsed successfully. Found {len(config['dependencies'])} dependencies.")
        return config

    def save_to_yaml(self, config, output_path="/home/sravya.venkamsetty/NLP-Based-KBA-Hackathon-/dependencies-validations/KB-2_config.yaml"):
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

        # Parse validations and dependencies
        config = self.parse_validations_and_dependencies(text)

        # Save to YAML
        self.save_to_yaml(config)

        return config


def main():
    # Path to PDF
    pdf_path = "/home/sravya.venkamsetty/NLP-Based-KBA-Hackathon-/data/raw/KB-2.pdf"

    # Initialize converter
    converter = KBArticleConverter(pdf_path)

    # Perform conversion
    config = converter.convert()

    # Output summary
    print("Conversion complete!")
    print(f"Validations: {config['validations']}")
    print(f"Dependencies: {config['dependencies']}")


if __name__ == "__main__":
    main()
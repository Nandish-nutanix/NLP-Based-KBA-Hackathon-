import paramiko
import json
import logging
import os

FILE_PATH = '/Users/sachinappasaheb.b/lcm/hackathon/NLP-Based-KBA-Hackathon-/execution/output_view.json'
LOGGING_FILE = '/Users/sachinappasaheb.b/lcm/hackathon/NLP-Based-KBA-Hackathon-/execution/commands_execution.log'

def load_config_from_json(file_path):
    """Load configuration from the JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def setup_logging(log_file, log_level):
    """Set up logging configuration."""
    log_level_dict = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logging.basicConfig(filename=log_file, 
                        level=log_level_dict.get(log_level, logging.INFO),
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging initialized. Log level set to %s.", log_level)

def execute_ssh_command(ssh_client, command):
    """Execute a single SSH command and capture output."""

    print(ssh_client, command)
    stdin, stdout, stderr = ssh_client.exec_command('bash -l -c "%s"' % command)
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')
    
    if output:
        logging.info("Command output:\n%s", output)
    if error:
        logging.error("Command error:\n%s", error)

def execute_commands_on_node(node_ip, username, log_file, commands):
    """SSH into a node and execute a sequence of commands."""
    try:
        # Connect to the VM
        logging.info("Connecting to node at IP: %s", node_ip)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Automatically add host keys
        ssh.connect(node_ip, username=username, password='nutanix/4u')

        # Execute each command in the list
        for command in commands:
            logging.info("Executing command: %s", command)
            execute_ssh_command(ssh, command)

        ssh.close()
        logging.info("SSH connection closed for node: %s", node_ip)
    
    except Exception as e:
        logging.error("Error while executing commands on node %s: %s", node_ip, str(e))

def main():
    # Load the configuration from the JSON file
    json_file_path = FILE_PATH  # Path to your JSON file
    config = load_config_from_json(json_file_path)

    # Setup logging configuration
    log_file = LOGGING_FILE #config['logging']['log_file']
    log_level = config['logging']['log_level']
    setup_logging(log_file, log_level)

    # Extract the list of commands to execute
    all_commands = config['commands']['ordered_commands']

    # Loop through each node and execute the commands
    for node_config in config['system_configuration']['nodes']:
        for node_name, node_details in node_config.items():
            node_ip = node_details['ip']
            username = node_details['username']
            description = node_details['description']
            
            logging.info("Starting commands for node '%s' (%s)", node_name, description)
            execute_commands_on_node(node_ip, username, log_file, all_commands)

if __name__ == "__main__":
    main()
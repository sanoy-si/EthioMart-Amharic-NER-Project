import pandas as pd
import os
import sys
import argparse

LABELS = ["O", "B-Product", "I-Product", "B-LOC", "I-LOC", "B-PRICE", "I-PRICE"]

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_label_menu():
    menu = {str(i): label for i, label in enumerate(LABELS)}
    return menu

def print_label_menu(menu):
    print("--- Entity Labels ---")
    for number, label_name in menu.items():
        print(f"  [{number}] : {label_name}")
    print("--------------------")

def label_data(input_file, output_file, num_samples):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        sys.exit(1) # Exit the script if the file doesn't exist

    if 'clean_text' not in df.columns:
        print("Error: 'clean_text' column not found in the input file.")
        sys.exit(1)
        
    messages_to_label = df['clean_text'].dropna().sample(n=num_samples, random_state=42).tolist()
    
    label_menu = get_label_menu()
    all_conll_lines = []

    for i, message in enumerate(messages_to_label):
        clear_screen()
        print(f"--- Labeling Message {i + 1} of {num_samples} ---")
        
        tokens = message.split()
        current_message_labels = []
        
        for token in tokens:
            print(f"\nMessage: {message}\n")
            print_label_menu(label_menu)
            print(f"Current Token: '{token}'")
            
            choice = None
            while choice not in label_menu:
                prompt = f"Enter label number for '{token}': "
                choice = input(prompt)
                if choice not in label_menu:
                    print("Invalid choice. Please enter a number from the menu.")
            
            selected_label = label_menu[choice]
            current_message_labels.append(f"{token} {selected_label}")

        all_conll_lines.extend(current_message_labels)
        all_conll_lines.append("")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_conll_lines))
        clear_screen()
        print("Labeling complete!")
        print(f"Labeled data for {num_samples} messages saved to: {output_file}\n")
    except IOError as e:
        print(f"Error: Could not write to output file '{output_file}'.\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A CLI tool for labeling text data for NER tasks in CoNLL format."
    )
    
    parser.add_argument(
        "-i", "--input-file",
        required=True,
        help="Path to the input CSV file containing the text messages."
    )
    parser.add_argument(
        "-o", "--output-file",
        required=True,
        help="Path to save the labeled output CoNLL file."
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=30,
        help="The number of messages to label from the input file. (Default: 30)"
    )
    
    args = parser.parse_args()
    
    label_data(args.input_file, args.output_file, args.num_samples)
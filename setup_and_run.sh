#!/bin/bash

# Set text colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Function to create directories
create_directories() {
    echo -e "${GREEN}Creating necessary directories...${NC}"
    
    directories=(
        "data/raw"
        "data/processed"
        "data/images"
        "data/cache"
        "logs/tensorboard"
        "models"
        "explanations"
        "config"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo -e "${CYAN}Created directory: $dir${NC}"
        fi
    done
}

# Function to set up virtual environment
setup_venv() {
    if [ ! -d "venv" ]; then
        echo -e "${GREEN}Creating Python 3.11 virtual environment...${NC}"
        python3.11 -m venv venv
    fi
    
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source venv/bin/activate
    
    echo -e "${GREEN}Installing dependencies from requirements.txt...${NC}"
    pip install -r requirements.txt
}

# Check if src directory exists
check_src() {
    if [ ! -d "src" ]; then
        echo -e "${YELLOW}Warning: 'src' directory not found. The pipeline may not run correctly.${NC}"
    fi
}

# Display menu for pipeline options
show_menu() {
    echo -e "\n${MAGENTA}========= Fake News Detection Pipeline =========${NC}"
    echo "1. Run Data Preprocessing"
    echo "2. Train Model"
    echo "3. Evaluate Model"
    echo "4. Generate Visualizations"
    echo "5. Run Complete Pipeline"
    echo "6. Exit"
    echo -e "${MAGENTA}===============================================${NC}"
    echo -ne "${GREEN}Enter your choice (1-6): ${NC}"
}

# Run the selected option
run_option() {
    case $1 in
        1)
            echo -e "\n${CYAN}Running Data Preprocessing...${NC}"
            python3.11 -c "from src.data.dataset import DatasetProcessor; import yaml; with open('config/config.yaml', 'r') as f: config = yaml.safe_load(f); processor = DatasetProcessor(config); processor.process_datasets(); processor.combine_datasets(); processor.preprocess_dataset()"
            ;;
        2)
            echo -e "\n${CYAN}Running Model Training...${NC}"
            python3.11 -m src.main --train
            ;;
        3)
            echo -e "\n${CYAN}Running Model Evaluation...${NC}"
            python3.11 -m src.main --evaluate
            ;;
        4)
            echo -e "\n${CYAN}Generating Visualizations...${NC}"
            python3.11 -m src.main --explain
            ;;
        5)
            echo -e "\n${CYAN}Running Complete Pipeline...${NC}"
            python3.11 -m src.main --all
            ;;
        6)
            echo -e "\n${YELLOW}Exiting...${NC}"
            exit 0
            ;;
        *)
            echo -e "\n${RED}Invalid option. Please try again.${NC}"
            ;;
    esac
}

# Main execution
main() {
    create_directories
    setup_venv
    check_src
    
    while true; do
        show_menu
        read choice
        
        if [[ "$choice" =~ ^[1-6]$ ]]; then
            run_option $choice
            if [ $choice -eq 6 ]; then
                break
            fi
        else
            echo -e "${RED}Invalid input. Please enter a number between 1 and 6.${NC}"
        fi
        
        echo -e "\n${GREEN}Press Enter to continue...${NC}"
        read
    done
}

# Run the main function
main 
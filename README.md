# Well Spacing Analyzer

A high-performance Python tool designed to compute optimal spacing between parent and child wells in unconventional reservoirs. It processes directional survey and header data from Excel, CSV, or IHS sources, delivering results for large datasets (e.g., ~20,000 wells in the Midland Basin) in under 15 minutes.

## Features

- **Flexible Data Input**: Accepts directional survey and header data from Excel, CSV, or IHS formats.
- **Efficient Processing**: Handles large datasets with optimized performance.
- **Automated Spacing Calculations**: Computes horizontal/vertical/3D distances, well-to-well directionality, and normalized lateral midpoints.
- **Visualization Integration**: Outputs compatible with Spotfire for rapid gun barrel dashboard creation.

## Repository Structure

```
├── data/
│   └── external/           # Sample input files
├── notebooks/              # Jupyter notebooks for testing and analysis
├── src/                    # Core Python modules
│   ├── spacing_calculator.py
│   └── utils.py
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/apoorvasaxena03/well-spacing-analyzer.git
   cd well-spacing-analyzer
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate     # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare Input Files**:
   - Directional survey and header files in `.csv` or `.xlsx` format.
   - Input should include UWI/API, measured depth, latitude/longitude or surface location, and TVD.

2. **Run the Spacing Calculation Script**:
   ```bash
   python src/spacing_calculator.py --input data/external/midland.csv --output results/midland_spacing.csv
   ```

3. **Load into Spotfire**:
   - Import the spacing output into Spotfire to generate a gun barrel dashboard for fast visual interpretation.

## Example

```bash
python src/spacing_calculator.py --input data/external/midland.csv --output results/midland_spacing.csv
```

## License

© 2025 Apoorva Saxena. This repository is shared for **viewing purposes only**. Redistribution, modification, or commercial use is prohibited without written permission.

## Author

**Apoorva Saxena**  
Reservoir Engineer  
[LinkedIn](https://www.linkedin.com/in/apoorvasaxena)  
[GitHub](https://github.com/apoorvasaxena03)

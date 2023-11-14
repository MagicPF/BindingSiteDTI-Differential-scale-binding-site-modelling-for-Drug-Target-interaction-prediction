
# BindingSiteDTI

## Introduction
BindingSiteDTI is a cutting-edge software tool developed for conducting sophisticated experiments on diverse datasets in the realm of Drug-Target Interactions (DTIs). Its primary objective is to streamline the analysis process of complex data, thereby significantly contributing to research in drug design and discovery.

## Environment Setup
Prior to running BindingSiteDTI, it is essential to set up the environment by installing all necessary dependencies. This can be efficiently done through the provided `requirements.txt` file.

### Installation Instructions
Execute the following command in your terminal to install the dependencies:

```bash
pip install -r requirements.txt
```

This command facilitates the automatic installation of all required packages and libraries, ensuring the software operates seamlessly.

## Usage Guide
BindingSiteDTI is highly versatile, capable of supporting a variety of datasets for experimentation. The tool automatically initiates data preprocessing if processed data packages are not detected. We offer convenient one-command scripts for each experiment, as follows:


### Experiment on the Human Cold Dataset
Run the following script in the terminal to conduct experiments on the human_cold dataset:

```bash
bash human.sh
```

### Experiment on the DUDE Dataset
To perform experiments on the DUDE dataset, use this command:

```bash
bash DUDE.sh
```

### Experiment on the BindingDB Dataset
BindingSiteDTI is equipped to handle two subsets of the BindingDB dataset. Utilize the respective scripts for these experiments:

- For the BindingDB 'seen' subset:
  ```bash
  bash BindingDB_seen.sh
  ```

- For the BindingDB 'unseen' subset:
  ```bash
  bash BindingDB_unseen.sh
  ```
### Preprocessed Dataset
For those who find data preprocessing to be time-consuming, we recommend downloading the preprocessed dataset from the following link:

[Download Preprocessed Dataset](https://lifehkbueduhk-my.sharepoint.com/:f:/g/personal/22481087_life_hkbu_edu_hk/EnTHROotTA9EgyUQWeQ2DC8BWDuvAXpj3GbBLFmvjvwFTg?e=H2AaFA)

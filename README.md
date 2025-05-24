# Multi Modal 3D Object Detection in Adverse Weather Conditions

## Overview

This repository contains the implementation of a cutting-edge approach for **Multi-Modal 3D Object Detection in Adverse Weather Conditions**. The project leverages data from multiple sensor modalities, such as LiDAR, and cameras, to achieve robust and accurate 3D object detection in challenging environments. By addressing the limitations of single-sensor systems, this work aims to enhance the reliability of autonomous systems operating in real-world scenarios.

<p align="center">
     <img src="C:\Users\matha\Desktop\Source_codes\sereact\sereact\sereact\media\rgb.jpg" alt="Image 1" width="200"/>
     <img src="C:\Users\matha\Desktop\Source_codes\sereact\sereact\sereact\media\pcd_withbbox3D.png" alt="Image 2" width="200"/>
     <img src="C:\Users\matha\Desktop\Source_codes\sereact\sereact\sereact\media\instance_mask.png" alt="Image 3" width="200"/>
   </p>

## Features

- **Multi-Modal Fusion**: Combines data from LiDAR, and cameras for improved detection accuracy.
- **Adverse Weather Robustness**: Designed to handle challenging conditions such as rain, fog, and snow.
- **State-of-the-Art Performance**: Implements advanced deep learning architectures for 3D object detection.
- **Scalable and Modular**: Easily adaptable to different datasets and sensor configurations.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Bharadhwajsaimatha/Multi-Modal-3D-Object-Detection-in-Adverse-Weather-Conditions.git
    cd BBox3DDetection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the dataset:
    - Download the dataset from [Dataset Source](#) and place it in the `data/` directory.
    - Follow the preprocessing instructions in `docs/preprocessing.md`.(Will be updated soon)

## Usage

### Training
To train the model, run:
```bash
python train.py --config configs/config.yaml
```

### Evaluation
To evaluate the model on the test set, use:
```bash
python evaluate.py --checkpoint checkpoints/model.pth
```

### Inference
For inference on new data:
```bash
python inference.py --input data/sample_input --output results/
```

## Dataset

This project supports multiple datasets, including:

- **CARLA Infrastructure Dataset**: Generated using the CARLA simulator, this dataset leverages infrastructure-based sensors such as roadside LiDAR and cameras. It provides diverse scenarios, including urban traffic, intersections, and adverse weather conditions, making it ideal for testing autonomous systems in controlled yet realistic environments.
- **Shopping Cart Dataset**: Captured using a custom setup mounted on a shopping cart, this dataset includes indoor and outdoor retail environments. It features challenging scenarios such as narrow aisles, dynamic obstacles, and varying lighting conditions, enabling robust object detection in retail applications.
- **Custom Datasets**: Easily integrate your own dataset by following the format guidelines in `docs/dataset_format.md`.(updated soon)

## Model Architecture

The model employs a multi-modal fusion strategy, combining:
- **LiDAR Point Clouds**: For precise spatial information.
- **Camera Images**: For rich texture and color information.

The fusion is achieved through a hybrid deep learning architecture that integrates convolutional neural networks (CNNs) and transformer-based modules.

## Results

Yet to be updated with quantitative results and qualitative examples. The model has shown promising performance in preliminary tests, achieving high accuracy in detecting 3D objects under various weather conditions.

## Contributing

We welcome contributions to improve this project. Please follow the guidelines in `CONTRIBUTING.md` and ensure that all code adheres to the style guide.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We would like to thank the contributors of the open-source datasets and libraries used in this project. Special thanks to the research community for their valuable insights and inspiration.

## ToDO

- [ ] Update dataset preprocessing instructions.
- [ ] Add quantitative results and performance metrics for 3D detection.
- [ ] Implement the denoising autoencoder for adverse weather conditions.
- [ ] Explore integration with real-world autonomous systems.
- [ ] Conduct extensive testing across various datasets and scenarios.
- [ ] Optimize model performance for real-time applications.

## Contact

For questions or collaboration opportunities, please contact:
- **Name**: Sai Bharadhwaj Matha
- **Email**: bharadhwaj2299@gmail.com
- **GitHub**: [Bharadhwaj](https://github.com/Bharadhwajsaimatha)

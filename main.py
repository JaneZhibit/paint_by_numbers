from pipeline import PaintPipeline

imgs = [
    "test_images/масштаб 4 к 3/animal_1.jpg",
    "test_images/масштаб 4 к 3/animal_2.jpg",
    "test_images/масштаб 4 к 3/animal_3.jpg",
    "test_images/масштаб 4 к 3/animal_4.jpg",
    "test_images/масштаб 4 к 3/animal_5.jpg",
    "test_images/масштаб 4 к 3/landscape_1.jpg",
    "test_images/масштаб 4 к 3/landscape_2.jpg",
    "test_images/масштаб 4 к 3/landscape_3.jpg",
    "test_images/масштаб 4 к 3/landscape_4.jpg",
    "test_images/масштаб 4 к 3/landscape_5.jpg",
    "test_images/масштаб 4 к 3/naturmort_1.jpg",
    "test_images/масштаб 4 к 3/naturmort_2.jpg",
    "test_images/масштаб 4 к 3/naturmort_3.jpg",
    "test_images/масштаб 4 к 3/naturmort_4.jpg",
    "test_images/масштаб 4 к 3/naturmort_5.jpg",
    "test_images/масштаб 4 к 3/portret_1.jpg",
    "test_images/масштаб 4 к 3/portret_2.jpg",
    "test_images/масштаб 4 к 3/portret_3.jpg",
    "test_images/масштаб 4 к 3/portret_4.jpg",
    "test_images/масштаб 4 к 3/portret_5.jpg",
]

custom_config = {
        "canvas": {
            "width_mm": 300,
            "height_mm": 400,
            "min_brush_mm": 3,
        },
        "preprocessing": {
            "smoothing": {
                "method": "bilateral",
                "params": {
                    "mean_shift": {
                        "spatial_radius": 15,
                        "color_radius": 30
                    },
                }
            },
            "contrast": {
                "method": "clahe",
                "params": {
                    "clahe": {
                        "clip_limit": 0.7
                    }
                }
            }
        },
        "quantization": {
            "method": "kmeans",
            "colors_count": 20,
            "color_space": "lab"
        },
        "vectorization": {
            "scale_factor": 4,
            "blur_size": 31,
            "epsilon_factor": 0.001
        }
    }

for img_path in [r"C:/Users/Andrey/Downloads/мы.jpg"]:

    pipeline = PaintPipeline(user_config=custom_config)

    pipeline.load_image(img_path)
    pipeline.preprocess()
    pipeline.quantize()
    pipeline.postprocess()
    pipeline.vectorize()
    pipeline.render_all()

    #pipeline.show_stage("preprocessed", save_path=f'results/preproc2{img_path.split("/")[-1]}')
    #pipeline.show_stage("quantized", save_path=f'results/quant2{img_path.split("/")[-1]}')
    #pipeline.show_stage("postprocessed", save_path=f'results/waterpost{img_path.split("/")[-1]}')
    #pipeline.show_stage("vectorized", save_path=f'results/watervector{img_path.split("/")[-1]}')
    pipeline.show_stage("color_reference", save_path=f'results/color_ref{img_path.split("/")[-1]}')
    #pipeline.show_stage("numbered", save_path=f'results/numbered2{img_path.split("/")[-1]}')

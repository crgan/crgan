from .env import base_dir

config = {
    "UCSDped1": {
        "train_dataset": f"{base_dir}/UCSDped1/Train",
        "test_dataset": f"{base_dir}/UCSDped1/Test",
        "bbox_roots": f"{base_dir}/UCSDped1_BBOX/",
        "flow_roots": f"{base_dir}/UCSDped1_OPT/",
        "resolution": (240, 360),
        "extension": ".tif"
    },
    "UCSDped2": {
        "train_dataset": f"{base_dir}/UCSDped2/Train",
        "test_dataset": f"{base_dir}/UCSDped2/Test",
        "bbox_roots": f"{base_dir}/UCSDped2_BBOX/",
        "flow_roots": f"{base_dir}/UCSDped2_OPT/",
        "resolution": (240, 360),
        "extension": ".tif"
    },
    "Avenue": {
        "train_dataset": f"{base_dir}/Avenue/Train",
        "test_dataset": f"{base_dir}/Avenue/Test",
        "bbox_roots": f"{base_dir}/Avenue_BBOX/",
        "flow_roots": f"{base_dir}/Avenue_OPT/",
        "resolution": (120, 180),
        "extension": ".jpg"
    },
    "UMN1": {
        "train_dataset": f"{base_dir}/UMN1/Train",
        "test_dataset": f"{base_dir}/UMN1/Test",
        "bbox_roots": f"{base_dir}/UMN1_BBOX/",
        "flow_roots": f"{base_dir}/UMN1_OPT/",
        "resolution": (240, 320),
        "extension": ".jpg"
    },
    "UMN2": {
        "train_dataset": f"{base_dir}/UMN2/Train",
        "test_dataset": f"{base_dir}/UMN2/Test",
        "bbox_roots": f"{base_dir}/UMN2_BBOX/",
        "flow_roots": f"{base_dir}/UMN2_OPT/",
        "resolution": (240, 320),
        "extension": ".jpg"
    },
    "UMN3": {
        "train_dataset": f"{base_dir}/UMN3/Train",
        "test_dataset": f"{base_dir}/UMN3/Test",
        "bbox_roots": f"{base_dir}/UMN3_BBOX/",
        "flow_roots": f"{base_dir}/UMN3_OPT/",
        "resolution": (240, 320),
        "extension": ".jpg"
    },
    "UMN": {
        "train_dataset": f"{base_dir}/UMN/Train",
        "test_dataset": f"{base_dir}/UMN/Test",
        "bbox_roots": f"{base_dir}/UMN_BBOX/",
        "flow_roots": f"{base_dir}/UMN_OPT/",
        "resolution": (240, 320),
        "extension": ".jpg"
    },
    "UMN_split": {
        "train_dataset": f"{base_dir}/UMN_split/Train",
        "test_dataset": f"{base_dir}/UMN_split/Test",
        "bbox_roots": f"{base_dir}/UMN_split_BBOX/",
        "flow_roots": f"{base_dir}/UMN_split_OPT/",
        "resolution": (240, 320),
        "extension": ".jpg"
    },
    "ShanghaiTech": {
        "train_dataset": f"{base_dir}/ShanghaiTech/Train",
        "test_dataset": f"{base_dir}/ShanghaiTech/Test",
        "bbox_roots": f"{base_dir}/ShanghaiTech_BBOX/",
        "flow_roots": f"{base_dir}/ShanghaiTech_OPT/",
        "resolution": (120, 200),
        "extension": ".jpg"
    },
    "ShanghaiTech1": {
        "train_dataset": f"{base_dir}/ShanghaiTech1/Train",
        "test_dataset": f"{base_dir}/ShanghaiTech1/Test",
        "bbox_roots": f"{base_dir}/ShanghaiTech1_BBOX/",
        "flow_roots": f"{base_dir}/ShanghaiTech1_OPT/",
        "resolution": (120, 200),
        "extension": ".jpg"
    },
    "ShanghaiTechn": {
        "train_dataset": f"{base_dir}/ShanghaiTechn/Train",
        "test_dataset": f"{base_dir}/ShanghaiTechn/Test",
        "bbox_roots": f"{base_dir}/ShanghaiTechn_BBOX/",
        "flow_roots": f"{base_dir}/ShanghaiTechn_OPT/",
        "resolution": (120, 200),
        "extension": ".jpg"
    }
}
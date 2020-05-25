from .env import base_dir

config = {
    "UCSDped1": {
        "source_dir": f"{base_dir}/UCSDped1/",
        "target_dir": f"{base_dir}/UCSDped1_BBOX/",
        "resolution": (240, 360),
        "extension": ".tif",
        "train_nms_thresh": 0.3,
        "test_nms_thresh": 0.3,
        "train_score_thresh": 0.1,
        "test_score_thresh": 0.5,
        "train_len": 34,
        "test_len": 36
    },
    "UCSDped2": {
        "source_dir": f"{base_dir}/UCSDped2/",
        "target_dir": f"{base_dir}/UCSDped2_BBOX/",
        "resolution": (240, 360),
        "extension": ".tif",
        "train_nms_thresh": 0.3,
        "test_nms_thresh": 0.3,
        "train_score_thresh": 0.5,
        "test_score_thresh": 0.5,
        "train_len": 16,
        "test_len": 12
    },
    "Avenue": {
        "source_dir": f"{base_dir}/Avenue/",
        "target_dir": f"{base_dir}/Avenue_BBOX/",
        "resolution": (120, 180),
        "extension": ".jpg",
        "train_nms_thresh": 0.3,
        "test_nms_thresh": 0.3,
        "train_score_thresh": 0.6,
        "test_score_thresh": 0.6,
        "train_len": 16,
        "test_len": 16
    },
    "UMN1": {
        "source_dir": f"{base_dir}/UMN1/",
        "target_dir": f"{base_dir}/UMN1_BBOX/",
        "resolution": (240, 320),
        "extension": ".jpg",
        "train_nms_thresh": 0.3,
        "test_nms_thresh": 0.3,
        "train_score_thresh": 0.1,
        "test_score_thresh": 0.5,
        "train_len": 1,
        "test_len": 1
    },
    "UMN2": {
        "source_dir": f"{base_dir}/UMN2/",
        "target_dir": f"{base_dir}/UMN2_BBOX/",
        "resolution": (240, 320),
        "extension": ".jpg",
        "train_nms_thresh": 0.3,
        "test_nms_thresh": 0.3,
        "train_score_thresh": 0.1,
        "test_score_thresh": 0.5,
        "train_len": 1,
        "test_len": 1
    },
    "UMN3": {
        "source_dir": f"{base_dir}/UMN3/",
        "target_dir": f"{base_dir}/UMN3_BBOX/",
        "resolution": (240, 320),
        "extension": ".jpg",
        "train_nms_thresh": 0.3,
        "test_nms_thresh": 0.3,
        "train_score_thresh": 0.1,
        "test_score_thresh": 0.5,
        "train_len": 1,
        "test_len": 1
    },
    "UMN": {
        "source_dir": f"{base_dir}/UMN/",
        "target_dir": f"{base_dir}/UMN_BBOX/",
        "resolution": (240, 320),
        "extension": ".jpg",
        "train_nms_thresh": 0.3,
        "test_nms_thresh": 0.3,
        "train_score_thresh": 0.3,
        "test_score_thresh": 0.3,
        "train_len": 3,
        "test_len": 3
    },
    "UMN_split": {
        "source_dir": f"{base_dir}/UMN_split/",
        "target_dir": f"{base_dir}/UMN_split_BBOX/",
        "resolution": (240, 320),
        "extension": ".jpg",
        "train_nms_thresh": 0.3,
        "test_nms_thresh": 0.3,
        "train_score_thresh": 0.3,
        "test_score_thresh": 0.3,
        "train_len": 3,
        "test_len": 11
    },
    "ShanghaiTech": {
        "source_dir": f"{base_dir}/ShanghaiTech/",
        "target_dir": f"{base_dir}/ShanghaiTech_BBOX/",
        "resolution": (120, 200),
        "extension": ".jpg",
        "train_nms_thresh": 0.3,
        "test_nms_thresh": 0.3,
        "train_score_thresh": 0.1,
        "test_score_thresh": 0.5,
        "train_len": 330,
        "test_len": 107
    },
    "ShanghaiTech1": {
        "source_dir": f"{base_dir}/ShanghaiTech1/",
        "target_dir": f"{base_dir}/ShanghaiTech1_BBOX/",
        "resolution": (120, 200),
        "extension": ".jpg",
        "train_nms_thresh": 0.3,
        "test_nms_thresh": 0.3,
        "train_score_thresh": 0.1,
        "test_score_thresh": 0.5,
        "train_len": 10,
        "test_len": 10
    },
    "ShanghaiTechn": {
        "source_dir": f"{base_dir}/ShanghaiTechn/",
        "target_dir": f"{base_dir}/ShanghaiTechn_BBOX/",
        "resolution": (120, 200),
        "extension": ".jpg",
        "train_nms_thresh": 0.3,
        "test_nms_thresh": 0.3,
        "train_score_thresh": 0.1,
        "test_score_thresh": 0.5,
        "train_len": 84,
        "test_len": 107
    }
}
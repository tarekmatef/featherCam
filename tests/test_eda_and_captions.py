from pathlib import Path

from PIL import Image
import torch

from src.data.captions import (
    analyze_captions,
    clean_caption,
    detect_corpus_typos,
    preprocess_description_column,
    split_raw_description,
)
from src.data.cub200 import (
    CUB_EXTRACTED_FOLDER,
    HFCUB200Dataset,
    create_test_dataloader,
    denormalize_tensor,
    get_cub200_transforms,
)
from src.data.eda import run_eda
from collections import Counter


def test_caption_splitting_and_cleaning():
    raw_desc = (
        "This bird has brown feathres and a hooked beack.\r\n"
        "\r\n"
        "A grey bird with white wing-bars and yellow brest!\n"
        "this bird has brown feathers and a hooked beak.\n"
        "tiny\n"
    )
    caps, empty_cnt = split_raw_description(raw_desc)
    assert len(caps) == 4
    assert empty_cnt == 2

    cleaned, reps = clean_caption(caps[0])
    assert cleaned == "this bird has brown feathers and a hooked beak"
    assert ("feathres", "feathers") in reps
    assert ("beack", "beak") in reps

    cleaned_2, reps_2 = clean_caption(caps[1])
    assert "gray" in cleaned_2
    assert "wing bars" in cleaned_2
    assert "breast" in cleaned_2
    assert ("grey", "gray") in reps_2

    per_img, audit = preprocess_description_column([raw_desc], min_tokens=2, deduplicate_per_image=True)
    # caps[0] and caps[2] become identical after typo fix -> deduplicated; "tiny" (<2 tokens) dropped
    assert len(per_img[0]) == 2
    assert audit["dropped_intra_image_duplicates"] == 1
    assert audit["dropped_too_short"] == 1


def test_corpus_typo_detector_and_analyze_captions():
    vocab = Counter({"feathers": 80, "white": 120, "crown": 45, "fetaehrs": 1, "whiet": 2, "crowm": 1})
    suspected = detect_corpus_typos(vocab, max_rare_freq=2, min_anchor_freq=30)
    typo_words = {item["typo"]: item["suggested_word"] for item in suspected}
    assert typo_words.get("whiet") == "white"
    assert typo_words.get("crowm") == "crown"

    descriptions = [
        "\n".join([f"this bird has white feathers and black crown {i}." for i in range(10)]),
        "short\nthis grey bird has a long bill and brown feathres.",
    ]
    report, cleaned = analyze_captions(descriptions)
    assert report["num_records"] == 2
    assert report["captions_per_image"]["exact_10_raw_records"] == 1
    assert report["captions_per_image"]["non_10_raw_records_count"] == 1
    assert report["vocabulary_stats"]["cleaned_vocab_size"] <= report["vocabulary_stats"]["raw_vocab_size"]
    assert len(cleaned) == 2


def test_hf_dataset_augmentations_and_eda_pipeline(tmp_path: Path):
    shared_dup_img = Image.new("RGB", (180, 160), color=(40, 120, 200))
    gray_img = Image.new("L", (200, 220), color=128)

    mock_train_records = [
        {
            "id": 0,
            "file_name": "001.Black_footed_Albatross/alb_1.jpg",
            "image": shared_dup_img,
            "label": "Black footed Albatross",
            "description": "\n".join(
                [f"this bird has dark brown feathres and a long hooked beack {k}." for k in range(10)]
            ),
        },
        {
            "id": 1,
            "file_name": "001.Black_footed_Albatross/alb_2_gray.jpg",
            "image": gray_img,
            "label": "Black footed Albatross",
            "description": "\n".join(
                [f"aquatic grey bird with white belly and dark wings {k}." for k in range(10)]
            ),
        },
        {
            "id": 2,
            "file_name": "002.Laysan_Albatross/laysan_1.jpg",
            "image": Image.new("RGB", (260, 240), color=(210, 210, 200)),
            "label": "Laysan Albatross",
            "description": "\n".join(
                [f"large white seabird with black wings and pink bill {k}." for k in range(10)]
            ),
        },
        {
            "id": 3,
            "file_name": "002.Laysan_Albatross/laysan_2.jpg",
            "image": Image.new("RGB", (250, 210), color=(190, 205, 215)),
            "label": "Laysan Albatross",
            "description": "\n".join(
                [f"white bird with dark back and pale yellow bill {k}." for k in range(10)]
            ),
        },
    ]
    mock_test_records = [
        {
            "id": 10,
            "file_name": "001.Black_footed_Albatross/alb_test_dup.jpg",
            "image": shared_dup_img.copy(),
            "label": "Black footed Albatross",
            "description": "\n".join([f"brown seabird on water {k}." for k in range(10)]),
        },
        {
            "id": 11,
            "file_name": "002.Laysan_Albatross/laysan_test.jpg",
            "image": Image.new("RGB", (240, 240), color=(220, 220, 220)),
            "label": "Laysan Albatross",
            "description": "\n".join([f"white bird with dark wings {k}." for k in range(10)]),
        },
    ]

    for aug_mode in ("basic", "fine_grained", "strong"):
        tr_tf, va_tf = get_cub200_transforms(image_size=64, augmentation=aug_mode)
        ds = HFCUB200Dataset(
            train=True,
            transform=tr_tf,
            hf_dataset=mock_train_records,
            return_captions=True,
        )
        img_t, lbl, caps = ds[1]  # exercises grayscale -> RGB conversion + caption cleaning
        assert img_t.shape == (3, 64, 64)
        assert isinstance(lbl, int)
        assert len(caps) == 10
        rgb_back = denormalize_tensor(img_t)
        assert rgb_back.shape == (64, 64, 3)
        assert 0.0 <= float(rgb_back.min()) and float(rgb_back.max()) <= 1.0

    # Verify official test dataloader on local CUB layout
    cub_dir = tmp_path / CUB_EXTRACTED_FOLDER
    img_folder = cub_dir / "images" / "001.Black_footed_Albatross"
    img_folder.mkdir(parents=True)
    Image.new("RGB", (64, 64), color=(80, 90, 100)).save(img_folder / "t1.jpg")
    (cub_dir / "images.txt").write_text("1 001.Black_footed_Albatross/t1.jpg\n", encoding="utf-8")
    (cub_dir / "image_class_labels.txt").write_text("1 1\n", encoding="utf-8")
    (cub_dir / "train_test_split.txt").write_text("1 0\n", encoding="utf-8")

    test_loader = create_test_dataloader(
        {"data": {"source": "local", "data_dir": str(tmp_path), "image_size": 32, "num_workers": 0, "download": False}}
    )
    assert len(test_loader.dataset) == 1

    # End-to-end EDA + figure generation check
    eda_out = tmp_path / "eda_out"
    eda_figs = tmp_path / "eda_figs"
    eda_report = run_eda(
        cfg={
            "data": {"image_size": 64, "val_split": 0.5, "seed": 42},
            "eda": {
                "output_dir": str(eda_out),
                "figures_dir": str(eda_figs),
                "num_sample_images": 4,
                "num_augmentation_views": 3,
            },
        },
        hf_train_data=mock_train_records,
        hf_test_data=mock_test_records,
    )

    assert eda_report["image_statistics"]["non_rgb_count"] == 1
    assert eda_report["image_statistics"]["cross_split_duplicate_groups"] == 1
    assert (eda_out / "eda_report.json").is_file()
    assert (eda_out / "eda_summary.md").is_file()
    assert (eda_out / "cleaned_captions_train.json").is_file()
    for fig_path_str in eda_report["generated_figures"].values():
        fig_p = Path(fig_path_str)
        assert fig_p.is_file() and fig_p.stat().st_size > 0

